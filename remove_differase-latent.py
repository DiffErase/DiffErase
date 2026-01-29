"""Remove DiffErase watermark using DiffErase-latent (AudioLDM latent diffusion pipeline).
"""

import argparse
import sys
import time
import os
from pathlib import Path

import numpy as np
import torch
import torchaudio
import soundfile as sf
import yaml
from tqdm import tqdm

_script_dir = Path(__file__).resolve().parent
_diff_latent = _script_dir / "DiffErase-latent"
if str(_diff_latent) not in sys.path:
    sys.path.insert(0, str(_diff_latent))

from audioldm_train.utilities.audio.stft import TacotronSTFT
from audioldm_train.utilities.model_util import instantiate_from_config, get_vocoder

# Root directory for DiffErase-latent.
LATENT_ROOT = _diff_latent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove watermark using DiffErase-latent",
    )
    parser.add_argument(
        "--watermarked_dir",
        type=str,
        required=True,
        help="Directory containing watermarked .wav files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write watermarked audio after removal",
    )
    parser.add_argument(
        "--noise_level",
        type=float,
        required=True,
        help="Noise strength in [0, 1].",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="CUDA device index (e.g., 0,1,2,3).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="DiffErase-latent/data/checkpoints/*.ckpt",
        help="DiffErase-latent checkpoint path.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="DiffErase-latent/audioldm_train/config/*.yaml",
        help="DiffErase-latent config path.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--generator_name",
        type=str,
        default=None,
        help="Output filename suffix.",
    )

    return parser.parse_args()


def load_audio_file(file_path, sampling_rate=16000, duration=10.24):
    waveform, sr = torchaudio.load(file_path)

    if sr != sampling_rate:
        resampler = torchaudio.transforms.Resample(sr, sampling_rate)
        waveform = resampler(waveform)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    waveform = waveform.squeeze(0).numpy()
    if np.max(np.abs(waveform)) > 0:
        waveform = waveform / np.max(np.abs(waveform))

    target_length = int(sampling_rate * duration)
    if len(waveform) > target_length:
        waveform = waveform[:target_length]
    elif len(waveform) < target_length:
        pad_length = target_length - len(waveform)
        waveform = np.pad(waveform, (0, pad_length), mode="constant")

    return waveform, sampling_rate


def get_mel_spectrogram(waveform, stft, device):
    audio_tensor = torch.clamp(torch.FloatTensor(waveform).unsqueeze(0), -1, 1)
    audio_tensor = audio_tensor.to(device)

    mel_spec, magnitudes, phases, energy = stft.mel_spectrogram(audio_tensor)

    return mel_spec


def mel_to_waveform(mel_spec, vocoder, device, sampling_rate):
    mel_spec = mel_spec.to(device)
    mel_spec = mel_spec.permute(0, 2, 1)  # [B, t-steps, fbins] -> [B, fbins, t-steps]

    with torch.no_grad():
        wavs = vocoder(mel_spec).squeeze(1)  # [B, 1, T] -> [B, T]

    wav_np = (wavs.cpu().numpy() * 32768).astype(np.int16)

    return wav_np[0]


def process_audio_batch(
    wav_files_batch,
    model,
    stft,
    vocoder,
    noise_strength,
    device,
    output_dir,
    generator_name,
    sampling_rate,
    duration,
    batch_id=0,
    show_progress=False,
):
    try:
        waveforms = []
        valid_files = []

        if batch_id == 1 or len(wav_files_batch) <= 4:
            print(f"\n[Batch {batch_id}] Loading {len(wav_files_batch)} files...")
        
        for wav_file in wav_files_batch:
            try:
                waveform, sr = load_audio_file(wav_file, sampling_rate, duration)
                waveforms.append(waveform)
                valid_files.append(wav_file)
            except Exception as e:
                print(f"  ⚠ Warning: Failed to load {Path(wav_file).name}: {e}")
                continue

        if not waveforms:
            print(f"[Batch {batch_id}] No valid files to process")
            return 0

        mel_specs = []
        for waveform in waveforms:
            mel_spec = get_mel_spectrogram(waveform, stft, device)
            mel_specs.append(mel_spec)

        mel_batch = torch.cat(mel_specs, dim=0)

        if batch_id == 1 or show_progress:
            print(f"[Batch {batch_id}] Removing watermark (noise_strength={noise_strength})...")
        
        with torch.no_grad():
            mel_batch_input = mel_batch.permute(0, 2, 1).unsqueeze(1)

            model.eval()
            with model.ema_scope("Removing watermark with diffusion"):
                encoder_posterior = model.encode_first_stage(mel_batch_input)
                z0 = model.get_first_stage_encoding(encoder_posterior)

                num_timesteps = model.num_timesteps
                noise_timestep = int(num_timesteps * noise_strength)
                
                if batch_id == 1:
                    print(f"  Total timesteps: {num_timesteps}")
                    print(f"  Noise timestep: {noise_timestep}")
                    print(f"  Denoising steps: {noise_timestep}")

                t_noise = torch.full((z0.shape[0],), noise_timestep, device=device, dtype=torch.long)
                noise = torch.randn_like(z0)
                z_noisy = model.q_sample(x_start=z0, t=t_noise, noise=noise)

                cond = {}  # Empty dict for unconditional generation
                z_denoised = z_noisy
                
                if show_progress and batch_id == 1:
                    iterator = tqdm(range(noise_timestep, 0, -1), desc="  Denoising", leave=False)
                else:
                    iterator = range(noise_timestep, 0, -1)
                
                for t in iterator:
                    t_tensor = torch.full((z_denoised.shape[0],), t, device=device, dtype=torch.long)
                    z_denoised = model.p_sample(
                        x=z_denoised,
                        c=cond,
                        t=t_tensor,
                        clip_denoised=model.clip_denoised,
                    )

                reconstructed_mel_batch = model.decode_first_stage(z_denoised)
                
                if reconstructed_mel_batch.shape[1] == 1:
                    reconstructed_mel_batch = reconstructed_mel_batch.squeeze(1)

        success_count = 0
        for idx, (wav_file, reconstructed_mel) in enumerate(zip(valid_files, reconstructed_mel_batch)):
            try:
                reconstructed_mel_single = reconstructed_mel.unsqueeze(0)
                reconstructed_wav = mel_to_waveform(
                    reconstructed_mel_single, vocoder, device, sampling_rate
                )

                filename = Path(wav_file).name
                output_path = output_dir / filename.replace(".wav", f"_{generator_name}.wav")
                reconstructed_wav_float = reconstructed_wav.astype(np.float32) / 32768.0
                sf.write(output_path, reconstructed_wav_float, sampling_rate)
                success_count += 1
            except Exception as e:
                print(f"  ⚠ Warning: Failed to save {Path(wav_file).name}: {e}")

        if batch_id == 1 or len(wav_files_batch) <= 4:
            print(f"[Batch {batch_id}] ✓ Completed {success_count}/{len(valid_files)} files")
        
        return success_count

    except Exception as e:
        print(f"[Batch {batch_id}] ✗ Error processing batch: {e}")
        import traceback
        traceback.print_exc()
        return 0


def split_into_batches(items, batch_size):
    """Yield consecutive batches from a list."""
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def main():
    args = parse_args()

    watermarked_dir = Path(args.watermarked_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    noise_strength = float(args.noise_level)
    
    if not 0.0 <= noise_strength <= 1.0:
        raise ValueError(f"noise_level must be between 0 and 1, got {noise_strength}")

    if args.generator_name:
        generator = args.generator_name
    else:
        noise_str = str(noise_strength).replace(".", "_")
        generator = f"differase_latent_noise{noise_str}"

    if args.device is not None:
        device_str = f"cuda:{args.device}"
    else:
        device_str = "cuda:0"
    
    if not torch.cuda.is_available():
        print("⚠ WARNING: CUDA is not available, using CPU (this will be slow)")
        device_str = "cpu"
    
    device = torch.device(device_str)

    print("=" * 80)
    print("DiffErase-latent Watermark Removal")
    print("=" * 80)
    print(f"Device: {device}")
    if torch.cuda.is_available() and device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
    print(f"Watermarked dir: {watermarked_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Noise strength: {noise_strength}")
    print(f"Batch size: {args.batch_size}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Generator name (suffix): {generator}")
    print("=" * 80)

    if not watermarked_dir.is_dir():
        raise FileNotFoundError(
            f"`watermarked_dir` does not exist or is not a directory: {watermarked_dir}"
        )

    wav_files = sorted(watermarked_dir.glob("*.wav"))
    print(f"\nFound {len(wav_files)} wav files in {watermarked_dir}")
    if not wav_files:
        print("No .wav files found. Exiting.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    repo_root = _script_dir
    checkpoint_path = repo_root / args.checkpoint
    config_path = repo_root / args.config

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    print("\n" + "=" * 80)
    print("Loading DiffErase-latent config & model...")
    print("=" * 80)
    print(f"Loading config from: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    duration = config["preprocessing"]["audio"]["duration"]
    filter_length = config["preprocessing"]["stft"]["filter_length"]
    hop_length = config["preprocessing"]["stft"]["hop_length"]
    win_length = config["preprocessing"]["stft"]["win_length"]
    n_mel_channels = config["preprocessing"]["mel"]["n_mel_channels"]
    mel_fmin = config["preprocessing"]["mel"]["mel_fmin"]
    mel_fmax = config["preprocessing"]["mel"]["mel_fmax"]

    print(f"  Sampling rate: {sampling_rate} Hz")
    print(f"  Duration: {duration} seconds")
    print(f"  Mel channels: {n_mel_channels}")

    print("\nInitializing STFT...")
    stft = TacotronSTFT(
        filter_length=filter_length,
        hop_length=hop_length,
        win_length=win_length,
        n_mel_channels=n_mel_channels,
        sampling_rate=sampling_rate,
        mel_fmin=mel_fmin,
        mel_fmax=mel_fmax,
    ).to(device)
    print("✓ STFT initialized")

    print(f"\nLoading AudioLDM model from checkpoint: {checkpoint_path}")

    original_cwd = os.getcwd()
    os.chdir(LATENT_ROOT)
    try:
        model = instantiate_from_config(config["model"])
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        model = model.to(device)
        model.eval()
        print(f"✓ Model loaded (global step: {ckpt.get('global_step', 'unknown')})")
    finally:
        os.chdir(original_cwd)

    print("\nLoading vocoder...")
    print("  Using HiFiGAN vocoder from AudioLDM...")
    os.chdir(LATENT_ROOT)
    try:
        vocoder = get_vocoder(config, device, n_mel_channels)
    finally:
        os.chdir(original_cwd)
    print("✓ Vocoder loaded")

    print("\n" + "=" * 80)
    print(f"Removing watermarks from {len(wav_files)} audio files")
    print(f"Batch size: {args.batch_size}")
    print("=" * 80)

    start_time = time.time()
    success_count = 0

    batches = list(split_into_batches(wav_files, args.batch_size))
    total_batches = len(batches)

    print(f"\nProcessing {total_batches} batches...")
    
    for batch_id, batch in enumerate(tqdm(batches, desc="Overall Progress", unit="batch"), 1):
        if batch_id == 1:
            print(f"\n{'=' * 80}")
            print(f"Batch {batch_id}/{total_batches} ({len(batch)} files)")
            print(f"{'=' * 80}")
            show_progress = True
        else:
            show_progress = False

        batch_success = process_audio_batch(
            wav_files_batch=batch,
            model=model,
            stft=stft,
            vocoder=vocoder,
            noise_strength=noise_strength,
            device=device,
            output_dir=output_dir,
            generator_name=generator,
            sampling_rate=sampling_rate,
            duration=duration,
            batch_id=batch_id,
            show_progress=show_progress,
        )
        success_count += batch_success

    elapsed_time = time.time() - start_time

    print("\n" + "=" * 80)
    print("Watermark Removal Complete!")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    output_count = len(list(output_dir.glob("*.wav")))
    print(f"Generated: {output_count}/{len(wav_files)} audio files")
    if len(wav_files) > 0:
        success_rate = 100 * success_count / len(wav_files)
        print(f"Success rate: {success_count}/{len(wav_files)} ({success_rate:.1f}%)")
        print(f"Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        print(f"Average time per file: {elapsed_time/len(wav_files):.2f} seconds")
    print("=" * 80)


if __name__ == "__main__":
    main()
