"""Remove DiffErase watermark using DiffErase-mel (mel-spectrogram pipeline).
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove watermark using DiffErase-mel",
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
        help="Noise strength in [0, 1]",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="CUDA device index (e.g., 0,1,2,3).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="DiffErase-mel/models/*.ckpt",
        help="DiffErase-mel model directory",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--generator_name",
        type=str,
        default=None,
        help="Output filename suffix",
    )

    return parser.parse_args()


def _setup_differase_mel_imports():
    """Add DiffErase-mel and its dependencies to `sys.path`."""
    this_dir = Path(__file__).resolve().parent  # DiffErase/
    mel_root = this_dir / "DiffErase-mel"
    bigvgan_dir = mel_root / "BigVGAN"
    scripts_dir = mel_root / "scripts"

    for p in (mel_root, bigvgan_dir, scripts_dir):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    return mel_root, bigvgan_dir, scripts_dir


def image_to_bigvgan_mel(image, log_mel_min=None, log_mel_max=None):
    """Convert a grayscale image back to a log-mel tensor for BigVGAN."""
    bytedata = np.frombuffer(image.tobytes(), dtype="uint8").reshape((image.height, image.width))

    if log_mel_min is None or log_mel_max is None:
        log_mel_min = -11.513
        high_percentile = np.percentile(bytedata, 99.9)
        if high_percentile > 250:
            log_mel_max = 2.0
        elif high_percentile > 240:
            log_mel_max = 1.7
        elif high_percentile > 220:
            log_mel_max = 1.5
        else:
            log_mel_max = 1.2

    log_mel = bytedata.astype(np.float32) / 255.0 * (log_mel_max - log_mel_min) + log_mel_min
    log_mel = np.clip(log_mel, -12.0, 3.0)
    return torch.FloatTensor(log_mel).unsqueeze(0)


def denoise_image_batch(model, scheduler, noisy_tensors, noise_strength=0.1, device="cuda", verbose=False):
    """Denoise a batch of noisy image tensors with a trained UNet."""
    latents = noisy_tensors.to(device)

    num_train_timesteps = scheduler.config.num_train_timesteps
    start_timestep = int(num_train_timesteps * noise_strength)

    num_inference_steps = 1000
    scheduler.set_timesteps(num_inference_steps)
    timesteps = scheduler.timesteps

    start_idx = 0
    for idx, t in enumerate(timesteps):
        if t <= start_timestep:
            start_idx = idx
            break

    timesteps_to_use = timesteps[start_idx:]

    if verbose:
        print(f"  Denoising steps: {len(timesteps_to_use)}")

    model.eval()
    from tqdm import tqdm

    with torch.no_grad():
        iterator = (
            tqdm(enumerate(timesteps_to_use), total=len(timesteps_to_use), desc="Denoising")
            if verbose
            else enumerate(timesteps_to_use)
        )
        for _, t in iterator:
            t_batch = t.unsqueeze(0).expand(latents.shape[0]).to(latents.device)
            noise_pred = model(latents, t_batch).sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample

    # [-1, 1] -> [0, 1] and convert to PIL images.
    image_tensor = (latents + 1.0) / 2.0
    image_tensor = image_tensor.clamp(0, 1)

    denoised_images = []
    for i in range(image_tensor.shape[0]):
        denoised_array = image_tensor[i].squeeze().cpu().numpy()
        denoised_array = (denoised_array * 255).astype(np.uint8)
        denoised_images.append(Image.fromarray(denoised_array))
    return denoised_images


def process_audio_batch(
    wav_files_batch,
    model,
    scheduler,
    vocoder,
    mel_converter,
    noise_strength,
    device,
    output_dir,
    generator_name,
    batch_id=0,
):
    """Process a batch of wav files end-to-end."""
    try:
        image_tensors = []
        log_mel_ranges = []
        valid_files = []

        print(f"\n[Batch {batch_id}] Loading {len(wav_files_batch)} files...")
        for wav_file in wav_files_batch:
            try:
                mel_converter.load_audio(str(wav_file))
                original_image, log_mel_min, log_mel_max = mel_converter.audio_slice_to_image(
                    0,
                    return_mel_range=True,
                )

                image_array = np.array(original_image).astype(np.float32) / 255.0
                image_tensor = torch.FloatTensor(image_array).unsqueeze(0)
                image_tensors.append(image_tensor)
                log_mel_ranges.append((log_mel_min, log_mel_max))
                valid_files.append(wav_file)
            except Exception as e:
                print(f"  Warning: Failed to load {wav_file}: {e}")
                continue

        if not image_tensors:
            print(f"[Batch {batch_id}] No valid files to process")
            return

        print(f"[Batch {batch_id}] Processing {len(image_tensors)} files...")

        batch_tensor = torch.cat(image_tensors, dim=0).unsqueeze(1).to(device)  # [B, 1, H, W]
        batch_tensor = batch_tensor * 2.0 - 1.0  # -> [-1, 1]

        noise = torch.randn_like(batch_tensor)
        timesteps_noise = int(scheduler.config.num_train_timesteps * noise_strength)
        timestep_tensor = torch.tensor([timesteps_noise] * batch_tensor.shape[0]).long().to(device)
        noisy_batch = scheduler.add_noise(batch_tensor, noise, timestep_tensor)

        denoised_images = denoise_image_batch(
            model,
            scheduler,
            noisy_batch,
            noise_strength=noise_strength,
            device=device,
            verbose=(batch_id == 1),
        )

        print(f"[Batch {batch_id}] Generating audio...")
        for wav_file, denoised_image, (log_mel_min, log_mel_max) in zip(valid_files, denoised_images, log_mel_ranges):
            try:
                denoised_mel_tensor = image_to_bigvgan_mel(
                    denoised_image,
                    log_mel_min=log_mel_min,
                    log_mel_max=log_mel_max,
                ).to(device)

                with torch.no_grad():
                    audio_denoised = vocoder(denoised_mel_tensor).cpu().numpy()

                audio_denoised = audio_denoised.ravel()
                filename = Path(wav_file).name
                output_path = output_dir / filename.replace(".wav", f"_{generator_name}.wav")
                sf.write(output_path, audio_denoised, int(mel_converter.sr))
            except Exception as e:
                print(f"  Warning: Failed to save {wav_file}: {e}")

        print(f"[Batch {batch_id}] ✓ Completed {len(valid_files)} files")

    except Exception as e:
        print(f"[Batch {batch_id}] Error processing batch: {e}")
        import traceback

        traceback.print_exc()


def split_into_batches(items, batch_size):
    """Yield consecutive batches from a list."""
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def main():
    args = parse_args()

    watermarked_dir = Path(args.watermarked_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    noise_strength = float(args.noise_level)

    if args.generator_name:
        generator = args.generator_name
    else:
        noise_str = str(noise_strength).replace(".", "_")
        generator = f"differase_mel_noise{noise_str}"

    # Device selection (default: cuda:0 if available).
    if args.device is not None:
        device_str = f"cuda:{args.device}"
    else:
        device_str = "cuda:0"
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available, using CPU")
        device_str = "cpu"
    device = torch.device(device_str)

    print("=" * 80)
    print("DiffErase-mel Watermark Removal")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Watermarked dir: {watermarked_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Noise strength: {noise_strength}")
    print(f"Batch size: {args.batch_size}")
    print(f"Model path: {args.model_path}")
    print(f"Generator name (suffix): {generator}")
    print("=" * 80)

    # Validate input/output.
    if not watermarked_dir.is_dir():
        raise FileNotFoundError(f"`watermarked_dir` does not exist or is not a directory: {watermarked_dir}")

    wav_files = sorted(watermarked_dir.glob("*.wav"))
    print(f"\nFound {len(wav_files)} wav files in {watermarked_dir}")
    if not wav_files:
        print("No .wav files found. Exiting.")
        return

    # Create output directory.
    output_dir.mkdir(parents=True, exist_ok=True)

    _setup_differase_mel_imports()

    # Resolve model directory relative to this script.
    repo_root = Path(__file__).resolve().parent
    model_path = (repo_root / args.model_path).resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"DiffErase-mel model path not found: {model_path}")

    # Load DiffErase-mel UNet and diffusion scheduler.
    print("\n" + "=" * 80)
    print("Loading DiffErase-mel UNet & scheduler...")
    print("=" * 80)

    from diffusers import DDPMScheduler, UNet2DModel

    unet = UNet2DModel.from_pretrained(str(model_path / "unet"))
    unet = unet.to(device)
    unet.eval()

    scheduler = DDPMScheduler.from_pretrained(str(model_path / "scheduler"))
    print("✓ DiffErase-mel UNet & scheduler loaded")

    # Init mel converter and vocoder (BigVGAN).
    print("Initializing mel converter (BigVGAN)...")
    from audio_to_images_bigvgan import BigVGANMel

    mel_converter = BigVGANMel(
        x_res=512,
        y_res=80,
        hop_length=256,
        sample_rate=22050,
        n_fft=1024,
        win_size=1024,
        fmin=0,
        fmax=None,
        top_db=80,
    )
    print("✓ Mel converter initialized")

    print("Loading BigVGAN vocoder...")
    import bigvgan

    # Call `_from_pretrained` with explicit kwargs for better HF Hub compatibility.
    vocoder = bigvgan.BigVGAN._from_pretrained(
        model_id="nvidia/bigvgan_v2_22khz_80band_256x",
        revision="main",
        cache_dir=None,
        force_download=False,
        proxies=None,
        resume_download=False,
        local_files_only=False,  # Needs network on first run unless cached.
        token=None,
        map_location="cpu",
        strict=False,
        use_cuda_kernel=False,
    )
    vocoder.remove_weight_norm()
    vocoder = vocoder.eval().to(device)
    print("✓ BigVGAN vocoder loaded")

    # Batch processing.
    print("\n" + "=" * 80)
    print(f"Processing {len(wav_files)} audio files in batches of {args.batch_size}...")
    print("=" * 80)

    start_time = time.time()

    batches = list(split_into_batches(wav_files, args.batch_size))

    for batch_id, batch in enumerate(batches, 1):
        print(f"\n{'=' * 80}")
        print(f"Batch {batch_id}/{len(batches)} ({len(batch)} files)")
        print(f"{'=' * 80}")

        process_audio_batch(
            wav_files_batch=batch,
            model=unet,
            scheduler=scheduler,
            vocoder=vocoder,
            mel_converter=mel_converter,
            noise_strength=noise_strength,
            device=device,
            output_dir=output_dir,
            generator_name=generator,
            batch_id=batch_id,
        )

    elapsed_time = time.time() - start_time

    print("\n" + "=" * 80)
    print("Processing Complete!")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    output_count = len(list(output_dir.glob("*.wav")))
    print(f"Generated {output_count}/{len(wav_files)} audio files")
    if len(wav_files) > 0:
        print(f"Total time: {elapsed_time:.2f} seconds")
        print(f"Average time per file: {elapsed_time/len(wav_files):.2f} seconds")
    print("=" * 80)


if __name__ == "__main__":
    main()

