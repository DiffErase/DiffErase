"""
Build BigVGAN-compatible mel-spectrogram image dataset for UNet training.
Output: HuggingFace Dataset (train split) saved to disk with image + log_mel range per slice.
"""

import argparse
import io
import json
import logging
import os
import re
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
bigvgan_path = os.path.join(parent_dir, "BigVGAN")
if bigvgan_path not in sys.path:
    sys.path.insert(0, bigvgan_path)

import numpy as np
import pandas as pd
import torch
import librosa
from PIL import Image as PILImage
from datasets import Dataset, DatasetDict, Features, Image, Value
from tqdm.auto import tqdm
from meldataset import get_mel_spectrogram

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Log-mel floor used by BigVGAN (log(1e-5) ≈ -11.513)
MEL_MIN = -11.513


class BigVGANMel:
    """BigVGAN-compatible mel: load audio, slice, convert to log-mel image."""

    def __init__(
        self,
        x_res: int = 256,
        y_res: int = 256,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_size: int = 1024,
        fmin: int = 0,
        fmax: int = None,
        top_db: int = 80,
    ):
        self.x_res, self.y_res = x_res, y_res
        self.n_mels = y_res
        self.sr = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.top_db = top_db
        self.slice_size = x_res * hop_length - 1
        self.audio = None
        self._h = type("H", (), {
            "n_fft": n_fft, "num_mels": y_res, "sampling_rate": sample_rate,
            "hop_size": hop_length, "win_size": win_size, "fmin": fmin, "fmax": fmax,
        })()

    def load_audio(self, audio_file: str):
        self.audio, _ = librosa.load(audio_file, mono=True, sr=self.sr)
        max_amp = np.abs(self.audio).max()
        if max_amp > 1.0:
            self.audio /= max_amp
        min_len = self.x_res * self.hop_length
        if len(self.audio) < min_len:
            self.audio = np.concatenate([self.audio, np.zeros(min_len - len(self.audio))])

    def get_number_of_slices(self) -> int:
        return len(self.audio) // self.slice_size

    def get_audio_slice(self, idx: int) -> np.ndarray:
        s = self.slice_size
        return self.audio[s * idx : s * (idx + 1)]

    def audio_slice_to_image(self, idx: int, return_mel_range: bool = False):
        """Convert one slice to uint8 PIL image; option to return (log_mel_min, log_mel_max) for vocoder."""
        audio_slice = self.get_audio_slice(idx)
        log_mel = get_mel_spectrogram(torch.FloatTensor(audio_slice).unsqueeze(0), self._h)
        log_mel = log_mel.squeeze(0).numpy()

        if log_mel.shape[1] != self.x_res:
            log_mel = log_mel[:, :self.x_res] if log_mel.shape[1] > self.x_res else np.pad(
                log_mel, ((0, 0), (0, self.x_res - log_mel.shape[1])), constant_values=log_mel.min()
            )

        mel_max = log_mel.max()
        img = ((log_mel - MEL_MIN) / (mel_max - MEL_MIN + 1e-8) * 255).clip(0, 255).astype(np.uint8)
        image = PILImage.fromarray(img)
        return (image, MEL_MIN, float(mel_max)) if return_mel_range else image


def _collect_audio_files(args):
    """Return list of audio file paths from JSON or directory scan."""
    if args.train_files_json:
        with open(args.train_files_json, "r", encoding="utf-8") as f:
            return [x["path"] for x in json.load(f)["files"]]
    return [
        os.path.join(r, f)
        for r, _, files in os.walk(args.input_dir)
        for f in files
        if re.search(r"\.(mp3|wav|m4a|flac)$", f, re.IGNORECASE)
    ]


def main(args):
    mel = BigVGANMel(
        x_res=args.resolution[0], y_res=args.resolution[1],
        hop_length=args.hop_length, sample_rate=args.sample_rate,
        n_fft=args.n_fft, win_size=args.win_size, fmin=args.fmin, fmax=args.fmax, top_db=args.top_db,
    )
    os.makedirs(args.output_dir, exist_ok=True)

    audio_files = _collect_audio_files(args)
    print(f"Audio files: {len(audio_files)}")

    examples = []
    for audio_file in tqdm(audio_files, desc="Processing"):
        if not os.path.exists(audio_file) or os.path.getsize(audio_file) == 0:
            logger.warning(f"Skip: {audio_file}")
            continue
        try:
            mel.load_audio(audio_file)
        except Exception as e:
            logger.warning(f"Load failed {audio_file}: {e}")
            continue

        for i in range(mel.get_number_of_slices()):
            try:
                image, log_mel_min, log_mel_max = mel.audio_slice_to_image(i, return_mel_range=True)
                if all(np.frombuffer(image.tobytes(), dtype=np.uint8) == 255):
                    continue
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                examples.append({
                    "image": {"bytes": buf.getvalue()},
                    "audio_file": audio_file,
                    "slice": i,
                    "log_mel_min": log_mel_min,
                    "log_mel_max": log_mel_max,
                })
            except Exception as e:
                logger.warning(f"Slice failed {audio_file} #{i}: {e}")

    if not examples:
        logger.warning("No valid slices.")
        return

    print(f"Slices: {len(examples)}")
    ds = Dataset.from_pandas(
        pd.DataFrame(examples),
        features=Features({
            "image": Image(),
            "audio_file": Value(dtype="string"),
            "slice": Value(dtype="int16"),
            "log_mel_min": Value(dtype="float32"),
            "log_mel_max": Value(dtype="float32"),
        }),
    )
    DatasetDict({"train": ds}).save_to_disk(args.output_dir)
    print(f"Saved: {args.output_dir}")

    if args.push_to_hub:
        DatasetDict({"train": ds}).push_to_hub(args.push_to_hub)

    config_path = os.path.join(args.output_dir, "mel_config.txt")
    with open(config_path, "w") as f:
        f.write(f"resolution={args.resolution[0]}x{args.resolution[1]}\n")
        f.write(f"hop_length={mel.hop_length} n_fft={mel.n_fft} sample_rate={mel.sr}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build BigVGAN mel spectrogram dataset")
    p.add_argument("--input_dir", type=str, default=None, help="Input audio directory")
    p.add_argument("--train_files_json", type=str, default=None, help="JSON with train file paths")
    p.add_argument("--output_dir", type=str, default="data/audio-diffusion-bigvgan-256")
    p.add_argument("--resolution", type=str, default="512,80", help="width,height")
    p.add_argument("--hop_length", type=int, default=256)
    p.add_argument("--n_fft", type=int, default=1024)
    p.add_argument("--win_size", type=int, default=1024)
    p.add_argument("--sample_rate", type=int, default=22050)
    p.add_argument("--fmin", type=int, default=0)
    p.add_argument("--fmax", type=int, default=None)
    p.add_argument("--top_db", type=int, default=80)
    p.add_argument("--push_to_hub", type=str, default=None)
    args = p.parse_args()

    if not args.train_files_json and not args.input_dir:
        p.error("Provide --input_dir or --train_files_json")

    try:
        args.resolution = tuple(int(x) for x in args.resolution.replace(" ", "").split(","))
    except ValueError:
        p.error("--resolution must be width,height (e.g. 512,80)")
    if len(args.resolution) != 2:
        p.error("--resolution must have 2 values")

    main(args)
