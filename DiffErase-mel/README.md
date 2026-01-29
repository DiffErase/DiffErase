# DiffErase-mel: Training

Train the mel-spectrogram diffusion model used by DiffErase for watermark removal.  

## Setup

```bash
pip install .
```

## Data

- Use WAV (or convert from MP3/FLAC).
- Datasets: e.g. [LibriSpeech](https://www.openslr.org/12/), [FMA](https://github.com/mdeff/fma), [Clotho](https://zenodo.org/records/3490684).

**1. Split train/test**  
Produces `train_files.json` / `test_files.json` 

```bash
python split_train_test.py /path/to/audio_dir [--output_dir /path] [--test_count 100] [--seed 42]
```

**2. Build mel-spectrogram dataset**  
BigVGAN-compatible mel images; output is a HuggingFace Dataset on disk.

```bash
python scripts/audio_to_images_bigvgan.py \
  --train_files_json /path/to/train_files.json \
  --output_dir data/audio-diffusion-bigvgan-256 \
  --resolution 512,80 \
  --hop_length 256 \
  --n_fft 1024 \
  --win_size 1024 \
  --sample_rate 22050
```

## Training

Single-GPU:

```bash
accelerate launch --config_file config/accelerate_local.yaml \
  scripts/train_unet.py \
  --dataset_name data/audio-diffusion-bigvgan-256 \
  --output_dir models/ddpm-ema-audio-256 \
  --train_batch_size 8 \
  --eval_batch_size 4 \
  --num_epochs 100 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-4 \
  --lr_warmup_steps 500 \
  --hop_length 256 \
  --sample_rate 22050 \
  --n_fft 1024 \
  --save_model_epochs 10 \
  --save_images_epochs 10
```

Multi-GPU: use `config/accelerate_multi_gpu.yaml` and tune `train_batch_size` / `gradient_accumulation_steps`.
