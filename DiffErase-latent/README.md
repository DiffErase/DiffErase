# DiffErase-latent: Training

Train the latent diffusion model (AudioLDM-based) used by DiffErase for watermark removal.  
This model operates in the latent space of mel-spectrograms, providing efficient watermark removal.

## Setup

```bash
pip install -r ../requirements.txt wandb
```

## Data

- Use WAV files (or convert from MP3/FLAC).
- Datasets: e.g. [LibriSpeech](https://www.openslr.org/12/), [FMA](https://github.com/mdeff/fma), [Clotho](https://zenodo.org/records/3490684).

**1. Prepare metadata**  
Creates train/val/test JSON metadata files and registers dataset in `data/dataset/metadata/dataset_root.json`.

```bash
python create_metadata.py /path/to/audio_dir --name librispeech [--n_val 100] [--n_test 100]
```

The script scans for all `.wav` files, splits them into train/val/test (default: val=100, test=100, rest for train) and registers the dataset under `--name`.

**2. Download pretrained checkpoints**  
The frozen VAE (`vae_mel_16k_64bins.ckpt`) and HiFi-GAN vocoder (`hifigan_16k_64bins.ckpt/.json`) are included in the [Hugging Face](https://huggingface.co/DiffErase/DiffErase) download (see the [main README](../README.md#checkpoints)) and placed in `data/checkpoints/`.

## Training

```bash
python -m audioldm_train.train.latent_diffusion \
  -c audioldm_train/config/differase/differase_unconditional.yaml
```

Set the `data` entries in the config to the `--name` used in step 1.