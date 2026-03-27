# DiffErase-latent: Training

Train the latent diffusion model (AudioLDM-based) used by DiffErase for watermark removal.  
This model operates in the latent space of mel-spectrograms, providing efficient watermark removal.

## Setup

```bash
pip install poetry
poetry install
```

## Data

- Use WAV files (or convert from MP3/FLAC).
- Datasets: e.g. [LibriSpeech](https://www.openslr.org/12/), [FMA](https://github.com/mdeff/fma), [Clotho](https://zenodo.org/records/3490684).

**1. Prepare metadata**  
Creates train/val/test JSON metadata files and registers dataset in `data/dataset/metadata/dataset_root.json`.

```bash
python create_metadata.py
```

Modify the script to point to your dataset directory. The script will:
- Scan for all `.wav` files
- Split into train/val/test (default: val=100, test=100, rest for train)
- Create metadata JSON files
- Register dataset in `dataset_root.json`

**2. Download pretrained checkpoints**  
Download required checkpoints (VAE, HiFiGAN, etc.):
- VAE checkpoint: `vae_mel_16k_64bins.ckpt`
- HiFiGAN vocoder checkpoints
- Place them in `data/checkpoints/`

## Training

```bash
python audioldm_train/train/latent_diffusion.py \
  -c audioldm_train/config/*.yaml
```