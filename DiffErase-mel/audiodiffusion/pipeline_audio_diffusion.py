# This code has been migrated to diffusers but can be run locally with
# pipe = DiffusionPipeline.from_pretrained("teticio/audio-diffusion-256", custom_pipeline="audio-diffusion/audiodiffusion/pipeline_audio_diffusion.py")

# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from math import acos, sin
from typing import List, Tuple, Union

import numpy as np
import torch
import librosa
from diffusers import (
    AudioPipelineOutput,
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    DiffusionPipeline,
    ImagePipelineOutput,
    UNet2DConditionModel,
)
from diffusers.utils import BaseOutput
from PIL import Image

from .mel import Mel


class AudioDiffusionPipeline(DiffusionPipeline):
    """
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        vqae ([`AutoencoderKL`]): Variational AutoEncoder for Latent Audio Diffusion or None
        unet ([`UNet2DConditionModel`]): UNET model
        mel ([`Mel`]): transform audio <-> spectrogram
        scheduler ([`DDIMScheduler` or `DDPMScheduler`]): de-noising scheduler
    """

    _optional_components = ["vqvae"]

    def __init__(
        self,
        vqvae: AutoencoderKL,
        unet: UNet2DConditionModel,
        mel: Mel,
        scheduler: Union[DDIMScheduler, DDPMScheduler],
    ):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler, mel=mel, vqvae=vqvae)

    def get_default_steps(self) -> int:
        """Returns default number of steps recommended for inference

        Returns:
            `int`: number of steps
        """
        return 50 if isinstance(self.scheduler, DDIMScheduler) else 1000

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        audio_file: str = None,
        raw_audio: np.ndarray = None,
        slice: int = 0,
        start_step: int = 0,
        steps: int = None,
        generator: torch.Generator = None,
        mask_start_secs: float = 0,
        mask_end_secs: float = 0,
        step_generator: torch.Generator = None,
        eta: float = 0,
        noise: torch.Tensor = None,
        encoding: torch.Tensor = None,
        return_dict=True,
        return_mel_tensor: bool = False,
    ) -> Union[
        Union[AudioPipelineOutput, ImagePipelineOutput],
        Tuple[List[Image.Image], Tuple[int, List[np.ndarray]]],
        Tuple[List[Image.Image], Tuple[int, List[np.ndarray]], torch.Tensor],
    ]:
        """Generate random mel spectrogram from audio input and convert to audio.

        Args:
            batch_size (`int`): number of samples to generate
            audio_file (`str`): must be a file on disk due to Librosa limitation or
            raw_audio (`np.ndarray`): audio as numpy array
            slice (`int`): slice number of audio to convert
            start_step (int): step to start from
            steps (`int`): number of de-noising steps (defaults to 50 for DDIM, 1000 for DDPM)
            generator (`torch.Generator`): random number generator or None
            mask_start_secs (`float`): number of seconds of audio to mask (not generate) at start
            mask_end_secs (`float`): number of seconds of audio to mask (not generate) at end
            step_generator (`torch.Generator`): random number generator used to de-noise or None
            eta (`float`): parameter between 0 and 1 used with DDIM scheduler
            noise (`torch.Tensor`): noise tensor of shape (batch_size, 1, height, width) or None
            encoding (`torch.Tensor`): for UNet2DConditionModel shape (batch_size, seq_length, cross_attention_dim)
            return_dict (`bool`): if True return AudioPipelineOutput, ImagePipelineOutput else Tuple
            return_mel_tensor (`bool`): if True, also return mel spectrogram as tensor in log scale (for BigVGAN)

        Returns:
            `List[PIL Image]`: mel spectrograms (`float`, `List[np.ndarray]`): sample rate and raw audios
            If return_mel_tensor=True, also returns `torch.Tensor`: mel spectrogram in log scale, shape (batch, n_mels, time)
        """

        steps = steps or self.get_default_steps()
        self.scheduler.set_timesteps(steps)
        step_generator = step_generator or generator
        # For backwards compatibility
        if type(self.unet.sample_size) == int:
            self.unet.sample_size = (self.unet.sample_size, self.unet.sample_size)
        if noise is None:
            noise = torch.randn(
                (
                    batch_size,
                    self.unet.in_channels,
                    self.unet.sample_size[0],
                    self.unet.sample_size[1],
                ),
                generator=generator,
                device=self.device,
            )
        images = noise
        mask = None

        num_timesteps = len(self.scheduler.timesteps)
        if start_step >= num_timesteps:
            # clamp start_step; use original input without noise
            start_step = num_timesteps
            use_original_input = True
        else:
            use_original_input = False

        if audio_file is not None or raw_audio is not None:
            self.mel.load_audio(audio_file, raw_audio)
            input_image = self.mel.audio_slice_to_image(slice)
            input_image = np.frombuffer(input_image.tobytes(), dtype="uint8").reshape(
                (input_image.height, input_image.width)
            )
            input_image = (input_image / 255) * 2 - 1
            input_images = torch.tensor(input_image[np.newaxis, :, :], dtype=torch.float).to(self.device)

            if self.vqvae is not None:
                input_images = self.vqvae.encode(torch.unsqueeze(input_images, 0)).latent_dist.sample(
                    generator=generator
                )[0]
                input_images = 0.18215 * input_images

            if use_original_input:
                # use original input as-is (no noise). ensure (B,C,H,W)
                if len(input_images.shape) == 2:
                    input_images = input_images.unsqueeze(0).unsqueeze(0)  # (1, 1, height, width)
                elif len(input_images.shape) == 3:
                    input_images = input_images.unsqueeze(0)
                images = input_images
            elif start_step > 0:
                images[0, 0] = self.scheduler.add_noise(input_images, noise, self.scheduler.timesteps[start_step - 1])

            pixels_per_second = (
                self.unet.sample_size[1] * self.mel.get_sample_rate() / self.mel.x_res / self.mel.hop_length
            )
            mask_start = int(mask_start_secs * pixels_per_second)
            mask_end = int(mask_end_secs * pixels_per_second)
            if start_step < num_timesteps:
                mask = self.scheduler.add_noise(input_images, noise, torch.tensor(self.scheduler.timesteps[start_step:]))
            else:
                mask = None

        denoising_timesteps = self.scheduler.timesteps[start_step:] if start_step < num_timesteps else []
        for step, t in enumerate(self.progress_bar(denoising_timesteps)):
            if isinstance(self.unet, UNet2DConditionModel):
                model_output = self.unet(images, t, encoding)["sample"]
            else:
                model_output = self.unet(images, t)["sample"]

            if isinstance(self.scheduler, DDIMScheduler):
                images = self.scheduler.step(
                    model_output=model_output,
                    timestep=t,
                    sample=images,
                    eta=eta,
                    generator=step_generator,
                )["prev_sample"]
            else:
                images = self.scheduler.step(
                    model_output=model_output,
                    timestep=t,
                    sample=images,
                    generator=step_generator,
                )["prev_sample"]

            if mask is not None:
                if mask_start > 0:
                    images[:, :, :, :mask_start] = mask[:, step, :, :mask_start]
                if mask_end > 0:
                    images[:, :, :, -mask_end:] = mask[:, step, :, -mask_end:]

        if self.vqvae is not None:
            # 0.18215 was scaling factor used in training to ensure unit variance
            images = 1 / 0.18215 * images
            images = self.vqvae.decode(images)["sample"]

        images = (images / 2 + 0.5).clamp(0, 1)
        
        # Extract mel tensor BEFORE quantization to uint8 (if requested)
        # This preserves the full dynamic range for BigVGAN
        mel_tensor = None
        if return_mel_tensor:
            mel_tensors = []
            # images shape: (batch, channels, height, width) - still in float [0, 1]
            for i in range(images.shape[0]):
                # Get single spectrogram: (height, width) in float [0, 1]
                if images.shape[1] == 1:
                    img_float = images[i, 0].cpu().numpy()  # (height, width)
                else:
                    # Convert RGB to grayscale if needed
                    img_float = torch.mean(images[i], dim=0).cpu().numpy()
                
                # Convert [0, 1] back to dB scale [-top_db, 0]
                # Original encoding: bytedata = ((log_S + top_db) * 255 / top_db).clip(0, 255)
                # So: img_float = bytedata / 255 = (log_S + top_db) / top_db
                # Therefore: log_S = img_float * top_db - top_db
                log_S_db = img_float * self.mel.top_db - self.mel.top_db
                
                # Convert dB to power scale
                # Since training used ref=np.max, we need to convert to absolute power scale
                # db_to_power with ref=1.0 gives: S = 10^(log_S_db/10)
                S_power = librosa.db_to_power(log_S_db, ref=1.0)
                
                # Convert power to log scale for BigVGAN
                # BigVGAN expects: log(mel_spectrogram) where mel_spectrogram is in power scale
                clip_val = 1e-5
                log_S = np.log(np.clip(S_power, a_min=clip_val, a_max=None))
                
                mel_tensors.append(torch.FloatTensor(log_S))
            
            # Stack into batch tensor: (batch, n_mels, time)
            mel_tensor = torch.stack(mel_tensors)
        
        # Convert to numpy for processing
        images_np = images.cpu().permute(0, 2, 3, 1).numpy()
        images_np = (images_np * 255).round().astype("uint8")
        
        images = list(
            map(lambda _: Image.fromarray(_[:, :, 0]), images_np)
            if images_np.shape[3] == 1
            else map(lambda _: Image.fromarray(_, mode="RGB").convert("L"), images_np)
        )

        audios = list(map(lambda _: self.mel.image_to_audio(_), images))
        if not return_dict:
            if return_mel_tensor:
                return images, (self.mel.get_sample_rate(), audios), mel_tensor
            return images, (self.mel.get_sample_rate(), audios)

        return BaseOutput(**AudioPipelineOutput(np.array(audios)[:, np.newaxis, :]), **ImagePipelineOutput(images))

    @torch.no_grad()
    def encode(self, images: List[Image.Image], steps: int = 50) -> np.ndarray:
        """Reverse step process: recover noisy image from generated image.

        Args:
            images (`List[PIL Image]`): list of images to encode
            steps (`int`): number of encoding steps to perform (defaults to 50)

        Returns:
            `np.ndarray`: noise tensor of shape (batch_size, 1, height, width)
        """

        # Only works with DDIM as this method is deterministic
        assert isinstance(self.scheduler, DDIMScheduler)
        self.scheduler.set_timesteps(steps)
        sample = np.array(
            [np.frombuffer(image.tobytes(), dtype="uint8").reshape((1, image.height, image.width)) for image in images]
        )
        sample = (sample / 255) * 2 - 1
        sample = torch.Tensor(sample).to(self.device)

        for t in self.progress_bar(torch.flip(self.scheduler.timesteps, (0,))):
            prev_timestep = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                self.scheduler.alphas_cumprod[prev_timestep]
                if prev_timestep >= 0
                else self.scheduler.final_alpha_cumprod
            )
            beta_prod_t = 1 - alpha_prod_t
            model_output = self.unet(sample, t)["sample"]
            pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * model_output
            sample = (sample - pred_sample_direction) * alpha_prod_t_prev ** (-0.5)
            sample = sample * alpha_prod_t ** (0.5) + beta_prod_t ** (0.5) * model_output

        return sample

    @staticmethod
    def slerp(x0: torch.Tensor, x1: torch.Tensor, alpha: float) -> torch.Tensor:
        """Spherical Linear intERPolation

        Args:
            x0 (`torch.Tensor`): first tensor to interpolate between
            x1 (`torch.Tensor`): seconds tensor to interpolate between
            alpha (`float`): interpolation between 0 and 1

        Returns:
            `torch.Tensor`: interpolated tensor
        """

        theta = acos(torch.dot(torch.flatten(x0), torch.flatten(x1)) / torch.norm(x0) / torch.norm(x1))
        return sin((1 - alpha) * theta) * x0 / sin(theta) + sin(alpha * theta) * x1 / sin(theta)
