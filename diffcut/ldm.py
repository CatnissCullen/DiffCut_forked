from typing import Optional, Tuple, Literal, List
from functools import partial

import torch
from torch import nn

from diffusers import AutoPipelineForText2Image
from diffusers import DDIMScheduler, DDIMInverseScheduler
from tqdm.notebook import tqdm_notebook as tqdm

def build_ldm_from_cfg(model_key: str,
                       inverse_scheduler: Literal[None, "inv_ddim"] = None,
                       device: int = 0):
    print('Loading SD model')
    device = torch.device(f'cuda:{device}') if torch.cuda.is_available() else torch.device('cpu')

    pipe = AutoPipelineForText2Image.from_pretrained(model_key, torch_dtype=torch.float16).to(device)

    pipe.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config,
            timestep_spacing="trailing",
            )

    if inverse_scheduler == "inv_ddim":
        pipe.inverse_scheduler = DDIMInverseScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing",
        )
        
    print('SD model loaded')
    return pipe, device

class LdmExtractor(nn.Module):

    LDM_CONFIGS = {
        "SSD-1B": ("segmind/SSD-1B", "XL")
    }

    def __init__(
        self,
        device: int = 0,
        model_name: str = "SSD-1B",
        inverse_scheduler: Optional[Literal[None, "inv_ddim"]] = None
    ):

        super().__init__()

        model_key, sd_version = self.LDM_CONFIGS[model_name]

        self.text_encoders = []
        self.pipe, self.device = build_ldm_from_cfg(model_key, inverse_scheduler, device)
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        self.text_encoders.append(self.pipe.text_encoder)

        if sd_version == "XL":
            self.text_encoders.append(self.pipe.text_encoder_2)

        self.scheduler = self.pipe.scheduler
        self.scheduler.set_timesteps(50)

        if inverse_scheduler is not None:
            self.inverse_scheduler = self.pipe.inverse_scheduler

    def get_inverse_timesteps(self, timesteps, strength):
        # get the original timestep using init_timestep
        num_inference_steps = len(timesteps)
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)

        # safety for t_start overflow to prevent empty timesteps slice
        timesteps = timesteps if t_start == 0 else timesteps[:-t_start]

        return timesteps, t_start

    @torch.no_grad()
    def denoise_step(self, x, t, prompt_embeds, added_cond_kwargs, guidance_scale, scheduler):
        latents = torch.cat([x] * 2) if self.do_classifier_free_guidance(guidance_scale) else x
        latents = scheduler.scale_model_input(latents, t)

        noise_pred = self.unet(latents, t, encoder_hidden_states=prompt_embeds, \
                               added_cond_kwargs=added_cond_kwargs).sample

        if self.do_classifier_free_guidance(guidance_scale):
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        denoised_latent = scheduler.step(noise_pred, t, x)['prev_sample']

        return denoised_latent

    @torch.no_grad()
    def sample_loop(self,
                    x: torch.Tensor,
                    prompt_embeds: List,
                    num_inference_steps: int = 50,
                    added_cond_kwargs = None,
                    guidance_scale: float = 1.,
                    strength: float = 1.):

        scheduler = self.scheduler

        scheduler.set_timesteps(num_inference_steps)
        timesteps = scheduler.timesteps

        _,  t_start = self.get_inverse_timesteps(timesteps, strength)
        timesteps = timesteps[t_start:]

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for t in tqdm(timesteps):
                x = self.denoise_step(x, t, prompt_embeds, added_cond_kwargs, guidance_scale, scheduler)
        return x

    @torch.no_grad()
    def invert(self,
               latent: torch.Tensor,
               source_prompt: str = "",
               num_inference_steps: int = 50,
               num_images_per_prompt: int = 1,
               guidance_scale: float = 1.,
               strength: float = 1.,
               device: str = "cuda"
               ):

        scheduler = self.inverse_scheduler

        scheduler.set_timesteps(num_inference_steps)
        timesteps = scheduler.timesteps
        timesteps, _ = self.get_inverse_timesteps(timesteps, strength)

        prompt_embeds, added_cond_kwargs = self.get_text_embeds(
            source_prompt, num_images_per_prompt, guidance_scale
            )

        batch_size = latent.shape[0]
        variance_noise_shape = (
            int(num_inference_steps*strength),
            batch_size,
            latent.shape[1],
            latent.shape[2],
            latent.shape[3])

        xts = torch.zeros(size=variance_noise_shape, device=device, dtype=prompt_embeds[0].dtype)
        xts = torch.cat([latent.unsqueeze(0), xts], dim=0)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for i, t in enumerate(timesteps):
                latent = self.denoise_step(latent, t, prompt_embeds, added_cond_kwargs, guidance_scale, scheduler)
                xts[i+1] = latent

        return xts

    def register_hooks(self):
        self_att_modules = {n:mod.attn1 for n, mod in self.unet.named_modules() if hasattr(mod, 'attn1')}
        for n, block in self_att_modules.items():
            if n == 'down_blocks.2.attentions.1.transformer_blocks.3':
                def hook_self_attn(mod, input, output, n):
                    self._down_features[n] = output.detach()

                block.register_forward_hook(partial(hook_self_attn, n=n))

    def do_classifier_free_guidance(self, guidance_scale):
        return guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    @torch.no_grad()
    def get_text_embeds(self, prompt, num_images_per_prompt=1, guidance_scale=1.0, img_size=1024):
        do_classifier_free_guidance = self.do_classifier_free_guidance(guidance_scale)
        batch_size = len(prompt)

        prompt_embeds_tuple = self.pipe.encode_prompt(
            prompt,
            device=self.device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance)

        if len(prompt_embeds_tuple) == 2:
            prompt_embeds, negative_prompt_embeds = prompt_embeds_tuple
            if do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            return prompt_embeds, None

        else:
            (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            ) = prompt_embeds_tuple

            add_text_embeds = pooled_prompt_embeds
            add_time_ids = self.pipe._get_add_time_ids(
                (img_size, img_size), (0, 0), (img_size, img_size), dtype=prompt_embeds.dtype, \
                    text_encoder_projection_dim=self.text_encoders[1].config.projection_dim)
            negative_add_time_ids = add_time_ids

            if self.do_classifier_free_guidance(guidance_scale):
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
                add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

            prompt_embeds = prompt_embeds.to(self.device)
            add_text_embeds = add_text_embeds.to(self.device)
            add_time_ids = add_time_ids.to(self.device).repeat(batch_size * num_images_per_prompt, 1)

            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            return prompt_embeds, added_cond_kwargs

    @torch.no_grad()
    def encode_to_latent(self, input_image):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            input_image = 2 * input_image - 1
            posterior = self.vae.encode(input_image).latent_dist
            latent_image = posterior.mean * self.vae.config.scaling_factor
        return latent_image

    @torch.no_grad()
    def decode_from_latent(self, latent):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            latent = 1 / self.vae.config.scaling_factor * latent
            img = self.vae.decode(latent).sample
            img = (img / 2 + 0.5).clamp(0, 1)
        return img

    def forward(self,
                     batched_inputs,
                     num_images_per_prompt: int = 1,
                     guidance_scale: float = 1.,
                     n_steps: int = 50,
                     steps: Tuple[int, ...] = (0,),
                     encode_method: Literal["noise", "inversion"] = "inversion"
                     ):

        captions = batched_inputs["caption"]
        images = batched_inputs["img"]
        batch_size = images.shape[0]
        img_size = images.shape[-1]

        rng = torch.Generator(device=self.device).manual_seed(42)

        prompt_embeds, added_cond_kwargs = self.get_text_embeds(captions, num_images_per_prompt, guidance_scale, img_size)

        latent_image = self.encode_to_latent(images)

        if encode_method == "noise":
            noise = torch.randn(1, 4, img_size//8, img_size//8, generator=rng, device=self.device)
            noise = noise.expand_as(latent_image)

        elif encode_method == "inversion":
            strength = (max(steps) + 1) / 1000.
            xts = self.invert(latent_image, captions, n_steps, num_images_per_prompt, guidance_scale, strength)

        self.register_hooks()

        for i, step in enumerate(steps):  # BEST IS step=50 (steps=(50, ))
            self._down_features = {}

            t = torch.tensor([step], device=self.device).expand(batch_size)

            # Either we noise or we invert the latent.
            if encode_method == "noise":  # use NOISED LATENT (encoded before unet) as the Xt through Unet Encoder
                noisy_latent_image = self.pipe.scheduler.add_noise(latent_image, noise, t)

            elif encode_method == "inversion":  # use INVERTED Xt as the Xt through Unet Encoder
                noisy_latent_image = xts[-(i+1)] #idx + 1 as the first idx is the orig image.

            if self.do_classifier_free_guidance(guidance_scale):
                noisy_latent_image = torch.cat([noisy_latent_image] * 2)

            t = t.repeat(noisy_latent_image.shape[0] // t.shape[0])
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                with torch.no_grad():
                    noisy_latent_image = self.pipe.scheduler.scale_model_input(noisy_latent_image, t)
                    self.unet(noisy_latent_image, t, encoder_hidden_states=prompt_embeds, \
                              added_cond_kwargs=added_cond_kwargs).sample

        return self._down_features  # ONLY RETURN THE LAST STEP'S _DOWN_FEATURES (BEST: THE ONLY STEP 50)
