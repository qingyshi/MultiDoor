from .cldm import ControlLDM
from ldm.util import instantiate_from_config
from ldm.modules.encoders.modules import AbstractEncoder
import torch
from torch import nn
from einops import rearrange, repeat
from torchvision.utils import make_grid
from diffusers import StableDiffusionPipeline        
        

class MultiDoor(ControlLDM):
    def __init__(self, ref_key, subject_stage_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_encoder = instantiate_from_config(subject_stage_config)
        self.ref_key = ref_key
    
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, cond = super().get_input(batch, k, bs, *args, **kwargs)
        # cond['c_crossattn'].shape: (b, 77, 1024)
        ref_image = batch[self.ref_key] # ref_image.shape: (b, n, 224, 224, 3)
        ref_token = self.ref_encoder(ref_image)
        cond['ref_token'] = [ref_token]
        return x, cond
    
    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_text = torch.cat(cond['c_crossattn'], dim=1)
        cond_subject = torch.cat(cond['ref_token'], dim=1)

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, subject=cond_subject, caption=cond_text, control=None, only_mid_control=self.only_mid_control)
        else:
            control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_text)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, subject=cond_subject, caption=cond_text, control=control, only_mid_control=self.only_mid_control)
        return eps
    
    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        params += list(self.model.diffusion_model.out.parameters())
        params += list(self.ref_encoder.projector.parameters())
        params.append(self.ref_encoder.null_token)
        for name, param in self.named_parameters():
            if 'transformer_blocks' in name and 'fuser' in name:
                params.append(param)
        
        opt = torch.optim.AdamW(params, lr=lr)
        return opt
    
    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat, c, ref = c["c_concat"][0][:N], c["c_crossattn"][0][:N], c['ref_token'][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z) 

        # ==== visualize the shape mask or the high-frequency map ====
        guide_mask = (c_cat[:, -1, :, :].unsqueeze(1) + 1) * 0.5
        guide_mask = torch.cat([guide_mask, guide_mask, guide_mask], 1)
        HF_map  = c_cat[:, :3, :, :]  # * 2.0 - 1.0

        log["control"] = HF_map

        cond_image = batch[self.ref_key][:N].clone()    # (b, 2, 224, 224, 3)
        cond_image = torch.cat([cond_image[:, i, ...] for i in range(cond_image.shape[1])], dim=2)
        log["conditioning"] = torch.permute(cond_image, (0, 3, 1, 2)) * 2.0 - 1.0  
        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c], "ref_token": [ref]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N, batch)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross], "ref_token": [ref]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c], "ref_token": [ref]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg #* 2.0 - 1.0
        return log
    
    @torch.no_grad()
    def get_unconditional_conditioning(self, N, batch):
        uncond = [" "] * N
        uncond = self.cond_stage_model(uncond)  # (N, 77, 1024)
        return uncond

    
class MultiDoorV2(ControlLDM):
    def __init__(self, text_key, class_token_key, text_encoder_config, fusion_module_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = instantiate_from_config(text_encoder_config)
        self.fusion_module = instantiate_from_config(fusion_module_config)
        self.text_key = text_key
        self.class_token_key = class_token_key
    
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        ref_image = batch[self.cond_stage_key]
        B, N, H, W, C = ref_image.shape
        x, cond = super().get_input(batch, k, bs, *args, **kwargs)
        # cond['c_crossattn'][0].shape: (B, N, 1, 1024)
        subject_embedding = cond['c_crossattn'][0]
        # subject_embedding.shape: (B, N, 1024)
        subject_embedding = subject_embedding.reshape(B, N, -1)
        caption = batch[self.text_key]  # list[str]
        # (B, N)
        class_token_ids = batch[self.class_token_key]
        # (B, 77, 1024)
        caption_embedding = self.text_encoder(caption).last_hidden_state.to(subject_embedding)
        
        batch_ids = torch.arange(B)[..., None].repeat(1, N)
        class_token_embedding = caption_embedding[batch_ids, class_token_ids]
        fuse_token_embedding = self.fusion_module(class_token_embedding, subject_embedding)
        caption_embedding[batch_ids, class_token_ids] = fuse_token_embedding.to(caption_embedding)
        cond['c_crossattn'] = [caption_embedding]
        return x, cond
    
    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        # params += list(self.cond_stage_model.projector.parameters())
        params += list(self.fusion_module.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt
    
    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z) 

        # ==== visualize the shape mask or the high-frequency map ====
        guide_mask = (c_cat[:, -1, :, :].unsqueeze(1) + 1) * 0.5
        guide_mask = torch.cat([guide_mask, guide_mask, guide_mask], 1)
        HF_map  = c_cat[:, :3, :, :]  # * 2.0 - 1.0

        log["control"] = HF_map

        cond_image = batch[self.cond_stage_key].cpu().numpy().copy()    # (B, N, 224, 224, 3)
        cond_image = torch.from_numpy(cond_image)[:N]
        cond_image = torch.cat([cond_image[:, i, ...] for i in range(cond_image.shape[1])], dim=2)
        log["conditioning"] = torch.permute(cond_image, (0, 3, 1, 2)) * 2.0 - 1.0  
        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N, batch)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg #* 2.0 - 1.0
        return log
    
    @torch.no_grad()
    def get_unconditional_conditioning(self, N, batch):
        uncond = {}
        uncond['subject'] = torch.zeros((N, 2, 1024))
        uncond['caption'] = batch['caption'][:N]
        uncond['class_token_ids'] = batch['class_token_ids'][:N]
        uncond = self.get_uncond(uncond)
        return uncond
    
    def get_uncond(self, uncond):
        # c.shape: (N, 2, 1024) 
        c = uncond['subject'].to('cuda')
        caption = uncond['caption']
        class_token_ids = uncond['class_token_ids']
        N = c.shape[0]
        if self.text_encoder is not None:
            # uncond_txt: (N, 77, 1024)
            uncond_txt = self.text_encoder(caption).last_hidden_state
            
            batch_ids = torch.arange(N)[..., None].repeat(1, 2)
            class_token_embedding = uncond_txt[batch_ids, class_token_ids]
            fuse_token_embedding = self.fusion_module(class_token_embedding, c).detach()
            fuse_token_embedding.requires_grad_(False)
            uncond_txt[batch_ids, class_token_ids] = fuse_token_embedding
        else:
            uncond_txt = torch.zeros((N, 77, 1024))
        
        uncond = uncond_txt
        return uncond