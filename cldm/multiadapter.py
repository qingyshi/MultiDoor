from .cldm import ControlLDM
from ldm.util import instantiate_from_config
import torch
from torch import nn
from einops import rearrange, repeat
from torchvision.utils import make_grid     
        

class MultiAdapter(ControlLDM):
    def __init__(self, ref_img_key, img_cond_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.img_encoder = instantiate_from_config(img_cond_config)
        self.ref_img_key = ref_img_key
    
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, cond = super().get_input(batch, k, bs, *args, **kwargs)
        # cond['c_crossattn'].shape: (b, 77, 1024)
        ref_image = batch[self.ref_img_key] # ref_image.shape: (b, n, 224, 224, 3)
        img_token = self.img_encoder(ref_image) # img_token.shape: (b, n * 257, 1024)
        cond['img_token'] = [img_token]
        return x, cond
    
    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_text = torch.cat(cond['c_crossattn'], dim=1)
        cond_subject = torch.cat(cond['img_token'], dim=1)

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, subject=cond_subject, caption=cond_text, control=None, only_mid_control=self.only_mid_control)
        else:
            control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_subject)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, caption=cond_text, subject=cond_subject, control=control, only_mid_control=self.only_mid_control)
        return eps
    
    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        params += list(self.img_encoder.projector.parameters())
        params += list(self.model.diffusion_model.output_blocks.parameters())
        params += list(self.model.diffusion_model.out.parameters())
        params.append(self.img_encoder.obj_emb)
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
        c_cat, c, ref = c["c_concat"][0][:N], c["c_crossattn"][0][:N], c['img_token'][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z) 

        # ==== visualize the shape mask or the high-frequency map ====
        guide_mask = (c_cat[:, -1, :, :].unsqueeze(1) + 1) * 0.5
        guide_mask = torch.cat([guide_mask, guide_mask, guide_mask], 1)
        HF_map  = c_cat[:, :3, :, :]  # * 2.0 - 1.0

        log["control"] = HF_map

        cond_image = batch[self.ref_img_key][:N].clone()    # (b, 2, 224, 224, 3)
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
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c], "img_token": [ref]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [c], "img_token": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c], "img_token": [ref]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg #* 2.0 - 1.0
        return log
    
    @torch.no_grad()
    def get_unconditional_conditioning(self, num_samples, image_guidance=True):
        if image_guidance:
            uncond = torch.zeros((num_samples, 2, 224, 224, 3)).to('cuda')
            uncond = self.img_encoder(uncond)
        else:
            uncond = [" "] * num_samples
            uncond = self.cond_stage_model(uncond)
        return uncond   # (b, n * 257, 1024) or (b, 77, 1024)