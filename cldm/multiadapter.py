from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import instantiate_from_config
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torchvision.utils import make_grid     
        

class MultiAdapter(LatentDiffusion):
    def __init__(self, control_image_key, ref_image_key, image_cond_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_encoder = instantiate_from_config(image_cond_config)
        self.control_image_key = control_image_key
        self.ref_image_key = ref_image_key
    
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, k, *args, **kwargs) # c.shape = (b, 77, 1024)
        control = batch[self.control_image_key]
        control = control.to(self.device)
        control = rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        self.time_steps = batch['time_steps']
        # cond['c_crossattn'].shape: (b, 77, 1024)
        inpaint = control[:, :3, :, :]
        mask = control[:, -1, :, :].unsqueeze(1)
        inpaint = self.encode_first_stage(inpaint)  # (b, 4, 64, 64)
        inpaint = self.get_first_stage_encoding(inpaint).detach()
        b, _, h, w = inpaint.shape
        mask = F.interpolate(mask, size=(h, w), mode="nearest")
        ref_image = batch[self.ref_image_key] # ref_image.shape: (b, n, 224, 224, 3)
        image_token = self.image_encoder(ref_image) # image_token.shape: (b, n * 256, 1024)
        cond = dict(
            c_crossattn=c,
            c_concat = torch.cat([inpaint, mask], dim=1),
            image_token = image_token,
            hf_map = control[:, :3, :, :]
        )
        return x, cond
    
    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_text = cond['c_crossattn']
        cond_subject = cond['image_token']
        inpaint = cond['c_concat']
        input = torch.cat([x_noisy, inpaint], dim=1) # (b, 9, h, w)
        eps = diffusion_model(x=input, timesteps=t, caption=cond_text, subject=cond_subject)
        return eps
    
    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.image_encoder.projector.parameters())
        params += list(self.model.diffusion_model.parameters())
        params.append(self.image_encoder.obj_emb)
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
        z, cond = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat, c, ref = cond["c_concat"][:N], cond["c_crossattn"][:N], cond['image_token'][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z) 

        # ==== visualize the shape mask or the high-frequency map ====
        guide_mask = (c_cat[:, -1, :, :].unsqueeze(1) + 1) * 0.5
        guide_mask = torch.cat([guide_mask, guide_mask, guide_mask], 1)
        HF_map  = cond["hf_map"]

        log["control"] = HF_map

        cond_image = batch[self.ref_image_key][:N].clone()    # (b, 2, 224, 224, 3)
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
            samples, z_denoise_row = self.sample_log(cond={"c_concat": c_cat, 
                                                           "c_crossattn": c, 
                                                           "image_token": ref},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N, image_guidance=False)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": uc_cat, 
                       "c_crossattn": uc_cross, 
                       "image_token": ref}
            samples_cfg, _ = self.sample_log(cond={"c_concat": c_cat, 
                                                   "c_crossattn": c, 
                                                   "image_token": ref},
                                             batch_size=N, 
                                             ddim=use_ddim,
                                             ddim_steps=ddim_steps, 
                                             eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full)
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg #* 2.0 - 1.0
        return log
    
    @torch.no_grad()
    def get_unconditional_conditioning(self, num_samples, image_guidance=True):
        if image_guidance:
            uncond = torch.zeros((num_samples, 2, 224, 224, 3)).to('cuda')
            uncond = self.image_encoder(uncond)
        else:
            uncond = [" "] * num_samples
            uncond = self.cond_stage_model(uncond)
        return uncond   # (b, n * 257, 1024) or (b, 77, 1024)