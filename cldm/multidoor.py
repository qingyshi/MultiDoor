from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import instantiate_from_config
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torchvision.utils import make_grid     
        

class MultiDoor(LatentDiffusion):
    def __init__(self, control_image_key, ref_image_key, image_stage_config, control_stage_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_encoder = instantiate_from_config(image_stage_config)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_image_key = control_image_key
        self.ref_image_key = ref_image_key
    
    def get_input(self, batch, k, *args, **kwargs):
        x, c = super().get_input(batch, k, *args, **kwargs) # c.shape = (b, 77, 1024)
        control = batch[self.control_image_key]
        control = control.to(self.device)
        control = rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        self.time_steps = batch['time_steps']
        
        ref_image = batch[self.ref_image_key] # ref_image.shape: (b, n, 224, 224, 3)
        image_token = self.image_encoder(ref_image) # image_token.shape: (b, n * 257, 1024)
        
        cond = dict(
            c_crossattn = c,
            c_concat = control,
            image_token = image_token,
            hf_map = control[:, :3, :, :]
        )
        return x, cond
    
    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_text = cond['c_crossattn']
        cond_subject = cond['image_token']
        control = cond['c_concat']

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, caption=cond_text, subject=cond_subject, control=None, only_mid_control=self.only_mid_control)
        else:
            control = self.control_model(x=x_noisy, hint=control, timesteps=t, caption=cond_subject)
            eps = diffusion_model(x=x_noisy, timesteps=t, caption=cond_text, subject=cond_subject, control=control, only_mid_control=self.only_mid_control)   
        return eps
    
    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.image_encoder.projector.parameters())
        params += list(self.model.diffusion_model.output_blocks.parameters())
        params += list(self.model.diffusion_model.out.parameters())
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
        HF_map = c_cat[:, :-1, :, :]

        log["control"] = HF_map

        cond_image = batch[self.ref_image_key][:N].clone()    # (b, 2, 224, 224, 3)
        cond_image = cond_image.transpose(1, 2).flatten(2, 3)
        log["conditioning"] = torch.permute(cond_image, (0, 3, 1, 2)) * 2.0 - 1.0  

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
    def get_unconditional_conditioning(self, num_samples, image_guidance=False):
        if image_guidance:
            uncond = torch.zeros((num_samples, 2, 224, 224, 3)).to('cuda')
            uncond = self.image_encoder(uncond)
        else:
            uncond = [""] * num_samples
            uncond = self.cond_stage_model(uncond)
        return uncond   # (b, n * 257, 1024) or (b, 77, 1024)