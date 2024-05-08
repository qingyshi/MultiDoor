from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import instantiate_from_config
from cldm.utils import add_unet_crossattn_map_store, fuse_object_embeddings, get_object_localization_loss

from typing import Optional, Tuple, Union
import gc
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torchvision.utils import make_grid
from transformers import CLIPTextModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import (
    _expand_mask,
    CLIPTextTransformer,
    CLIPPreTrainedModel,
)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, use_residual=True):
        super().__init__()
        if use_residual:
            assert in_dim == out_dim
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.use_residual = use_residual
        self.act_fn = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.layernorm(x)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        if self.use_residual:
            x = x + residual
        return x


class MultiDoorPostfuseModule(nn.Module):
    def __init__(self, text_token_dim, image_token_dim):
        super().__init__()
        embed_dim = text_token_dim + image_token_dim
        self.mlp1 = MLP(embed_dim, text_token_dim, text_token_dim, use_residual=False)
        self.mlp2 = MLP(text_token_dim, text_token_dim, text_token_dim, use_residual=True)
        self.layer_norm = nn.LayerNorm(text_token_dim)

    def fuse_fn(self, text_embeds, object_embeds):
        text_object_embeds = torch.cat([text_embeds, object_embeds], dim=-1)
        text_object_embeds = self.mlp1(text_object_embeds) + text_embeds
        text_object_embeds = self.mlp2(text_object_embeds)
        text_object_embeds = self.layer_norm(text_object_embeds)
        return text_object_embeds

    def forward(
        self,
        text_embeds,
        object_embeds,
        image_token_mask,
        num_objects,
    ) -> torch.Tensor:
        text_object_embeds = fuse_object_embeddings(
            text_embeds, image_token_mask, object_embeds, num_objects, self.fuse_fn)

        return text_object_embeds


class MultiDoorTextEncoder(CLIPPreTrainedModel):
    _build_causal_attention_mask = CLIPTextTransformer._build_causal_attention_mask

    def __init__(self, model_name_or_path):
        model = CLIPTextModel.from_pretrained(model_name_or_path, subfolder="text_encoder")
        text_model = model.text_model
        super().__init__(text_model.config)
        self.config = text_model.config
        self.final_layer_norm = text_model.final_layer_norm
        self.embeddings = text_model.embeddings
        self.encoder = text_model.encoder
        self.freeze()
    
    def freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(
        self,
        input_ids,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_ids)

        bsz, seq_len = input_shape
        causal_attention_mask = self._build_causal_attention_mask(
            bsz, seq_len, hidden_states.dtype
        ).to(hidden_states.device)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(
                dim=-1
            ),
        ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class BalancedL1Loss(nn.Module):
    def __init__(self, threshold=1.0, normalize=False):
        super().__init__()
        self.threshold = threshold
        self.normalize = normalize

    def forward(self, object_token_attn_prob, object_segmaps):
        if self.normalize:
            object_token_attn_prob = object_token_attn_prob / (
                object_token_attn_prob.max(dim=2, keepdim=True)[0] + 1e-5
            )
        background_segmaps = 1 - object_segmaps
        background_segmaps_sum = background_segmaps.sum(dim=2) + 1e-5
        object_segmaps_sum = object_segmaps.sum(dim=2) + 1e-5

        background_loss = (object_token_attn_prob * background_segmaps).sum(
            dim=2
        ) / background_segmaps_sum

        object_loss = (object_token_attn_prob * object_segmaps).sum(
            dim=2
        ) / object_segmaps_sum

        return background_loss - object_loss

class MultiDoor(LatentDiffusion):
    def __init__(
        self, 
        image_key, 
        inpaint_image_key, 
        object_localization,
        object_localization_weight,
        object_localization_threshold,
        object_localization_normalize,
        image_cond_config, 
        fusion_stage_config, 
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.image_encoder = instantiate_from_config(image_cond_config)
        self.fuser: MultiDoorPostfuseModule = instantiate_from_config(fusion_stage_config)
        self.image_key = image_key
        self.inpaint_image_key = inpaint_image_key
        self.object_localization = object_localization
        self.object_localization_weight = object_localization_weight
        if self.object_localization:
            self.cross_attn_map_store = {}
            unet = self.model.diffusion_model
            self.model.diffusion_model = add_unet_crossattn_map_store(unet, self.cross_attn_map_store)
            self.object_localization_loss_fn = BalancedL1Loss(
                object_localization_threshold,
                object_localization_normalize,
            )
        self.pad_token_id = 0
    
    def _clear_cross_attention_scores(self, ):
        if hasattr(self, "cross_attn_map_store"):
            keys = list(self.cross_attn_map_store.keys())
            for k in keys:
                del self.cross_attn_map_store[k]
        gc.collect()
    
    def get_input(self, batch, k, *args, **kwargs):
        x, c = super().get_input(batch, k, *args, **kwargs)
        if isinstance(c, BaseModelOutputWithPooling):
            text_token = c.last_hidden_state  # text_token.shape = (b, 77, 1024)
        elif isinstance(c, torch.Tensor):
            text_token = c
        else:
            raise Exception
        control = batch[self.inpaint_image_key]
        control = control.to(self.device)
        control = rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        self.time_steps = batch.get('time_steps')
        # cond['c_crossattn'].shape: (b, 77, 1024)
        inpaint = control[:, :3, :, :]
        mask = control[:, -1, :, :].unsqueeze(1)
        inpaint = self.encode_first_stage(inpaint)  # (b, 4, 64, 64)
        inpaint = self.get_first_stage_encoding(inpaint).detach()
        b, _, h, w = inpaint.shape
        mask = F.interpolate(mask, size=(h, w), mode="nearest")
        c_concat = torch.cat([inpaint, mask], dim=1)
        image = batch[self.image_key] # image.shape: (b, n, 224, 224, 3)
        image_token = self.image_encoder(image) # image_token.shape: (b, n, 1, 1536)
        
        image_token_masks = batch["image_token_masks"]  # (b, 77)
        image_token_ids = batch["image_token_ids"]  # (b, max_num_objects)
        image_token_ids_mask = batch["image_token_ids_mask"]    # (b, max_num_objects)
        target_masks = batch["target_masks"]   # (b, n, 512, 512)
        num_objects = batch["num_objects"]  # (b, 1)
        
        context = self.fuser(
            text_token, # (b, 77, 1024)
            image_token,    # (b, n, 1, 1536)
            image_token_masks,  # (b, 77)
            num_objects, # (b, 1)
        )
        
        cond = dict(
            c_crossattn = context,
            c_concat = c_concat,
            hf_map = control[:, :3, :, :],
            image_token_ids = image_token_ids,
            image_token_ids_mask = image_token_ids_mask,
            target_masks = target_masks            
        )
        return x, cond
    
    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        context = cond['c_crossattn']
        inpaint = cond['c_concat']
        attn_bias = cond.get('attn_bias', None) # dict (res ** 2, 77)
        
        input = torch.cat([x_noisy, inpaint], dim=1) # (b, 9, h, w)
        eps = diffusion_model(x=input, timesteps=t, context=context)
        return eps
    
    def p_losses(self, x_start, cond, t, noise=None):
        loss, loss_dict = super().p_losses(x_start, cond, t)
        if self.object_localization:
            target_masks = cond["target_masks"]    # (bsz, 2, 512, 512)
            image_token_ids = cond["image_token_ids"]   # (bsz, 2)
            image_token_ids_mask = cond["image_token_ids_mask"]    # (bsz, 2)
            localization_loss = get_object_localization_loss(
                self.cross_attn_map_store,
                target_masks,     # (bsz, 2, 512, 512)
                image_token_ids,    # (bsz, 2)
                image_token_ids_mask,   # (bsz, 2)
                self.object_localization_loss_fn,
            )
            loss_dict["localization_loss"] = localization_loss
            loss = self.object_localization_weight * localization_loss + loss
            self._clear_cross_attention_scores()
        return loss, loss_dict
    
    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.image_encoder.model.blocks[-2:].parameters())
        params += list(self.model.diffusion_model.parameters())
        params += list(self.fuser.parameters())
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
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value[:N]
        z, cond = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat, context = cond["c_concat"], cond["c_crossattn"]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z) 

        # ==== visualize the shape mask or the high-frequency map ====
        guide_mask = (c_cat[:, -1, :, :].unsqueeze(1) + 1) * 0.5
        guide_mask = torch.cat([guide_mask, guide_mask, guide_mask], 1)
        HF_map = cond["hf_map"]

        log["control"] = HF_map

        cond_image = batch[self.image_key][:N].clone()    # (b, 2, 224, 224, 3)
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
            samples, z_denoise_row = self.sample_log(cond={"c_concat": c_cat, "c_crossattn": context},
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
            uc_full = {"c_concat": uc_cat, "c_crossattn": uc_cross}
            samples_cfg, _ = self.sample_log(cond={"c_concat": c_cat, "c_crossattn": context},
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
    def get_unconditional_conditioning(self, num_samples):
        uncond = torch.tensor([self.pad_token_id] * num_samples).unsqueeze(1).repeat(1, 77).to("cuda")
        uncond = self.cond_stage_model(uncond).last_hidden_state
        return uncond   # (b, 77, 1024)