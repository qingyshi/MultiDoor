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


class MultiDoor(LatentDiffusion):
    def __init__(
        self, 
        image_key, 
        inpaint_image_key, 
        control_stage_config,
        image_cond_config, 
        fusion_stage_config, 
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.image_encoder = instantiate_from_config(image_cond_config)
        self.fuser: MultiDoorPostfuseModule = instantiate_from_config(fusion_stage_config)
        self.image_key = image_key
        self.inpaint_image_key = inpaint_image_key
        self.pad_token_id = 0
    
    def get_input(self, batch, k, *args, **kwargs):
        x, c = super().get_input(batch, k, *args, **kwargs)
        if isinstance(c, BaseModelOutputWithPooling):
            text_token = c.last_hidden_state  # text_token.shape = (b, 77, 1024)
        elif isinstance(c, torch.Tensor):
            text_token = c
        else:
            raise Exception

        self.time_steps = batch.get('time_steps')
        control = batch[self.inpaint_image_key]
        control = control.to(self.device)
        control = rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()

        image = batch[self.image_key] # image.shape: (b, n, 224, 224, 3)
        patch_token, cls_token = self.image_encoder(image) # cls_token.shape: (b, n, 1, 1536)
        
        image_token_masks = batch["image_token_masks"]  # (b, 77)
        image_token_ids = batch["image_token_ids"]  # (b, max_num_objects)
        image_token_ids_mask = batch["image_token_ids_mask"]    # (b, max_num_objects)
        target_masks = batch["target_masks"]   # (b, n, 512, 512)
        num_objects = batch["num_objects"]  # (b, 1)
        
        context = self.fuser(
            text_token, # (b, 77, 1024)
            cls_token,    # (b, n, 1, 1536)
            image_token_masks,  # (b, 77)
            num_objects, # (b, 1)
        )
        
        cond = dict(
            c_ip = patch_token,    # (b, n * 256, 1024)
            c_crossattn = context,
            c_concat = control,
            hf_map = control[:, :3],
            image_token_ids = image_token_ids,
            image_token_ids_mask = image_token_ids_mask,
            target_masks = target_masks            
        )
        return x, cond
    
    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        ip = cond['c_ip']
        context = cond['c_crossattn']
        inpaint = cond['c_concat']
        control = self.control_model(x=x_noisy, hint=inpaint, timesteps=t, context=ip)
        eps = diffusion_model(x=x_noisy, timesteps=t, context=context, ip=ip, control=control)
        return eps
    
    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.fuser.parameters())
        params += list(self.image_encoder.project.parameters())
        params += list(self.control_model.parameters())
        params += list(self.model.diffusion_model.output_blocks.parameters())
        params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt
    
    @torch.no_grad()
    def log_images(
        self, 
        batch, 
        N=4, 
        n_row=2, 
        sample=False, 
        ddim_steps=50, 
        ddim_eta=0.0, 
        unconditional_guidance_scale=9.0,
        **kwargs
    ):
        use_ddim = ddim_steps is not None

        log = dict()
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value[:N]
        z, cond = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat, context, ip = cond["c_concat"], cond["c_crossattn"], cond["c_ip"]
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
        
        if unconditional_guidance_scale > 1.0:
            uc_text, uc_image = self.get_unconditional_conditioning(N)
            uc_full = {"c_concat": c_cat, "c_crossattn": uc_text, "c_ip": uc_image}
            samples_cfg, _ = self.sample_log(cond={"c_concat": c_cat, 
                                                   "c_crossattn": context,
                                                   "c_ip": ip},
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
        uncond_text = torch.tensor([self.pad_token_id] * num_samples).unsqueeze(1).repeat(1, 77).cuda()
        uncond_text = self.cond_stage_model(uncond_text).last_hidden_state
        uncond_image = torch.zeros(num_samples, 512, 1024).cuda()
        return uncond_text, uncond_image   # (b, 77, 1024)