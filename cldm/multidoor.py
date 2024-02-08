from .cldm import ControlLDM
from ldm.util import instantiate_from_config
import torch
from torch import nn


class FusionModule(nn.Module):
    def __init__(self, text_dim, subject_dim):
        super().__init__()
        self.text_dim = text_dim
        self.subject_dim = subject_dim
        self.fusion_dim = text_dim + subject_dim
        self.MLP = nn.Sequential(
            nn.Linear(self.fusion_dim, self.text_dim),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.text_dim, self.text_dim)
        )
    
    def forward(self, fusion_emb):
        return self.MLP(fusion_emb)
         

class MultiDoor(ControlLDM):
    def __init__(self, text_key, class_token_key, text_encoder_config, fusion_module_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = instantiate_from_config(text_encoder_config)
        self.fusion_module = instantiate_from_config(fusion_module_config)
        self.text_key = text_key
        self.class_token_key = class_token_key
    
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        ref_image = batch[self.cond_stage_key]
        B, N, H, W, C = ref_image.shape
        batch[self.cond_stage_key] = torch.from_numpy(ref_image).flatten(0, 1)
        x, cond = super().get_input(batch, k, bs, *args, **kwargs)
        subject_embedding = cond['c_crossattn'][0][:, 0, ...]
        subject_embedding = subject_embedding.reshape(B, N, -1)
        caption = batch[self.text_key]
        class_token_ids = torch.tensor(batch[self.class_token_key])
        caption_embedding = self.text_encoder.encode(caption)
        batch_ids = torch.arange(B)[..., None].repeat(1, N)
        class_token_embedding = caption_embedding[batch_ids, class_token_ids]
        fuse_token_embedding = torch.cat([subject_embedding, 
                                          class_token_embedding], dim=-1)
        fuse_token_embedding = self.fusion_module(fuse_token_embedding)
        caption_embedding[batch_ids, caption] = fuse_token_embedding
        cond['c_crossattn'] = [caption_embedding]
        return x, cond