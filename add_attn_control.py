from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import torch
from ldm.modules.encoders.modules import FrozenMultiDoorEncoder


image_encoder = FrozenMultiDoorEncoder()

cfg = OmegaConf.load('configs/multidoor.yaml')
unet_cfg = cfg.model.params.unet_config

def add_unet_crossattn_map_store(unet, corss_attn_map_store):
    count = 0
    def store_attn_map(name, cross_attn_map_store):
        def store_attn_map_(attn_map):
            cross_attn_map_store[name] = attn_map
        
        return store_attn_map_

    for name, module in unet.named_modules():
        if 'attn2' in name and module.__class__.__name__ == 'CrossAttention':
            module.cross_attn_map_store = store_attn_map(name, cross_attn_map_store)
            count += 1
    print(f"total layers: {count}")
    return unet
    
unet = instantiate_from_config(unet_cfg)
cross_attn_map_store = {}
unet = add_unet_crossattn_map_store(unet=unet, corss_attn_map_store=cross_attn_map_store)
x = torch.randn((1, 9, 64, 64))
t = torch.tensor([999])
context = torch.randn((1, 77, 1024))

out = unet(x, timesteps=t, context=context)
for k, v in cross_attn_map_store.items():
    print(k, '----------->', v.shape)