import cv2
import einops
import numpy as np
import torch
from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from cldm.hack import disable_verbosity, enable_sliced_attention
from cldm.multiadapter import MultiAdapter
from datasets.data_utils import * 
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import albumentations as A
from omegaconf import OmegaConf
from datasets.coco_val import CocoValDataset


save_memory = False
disable_verbosity()
if save_memory:
    enable_sliced_attention()

DConf = OmegaConf.load('./configs/datasets.yaml')
dataset = CocoValDataset(**DConf.Train.COCOVal)

config = OmegaConf.load('./configs/inference.yaml')
model_ckpt =  config.pretrained_model
model_config = config.config_file

model: MultiAdapter = create_model(model_config).cpu()
model.load_state_dict(load_state_dict(model_ckpt, location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)

def inference_single_image(item, guidance_scale = 5.0):
    '''
    inputs:
        ref_image.shape: (H, W, 3) or [(H1, W1, 3), (H2, W2, 3)]
        ref_mask.shape: [(H, W), (H, W)] or [(H1, W1), (H2, W2)]
        tar_image.shape: (H, W, 3)
        tar_mask.shape: [(H, W), (H, W)]
    '''

    ref = item['ref']
    tar = item['jpg'] 
    hint = item['hint']
    caption = item['caption']
    num_samples = 1

    control = torch.from_numpy(hint.copy()).float().cuda() 
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()

    dino_input = torch.from_numpy(ref.copy()).float().cuda() 
    dino_input = torch.stack([dino_input for _ in range(num_samples)], dim=0)
    dino_input = dino_input.clone().to('cuda')
    ref_token = model.img_encoder(dino_input)   # (b, n * 257, 1024)

    guess_mode = False
    H, W = 512, 512
    
    clip_input = caption * num_samples
    cond_text = model.get_learned_conditioning(clip_input)  # (b, 77, 1024)
    uncond = model.get_unconditional_conditioning(num_samples)  # (b, n * 257, 1024)
    # caption_embedding: (b, 77, 1024)
    
    cond = {"c_concat": [control], "c_crossattn": [cond_text], "img_token": [ref_token]}
    un_cond = {"c_concat": None if guess_mode else [control], 
               "c_crossattn": [cond_text], "img_token": [uncond]}
    shape = (4, H // 8, W // 8)

    if save_memory:
        model.low_vram_shift(is_diffusing=True)

    # ====
    num_samples = 1 # gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
    image_resolution = 512  # gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
    strength = 1  # gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
    guess_mode = False # gr.Checkbox(label='Guess Mode', value=False)
    # detect_resolution = 512  #gr.Slider(label="Segmentation Resolution", minimum=128, maximum=1024, value=512, step=1)
    ddim_steps = 50 # gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
    scale = guidance_scale  # gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
    seed = -1  # gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
    eta = 0.0   # gr.Number(label="eta (DDIM)", value=0.0)

    model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
    samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                 shape, cond, verbose=False, eta=eta,
                                                 unconditional_guidance_scale=scale,
                                                 unconditional_conditioning=un_cond)
    if save_memory:
        model.low_vram_shift(is_diffusing=False)

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy()

    pred = x_samples[0]
    pred = np.clip(pred, 0, 255)
    h, w, _ = tar.shape
    pred = cv2.resize(pred, (w, h))
    return pred


if __name__ == '__main__': 
    count = 98
    while True:
        save_path = f'examples/cocoval/visual{count}.jpg'
        data = next(iter(dataset))
        back_image = data['jpg'] * 127.5 + 127.5
        ref_image = data['ref'] * 127.5 + 127.5
        collage = data['hint'] * 127.5 + 127.5
        gen_image = inference_single_image(item=data)
        h, w = back_image.shape[0], back_image.shape[1]
        if len(ref_image) == 2:
            ref_image = [cv2.resize(ref, (w, h)) for ref in ref_image]
            collage = cv2.resize(collage, (w, h))
            vis_image = cv2.hconcat([ref_image[0].astype(np.float32), 
                                     ref_image[1].astype(np.float32), 
                                     collage[:, :, :-1].astype(np.float32), 
                                     back_image, gen_image])
        else:
            ref_image = cv2.resize(ref_image, (w, h))
            vis_image = cv2.hconcat([ref_image, back_image, gen_image])
        
        cv2.imwrite(save_path, vis_image[:, :, ::-1])
        print(data['caption'])
        print('finish!')
        count += 1