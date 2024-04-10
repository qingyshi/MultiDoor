import cv2
import einops
import numpy as np
import torch
from torch.nn import functional as F
from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from cldm.hack import disable_verbosity, enable_sliced_attention
from cldm.multidoor import MultiDoor
from datasets.data_utils import * 
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import albumentations as A
from omegaconf import OmegaConf
from PIL import Image
import os
from transformers import CLIPTokenizer


save_memory = False
disable_verbosity()
if save_memory:
    enable_sliced_attention()


config = OmegaConf.load('./configs/inference.yaml')
model_ckpt = config.pretrained_model
model_config = config.config_file

model: MultiDoor = create_model(model_config).cpu()
model.load_state_dict(load_state_dict(model_ckpt, location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)



def aug_data_mask(image, mask):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        ])
    transformed = transform(image=image.astype(np.uint8), mask = mask)
    transformed_image = transformed["image"]
    transformed_mask = transformed["mask"]
    return transformed_image, transformed_mask

def process_nouns_format(nouns, caption):
    if isinstance(caption, list):
        caption = caption[0]
    _nouns = []
    for noun in nouns:
        start = caption.index(noun)
        end = start + len(noun)
        word = noun
        noun = dict(
            word = noun,
            start = start,
            end = end
        )
        _nouns.append(noun)
    return _nouns

def process_nouns_in_caption(nouns, caption, tokenizer, image_token, image_token_id):
    nouns = sorted(nouns, key=lambda x: x["end"], reverse=True)
    for noun in nouns:
        end = noun["end"]
        caption = caption[:end] + image_token + caption[end:]

    input_ids = tokenizer(
        caption, 
        truncation=True, 
        max_length=77,
        padding="max_length", 
        return_tensors="pt"
    ).input_ids[0]
    noun_phrase_end_mask = [False for _ in input_ids]
    clean_input_ids = []
    track_idx = 0
    for idx in input_ids:
        if idx == image_token_id:
            noun_phrase_end_mask[track_idx - 1] = True
        else:
            track_idx += 1
            clean_input_ids.append(idx)
            
    max_len = tokenizer.model_max_length
    if len(clean_input_ids) > max_len:
        clean_input_ids = clean_input_ids[:max_len]
    else:
        clean_input_ids = clean_input_ids + [tokenizer.pad_token_id] * (
            max_len - len(clean_input_ids)
        )
    noun_phrase_end_mask = torch.tensor(noun_phrase_end_mask, dtype=torch.bool)
    assert noun_phrase_end_mask.sum() == 2
    clean_input_ids = torch.tensor(clean_input_ids)
    
    image_token_ids = torch.nonzero(noun_phrase_end_mask[None], as_tuple=True)[1]
    image_token_ids_mask = torch.ones_like(image_token_ids, dtype=torch.bool)
    if len(image_token_ids) < 2:
        image_token_ids = torch.cat(
            [
                image_token_ids,
                torch.zeros(2 - len(image_token_ids), dtype=torch.long),
            ]
        )
        image_token_ids_mask = torch.cat(
            [
                image_token_ids_mask,
                torch.zeros(
                    2 - len(image_token_ids_mask),
                    dtype=torch.bool,
                ),
            ]
        )
    return dict(
        caption=clean_input_ids,
        image_token_masks=noun_phrase_end_mask,
        image_token_ids=image_token_ids,
        image_token_ids_mask=image_token_ids_mask,
    )

def process_single_ref_image(ref_image, ref_mask, tar_image, tar_mask):
    '''
    inputs:
        ref_image: (H, W, 3)
        ref_mask: [(H, W), (H, W)]
        tar_image: (H, W, 3)
        tar_mask: [(H, W), (H, W)]
    
    outputs:
        ref: (2, 224, 224, 3)
        
    '''
    # ========= Reference ===========
        # Get the outline Box of the reference image
    multi_subject_ref_image = []
    multi_subject_ref_mask = []
    
    for single_mask in ref_mask:
        ref_box_yyxx = get_bbox_from_mask(single_mask)
    
        # Filtering background for the reference image
        ref_mask_3 = np.stack([single_mask, single_mask, single_mask], -1)
        masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1 - ref_mask_3)

        y1, y2, x1, x2 = ref_box_yyxx
        masked_ref_image = masked_ref_image[y1: y2, x1: x2, :]
        single_mask = single_mask[y1: y2, x1: x2]

        ratio = np.random.randint(11, 15) / 10 
        masked_ref_image, single_mask = expand_image_mask(masked_ref_image, single_mask, ratio=ratio)
        ref_mask_3 = np.stack([single_mask, single_mask, single_mask], -1)

        # Padding reference image to square and resize to 224
        masked_ref_image = pad_to_square(masked_ref_image, pad_value = 255, random = False)
        masked_ref_image = cv2.resize(masked_ref_image.astype(np.uint8), (224, 224)).astype(np.uint8)

        ref_mask_3 = pad_to_square(ref_mask_3 * 255, pad_value = 0, random = False)
        ref_mask_3 = cv2.resize(ref_mask_3.astype(np.uint8), (224, 224)).astype(np.uint8)
        single_mask = ref_mask_3[:, :, 0]

        # Augmenting reference image
        # masked_ref_image_aug = self.aug_data(masked_ref_image) 
        
        # Getting for high-freqency map
        masked_ref_image_compose, ref_mask_compose = aug_data_mask(masked_ref_image, single_mask) 
        masked_ref_image_aug = masked_ref_image_compose.copy()
        multi_subject_ref_image.append(masked_ref_image_aug)
        multi_subject_ref_mask.append(ref_mask_compose)

    ref_mask_compose = np.concatenate(multi_subject_ref_mask, axis=1)
    ref_mask_3 = np.stack([ref_mask_compose, ref_mask_compose, ref_mask_compose], -1)
    masked_ref_image_compose = np.stack(multi_subject_ref_image, axis=0)
    masked_ref_image_aug = masked_ref_image_compose.copy()  # (2, 224, 244, 3)
    # ref_image_collage = sobel(masked_ref_image_compose, ref_mask_compose / 255) # (224, 448, 3)
    multi_ref_image_collage = [sobel(masked_ref_image_compose, ref_mask_compose / 255) for 
                                masked_ref_image_compose in multi_subject_ref_image]

    

    # ========= Training Target ===========
    multi_subject_bbox = []
    multi_subject_bbox_crop = []
    
    for single_mask in tar_mask:
        tar_box_yyxx = get_bbox_from_mask(single_mask)
        tar_box_yyxx = expand_bbox(single_mask, tar_box_yyxx, ratio=[1.1, 1.2]) # 1.1  1.3
        multi_subject_bbox.append(tar_box_yyxx)
        
        # Cropping around the target object 
        tar_box_yyxx_crop = expand_bbox(tar_image, tar_box_yyxx, ratio=[1.3, 3.0])   
        tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop) # crop box
        multi_subject_bbox_crop.append(tar_box_yyxx_crop)
    
    # bbox which contains multi-subjects
    y1 = min([bbox[0] for bbox in multi_subject_bbox_crop])
    x1 = min([bbox[2] for bbox in multi_subject_bbox_crop])
    y2 = max([bbox[1] for bbox in multi_subject_bbox_crop])
    x2 = max([bbox[3] for bbox in multi_subject_bbox_crop])
    tar_box_yyxx_crop = (y1, y2, x1, x2)
    cropped_target_image = tar_image[y1: y2, x1: x2, :]
    collage = cropped_target_image.copy()
    collage_mask = cropped_target_image.copy() * 0.0
    tar_mask = np.max(tar_mask, axis=0)
    
    for single_bbox, ref_image_collage in zip(multi_subject_bbox, multi_ref_image_collage):
        tar_box_yyxx = box_in_box(single_bbox, tar_box_yyxx_crop)
        y1, y2, x1, x2 = tar_box_yyxx

        # Prepairing collage image
        ref_image_collage = cv2.resize(ref_image_collage.astype(np.uint8), (x2-x1, y2-y1))
        ref_mask_compose = cv2.resize(ref_mask_compose.astype(np.uint8), (x2-x1, y2-y1))
        ref_mask_compose = (ref_mask_compose > 128).astype(np.uint8)

        # stitch the hf map into the target image
        collage[y1: y2, x1: x2, :] = ref_image_collage
        collage_mask[y1: y2, x1: x2, :] = 1.0

    H1, W1 = collage.shape[0], collage.shape[1]
    cropped_target_image = pad_to_square(cropped_target_image, pad_value = 0, random = False).astype(np.uint8)
    collage = pad_to_square(collage, pad_value = 0, random = False).astype(np.uint8)
    collage_mask = pad_to_square(collage_mask, pad_value = 2, random = False).astype(np.uint8)
    
    H2, W2 = collage.shape[0], collage.shape[1]
    cropped_target_image = cv2.resize(cropped_target_image.astype(np.uint8), (512, 512)).astype(np.float32)
    collage = cv2.resize(collage.astype(np.uint8), (512, 512)).astype(np.float32)
    collage_mask = cv2.resize(collage_mask.astype(np.uint8), (512, 512), interpolation = cv2.INTER_NEAREST).astype(np.float32)
    
    # Prepairing dataloader items
    masked_ref_image_aug = masked_ref_image_aug / 255 
    cropped_target_image = cropped_target_image / 127.5 - 1.0
    collage = collage / 127.5 - 1.0 
    collage = np.concatenate([collage, collage_mask[:, :, :1]] , -1)

    item = dict(ref=masked_ref_image_aug.copy(), 
                jpg=cropped_target_image.copy(), 
                hint=collage.copy(), 
                extra_sizes=np.array([H1, W1, H2, W2]), 
                tar_box_yyxx_crop=np.array(tar_box_yyxx_crop) ) 
    return item


def process_multi_ref_images(ref_image, ref_mask, tar_image, tar_mask):
    '''
    inputs:
        ref_image: [(H1, W1, 3), (H2, W2, 3)]
        ref_mask: [(H1, W1), (H2, W2)]
        tar_image: (H, W, 3)
        tar_mask: [(H, W), (H, W)]
    
    outputs:
        ref: (2, 224, 224, 3)
        
    '''
    # ========= Reference ===========
    # Get the outline Box of the reference image
    multi_subject_ref_image = []
    multi_subject_ref_mask = []
    
    for single_image, single_mask in zip(ref_image, ref_mask):
        ref_box_yyxx = get_bbox_from_mask(single_mask)
    
        # Filtering background for the reference image
        ref_mask_3 = np.stack([single_mask, single_mask, single_mask], -1)
        masked_ref_image = single_image * ref_mask_3 + np.ones_like(single_image) * 255 * (1 - ref_mask_3)

        y1, y2, x1, x2 = ref_box_yyxx
        masked_ref_image = masked_ref_image[y1: y2, x1: x2, :]
        single_mask = single_mask[y1: y2, x1: x2]

        ratio = np.random.randint(11, 15) / 10 
        masked_ref_image, single_mask = expand_image_mask(masked_ref_image, single_mask, ratio=ratio)
        ref_mask_3 = np.stack([single_mask, single_mask, single_mask], -1)

        # Padding reference image to square and resize to 224
        masked_ref_image = pad_to_square(masked_ref_image, pad_value = 255, random = False)
        masked_ref_image = cv2.resize(masked_ref_image.astype(np.uint8), (224, 224)).astype(np.uint8)

        ref_mask_3 = pad_to_square(ref_mask_3 * 255, pad_value = 0, random = False)
        ref_mask_3 = cv2.resize(ref_mask_3.astype(np.uint8), (224, 224)).astype(np.uint8)
        single_mask = ref_mask_3[:, :, 0]

        # Augmenting reference image
        # masked_ref_image_aug = self.aug_data(masked_ref_image) 
        
        # Getting for high-freqency map
        masked_ref_image_compose, ref_mask_compose = aug_data_mask(masked_ref_image, single_mask) 
        masked_ref_image_aug = masked_ref_image_compose.copy()
        multi_subject_ref_image.append(masked_ref_image_aug)
        multi_subject_ref_mask.append(ref_mask_compose)

    # ref_mask_compose = np.concatenate(multi_subject_ref_mask, axis=1)
    ref_mask_3 = np.stack([ref_mask_compose, ref_mask_compose, ref_mask_compose], -1)
    masked_ref_image_compose = np.stack(multi_subject_ref_image, axis=0)
    masked_ref_image_aug = masked_ref_image_compose.copy()  # (2, 224, 244, 3)
    # ref_image_collage = sobel(masked_ref_image_compose, ref_mask_compose / 255) # (224, 448, 3)
    multi_ref_image_collage = [sobel(masked_ref_image_compose, ref_mask_compose / 255) for 
                                masked_ref_image_compose in multi_subject_ref_image]

    

    # ========= Training Target ===========
    multi_subject_bbox = []
    multi_subject_bbox_crop = []
    
    for single_mask in tar_mask:
        tar_box_yyxx = get_bbox_from_mask(single_mask)
        tar_box_yyxx = expand_bbox(single_mask, tar_box_yyxx, ratio=[1.1, 1.2]) # 1.1  1.3
        multi_subject_bbox.append(tar_box_yyxx)
        
        # Cropping around the target object 
        tar_box_yyxx_crop = expand_bbox(tar_image, tar_box_yyxx, ratio=[1.3, 3.0])   
        tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop) # crop box
        multi_subject_bbox_crop.append(tar_box_yyxx_crop)
    
    y1, x1 = min([[bbox[0], bbox[2]] for bbox in multi_subject_bbox])
    y2, x2 = max([[bbox[1], bbox[3]] for bbox in multi_subject_bbox])
    tar_box_yyxx = (y1, y2, x1, x2)
    
    # bbox which contains multi-subjects
    y1 = min([bbox[0] for bbox in multi_subject_bbox_crop])
    x1 = min([bbox[2] for bbox in multi_subject_bbox_crop])
    y2 = max([bbox[1] for bbox in multi_subject_bbox_crop])
    x2 = max([bbox[3] for bbox in multi_subject_bbox_crop])
    tar_box_yyxx_crop = (y1, y2, x1, x2)
    cropped_target_image = tar_image[y1: y2, x1: x2, :]
    collage = cropped_target_image.copy()
    collage_mask = cropped_target_image.copy() * 0.0
    tar_mask = np.max(tar_mask, axis=0)
    cropped_tar_mask = tar_mask[y1: y2, x1: x2]
    
    for single_bbox, ref_image_collage in zip(multi_subject_bbox, multi_ref_image_collage):
        tar_box_yyxx = box_in_box(single_bbox, tar_box_yyxx_crop)
        y1, y2, x1, x2 = tar_box_yyxx

        # Prepairing collage image
        ref_image_collage = cv2.resize(ref_image_collage.astype(np.uint8), (x2-x1, y2-y1))
        # ref_mask_compose = cv2.resize(ref_mask_compose.astype(np.uint8), (x2-x1, y2-y1))
        # ref_mask_compose = (ref_mask_compose > 128).astype(np.uint8)

        # stitch the hf map into the target image
        collage[y1: y2, x1: x2, :] = ref_image_collage
        collage_mask[y1: y2, x1: x2, :] = 1.0

    H1, W1 = collage.shape[0], collage.shape[1]
    cropped_target_image = pad_to_square(cropped_target_image, pad_value = 0, random = False).astype(np.uint8)
    collage = pad_to_square(collage, pad_value = 0, random = False).astype(np.uint8)
    collage_mask = pad_to_square(collage_mask, pad_value = 2, random = False).astype(np.uint8)
    
    H2, W2 = collage.shape[0], collage.shape[1]
    cropped_target_image = cv2.resize(cropped_target_image.astype(np.uint8), (512, 512)).astype(np.float32)
    collage = cv2.resize(collage.astype(np.uint8), (512, 512)).astype(np.float32)
    collage_mask = cv2.resize(collage_mask.astype(np.uint8), (512, 512), interpolation = cv2.INTER_NEAREST).astype(np.float32)
    
    # Prepairing dataloader items
    masked_ref_image_aug = masked_ref_image_aug / 255 
    cropped_target_image = cropped_target_image / 127.5 - 1.0
    collage = collage / 127.5 - 1.0 
    collage = np.concatenate([collage, collage_mask[:, :, :1]] , -1)

    item = dict(ref=masked_ref_image_aug.copy(), 
                jpg=cropped_target_image.copy(), 
                hint=collage.copy(), 
                extra_sizes=np.array([H1, W1, H2, W2]), 
                tar_box_yyxx_crop=np.array(tar_box_yyxx_crop) ) 
    return item


def crop_back(pred, tar_image, extra_sizes, tar_box_yyxx_crop):
    H1, W1, H2, W2 = extra_sizes
    y1, y2, x1, x2 = tar_box_yyxx_crop    
    pred = cv2.resize(pred, (W2, H2))
    m = 5 # maigin_pixel

    if W1 == H1:
        tar_image[y1+m: y2-m, x1+m: x2-m, :] = pred[m:-m, m:-m]
        return tar_image

    if W1 < W2:
        pad1 = int((W2 - W1) / 2)
        pad2 = W2 - W1 - pad1
        pred = pred[:,pad1: -pad2, :]
    else:
        pad1 = int((H2 - H1) / 2)
        pad2 = H2 - H1 - pad1
        pred = pred[pad1: -pad2, :, :]

    gen_image = tar_image.copy()
    gen_image[y1+m: y2-m, x1+m: x2-m, :] =  pred[m:-m, m:-m]
    return gen_image

def encode_inpainting(model: MultiDoor, control):
    inpaint = control[:, :3, :, :]
    mask = control[:, -1, :, :].unsqueeze(1)
    inpaint = model.encode_first_stage(inpaint)  # (b, 4, 64, 64)
    inpaint = model.get_first_stage_encoding(inpaint).detach()
    b, _, h, w = inpaint.shape
    mask = F.interpolate(mask, size=(h, w), mode="nearest")
    return torch.cat([inpaint, mask], dim=1)


def inference(ref_image, ref_mask, tar_image, tar_mask, ext, need_process, guidance_scale=5.0):
    '''
    inputs:
        ref_image.shape: (H, W, 3) or [(H1, W1, 3), (H2, W2, 3)]
        ref_mask.shape: [(H, W), (H, W)] or [(H1, W1), (H2, W2)]
        tar_image.shape: (H, W, 3)
        tar_mask.shape: [(H, W), (H, W)]
    '''
    if need_process:
        if isinstance(ref_image, list):
            item = process_multi_ref_images(ref_image, ref_mask, tar_image, tar_mask)
        else:
            item = process_single_ref_image(ref_image, ref_mask, tar_image, tar_mask)
        ref = item['ref']
        tar = item['jpg'] 
        hint = item['hint']
    else:
        multi_ref_image_collage = [sobel(masked_ref_image_compose, ref_mask_compose / 255) for 
                            masked_ref_image_compose, ref_mask_compose in zip(ref_image, ref_mask)]
        multi_bbox = []
        for single_mask in tar_mask:
            tar_box_yyxx = get_bbox_from_mask(single_mask)
            multi_bbox.append(tar_box_yyxx)
        collage = tar_image.copy()
        collage_mask = tar_image.copy() * 0.0
        tar_mask = np.max(tar_mask, axis=0)
        for single_bbox, ref_image_collage in zip(multi_bbox, multi_ref_image_collage):
            y1, y2, x1, x2 = single_bbox
            # Prepairing collage image
            ref_image_collage = cv2.resize(ref_image_collage.astype(np.uint8), (x2-x1, y2-y1))
            # stitch the hf map into the target image
            collage[y1: y2, x1: x2, :] = ref_image_collage
            collage_mask[y1: y2, x1: x2, :] = 1.0
        
        ref = np.stack(ref_image, axis=0) / 255 
        hint = np.concatenate([collage / 127.5 - 1.0, collage_mask[:, :, :1]] , -1)
        item = None

    input_ids: list = ext["caption"]
    image_token_masks = ext["image_token_masks"].cuda()
    
    num_samples = 1
    control = torch.from_numpy(hint.copy()).float().cuda() 
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()
    control = encode_inpainting(model, control) # (b, 5, 64, 64)

    # dino_input.shape: (n, 224, 224, 3)
    dino_input = torch.from_numpy(ref).float().cuda() 
    dino_input = torch.stack([dino_input for _ in range(num_samples)], dim=0)
    dino_input = dino_input.clone()
    image_token = model.image_encoder(dino_input)   # (b, n, 1, 1536)
    clip_input = input_ids.unsqueeze(0)
    text_token = model.get_learned_conditioning(clip_input.cuda()).last_hidden_state # (b, 77, 1024)
    context = model.fuser(
        text_token,
        image_token,
        image_token_masks,
    )

    guess_mode = False
    H, W = 512, 512
    
    uncond = model.get_unconditional_conditioning(num_samples) # (b, 77, 1024)
    
    cond = {"c_concat": control, "c_crossattn": context}
    if image_guidance:
        un_cond = {"c_concat": None if guess_mode else control, "c_crossattn": context}
    else:
        un_cond = {"c_concat": None if guess_mode else control, "c_crossattn": uncond}
    shape = (4, H // 8, W // 8)

    if save_memory:
        model.low_vram_shift(is_diffusing=True)

    # ====
    num_samples = 1
    image_resolution = 512
    strength = 1
    guess_mode = False
    ddim_steps = 50
    scale = guidance_scale
    seed = -1 
    eta = 0.0

    model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
    samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                 shape, cond, verbose=False, eta=eta,
                                                 unconditional_guidance_scale=scale,
                                                 unconditional_conditioning=un_cond)
    if save_memory:
        model.low_vram_shift(is_diffusing=False)

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy()

    result = x_samples[0][:, :, ::-1]
    result = np.clip(result, 0, 255)

    pred = x_samples[0]
    pred = np.clip(pred, 0, 255)
    if item is not None:
        sizes = item['extra_sizes']
        tar_box_yyxx_crop = item['tar_box_yyxx_crop'] 
        gen_image = crop_back(pred, tar_image, sizes, tar_box_yyxx_crop)
    else:
        gen_image = pred
    return gen_image, hint


if __name__ == '__main__': 
    # ==== Example for inferring a single image ===
    # reference_image_path = ['examples/dataset/dog/04.jpg', 'examples/dataset/dog2/04.jpg']
    # reference_mask_path = [ref_image_path.replace('jpg', 'png') for ref_image_path in reference_image_path]
    # bg_image_path = 'examples/background/11/00.png'
    # bg_mask_path = [bg_image_path.replace("00.png", "mask_1.png"), bg_image_path.replace("00.png", "mask_0.png")]
    reference_image_path = ['examples/cocoval/ref/39.jpg', 'examples/cocoval/ref/1125.jpg']
    reference_mask_path = [image_path.replace(".jpg", ".png") for 
        image_path in reference_image_path]
    bg_id = 142
    bg_image_path = f'examples/cocoval/bg/{bg_id}/bg.jpg'
    bg_mask_path = [f'examples/cocoval/bg/{bg_id}/0.png', f'examples/cocoval/bg/{bg_id}/1.png']
    caption = "The person is sitting on the yellow sofa"
    nouns = ["person", "sofa"]
    start = 0
    need_process = False
    image_guidance = False
    
    pretrained_model_name_or_path = "/data00/sqy/checkpoints/stable-diffusion-2-1-base"
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="tokenizer",
        revision=None,
    )
    image_token = "<|image|>"
    tokenizer.add_tokens([image_token], special_tokens=True)
    image_token_id = tokenizer.convert_tokens_to_ids(image_token)
    
    nouns = process_nouns_format(nouns, caption)
    ext = process_nouns_in_caption(
        nouns=nouns,
        caption=caption,
        tokenizer=tokenizer,
        image_token=image_token,
        image_token_id=image_token_id
    )
    
    while True:
        while True:
            save_path = os.path.join(os.path.dirname(bg_image_path), "GEN", f"{start}.png")
            if os.path.exists(save_path):
                start += 1
            else:
                break
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # reference image + reference mask
        if isinstance(reference_image_path, list):
            image = [cv2.cvtColor(cv2.imread(ref_image_path), cv2.COLOR_BGR2RGB) for ref_image_path in reference_image_path]
        else:
            image = cv2.cvtColor(cv2.imread(reference_image_path), cv2.COLOR_BGR2RGB)
        mask = [np.array(Image.open(file).convert('L')) == 255 for file in reference_mask_path]
        ref_image = image 
        ref_mask = mask

        # background image
        back_image = cv2.imread(bg_image_path).astype(np.uint8)
        back_image = cv2.cvtColor(back_image, cv2.COLOR_BGR2RGB)

        # background mask 
        tar_mask = [np.array(Image.open(file).convert('L')) == 255 for file in bg_mask_path]
        tar_mask = np.stack(tar_mask, axis=0).astype(np.uint8)
        
        gen_image, hint = inference(ref_image, ref_mask, back_image, tar_mask, ext, need_process=need_process, guidance_scale=7.5)
        
        h, w = back_image.shape[0], back_image.shape[1]
        hint = cv2.resize(hint, (w, h))
        hint = hint[:, :, :-1] * 127.5 + 127.5
        hint = hint.astype(np.uint8)
        if isinstance(ref_image, list):
            ref_image = [cv2.resize(ref, (w, h)).astype(np.uint8) for ref in ref_image]
            vis_image = cv2.hconcat([ref_image[0], ref_image[1], back_image, hint, gen_image.astype(np.uint8)])
        else:
            ref_image = cv2.resize(ref_image, (w, h))
            tar_mask = [cv2.resize(tar_m, (w, h)) for tar_m in tar_mask]
            vis_image = cv2.hconcat([ref_image, back_image, hint, gen_image])
        
        cv2.imwrite(save_path, vis_image [:, :, ::-1])
        print('finish!')