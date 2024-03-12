import cv2
import einops
import numpy as np
import torch
import random
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
from datasetsv2.coco import CocoDataset


save_memory = False
disable_verbosity()
if save_memory:
    enable_sliced_attention()

DConf = OmegaConf.load('./configs/datasetsv2.yaml')
dataset = CocoDataset(**DConf.Train.COCO)

config = OmegaConf.load('./configs/inference.yaml')
model_ckpt =  config.pretrained_model
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


def process_pairs(ref_image, ref_mask, tar_image, tar_mask):
    '''
    inputs:
        ref_image: (H, W, 3)
        ref_mask: (2, H, W)
        tar_image: (H, W, 3)
        tar_mask: (2, H, W)
    
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
    
    y1, x1 = min([[bbox[0], bbox[2]] for bbox in multi_subject_bbox])
    y2, x2 = max([[bbox[1], bbox[3]] for bbox in multi_subject_bbox])
    tar_box_yyxx = (y1, y2, x1, x2)
    
    # bbox which contains multi-subjects
    y1, x1 = min([[bbox[0], bbox[2]] for bbox in multi_subject_bbox_crop])
    y2, x2 = max([[bbox[1], bbox[3]] for bbox in multi_subject_bbox_crop])
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


def process_multi_pairs(ref_image, ref_mask, tar_image, tar_mask):
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
    y1, x1 = min([[bbox[0], bbox[2]] for bbox in multi_subject_bbox_crop])
    y2, x2 = max([[bbox[1], bbox[3]] for bbox in multi_subject_bbox_crop])
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
    h, w, _ = tar_image.shape
    y2 = min(h, y2)
    x2 = min(w, x2) 
    pred = cv2.resize(pred, (x2 - x1, y2 - y1))
    m = 5 # maigin_pixel

    # if W1 == H1:
    #     tar_image[y1+m: y2-m, x1+m: x2-m, :] = pred[m:-m, m:-m]
    #     return tar_image

    # if W1 < W2:
    #     pad1 = int((W2 - W1) / 2)
    #     pad2 = W2 - W1 - pad1
    #     pred = pred[:,pad1: -pad2, :]
    # else:
    #     pad1 = int((H2 - H1) / 2)
    #     pad2 = H2 - H1 - pad1
    #     pred = pred[pad1: -pad2, :, :]

    gen_image = tar_image.copy()
    gen_image[y1+m: y2-m, x1+m: x2-m, :] =  pred[m:-m, m:-m]
    return gen_image


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
    ref_token = model.ref_encoder(dino_input)   # (b, 30, 1024)

    guess_mode = False
    H, W = 512, 512
    
    clip_input = caption * num_samples
    cond_text = model.get_learned_conditioning(clip_input) # (b, 77, 1024)
    uncond = model.get_unconditional_conditioning(num_samples) # (b, 77, 1024)
    # caption_embedding: (b, 77, 1024)
    
    cond = {"c_concat": [control], "c_crossattn": [cond_text], "ref_token": [ref_token]}
    un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [uncond], "ref_token": [ref_token]}
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
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy()#.clip(0, 255).astype(np.uint8)

    result = x_samples[0][:, :, ::-1]
    result = np.clip(result, 0, 255)

    pred = x_samples[0]
    pred = np.clip(pred, 0, 255)[1:, :, :]
    sizes = item['extra_sizes']
    tar_box_yyxx_crop = item['tar_box_yyxx_crop'] 
    gen_image = crop_back(pred, tar, sizes, tar_box_yyxx_crop) 
    return gen_image


if __name__ == '__main__': 
    count = 0
    while True:
        save_path = f'examples/train/visual{count}.jpg'
        data = next(iter(dataset))
        back_image = data['jpg'] * 127.5 + 127.5
        ref_image = data['ref'] * 127.5 + 127.5
        collage = data['hint'] * 127.5 + 127.5
        gen_image = inference_single_image(item=data)
        h, w = back_image.shape[0], back_image.shape[1]
        if len(ref_image) == 2:
            ref_image = [cv2.resize(ref, (w, h)) for ref in ref_image]
            collage = cv2.resize(collage, (w, h))
            vis_image = cv2.hconcat([ref_image[0].astype(np.float32), ref_image[1].astype(np.float32), collage[:, :, :-1].astype(np.float32), back_image, gen_image])
        else:
            ref_image = cv2.resize(ref_image, (w, h))
            vis_image = cv2.hconcat([ref_image, back_image, gen_image])
        
        cv2.imwrite(save_path, vis_image[:, :, ::-1])
        print(data['caption'])
        print('finish!')
        count += 1
    #'''
    #'''
    # ==== Example for inferring VITON-HD Test dataset ===

    # from omegaconf import OmegaConf
    # import os 
    # DConf = OmegaConf.load('./configs/datasets.yaml')
    # save_dir = './VITONGEN'
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)

    # test_dir = DConf.Test.VitonHDTest.image_dir
    # image_names = os.listdir(test_dir)
    
    # for image_name in image_names:
    #     ref_image_path = os.path.join(test_dir, image_name)
    #     tar_image_path = ref_image_path.replace('/cloth/', '/image/')
    #     ref_mask_path = ref_image_path.replace('/cloth/','/cloth-mask/')
    #     tar_mask_path = ref_image_path.replace('/cloth/', '/image-parse-v3/').replace('.jpg','.png')

    #     ref_image = cv2.imread(ref_image_path)
    #     ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

    #     gt_image = cv2.imread(tar_image_path)
    #     gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)

    #     ref_mask = (cv2.imread(ref_mask_path) > 128).astype(np.uint8)[:, :, 0]

    #     tar_mask = Image.open(tar_mask_path ).convert('P')
    #     tar_mask = np.array(tar_mask)
    #     tar_mask = tar_mask == 5

    #     gen_image = inference_single_image(ref_image, ref_mask, gt_image.copy(), tar_mask)
    #     gen_path = os.path.join(save_dir, image_name)

    #     vis_image = cv2.hconcat([ref_image, gt_image, gen_image])
    #     cv2.imwrite(gen_path, vis_image[:, :, ::-1])
    #'''

    
    

