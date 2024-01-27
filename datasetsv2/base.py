import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import cv2
from .data_utils import * 
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import albumentations as A


class BaseDataset(Dataset):
    def __init__(self):
        image_mask_dict = {}
        self.data = []

    def __len__(self):
        # We adjust the ratio of different dataset by setting the length.
        pass

    
    def aug_data_back(self, image):
        transform = A.Compose([
            A.ColorJitter(p=0.5, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            A.ChannelShuffle()
            ])
        transformed = transform(image=image.astype(np.uint8))
        transformed_image = transformed["image"]
        return transformed_image
    
    def aug_data_mask(self, image, mask):
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            #A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT,  value=(0,0,0)),
            ])

        transformed = transform(image=image.astype(np.uint8), mask=mask)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        return transformed_image, transformed_mask


    def check_region_size(self, image, yyxx, ratio, mode = 'max'):
        pass_flag = True
        H,W = image.shape[0], image.shape[1]
        H,W = H * ratio, W * ratio
        y1,y2,x1,x2 = yyxx
        h,w = y2-y1,x2-x1
        if mode == 'max':
            if h > H or w > W:
                pass_flag = False
        elif mode == 'min':
            if h < H or w < W:
                pass_flag = False
        return pass_flag


    def __getitem__(self, idx):
        while(True):
            try:
                idx = np.random.randint(0, len(self.data)-1)
                item = self.get_sample(idx)
                return item
            except:
                idx = np.random.randint(0, len(self.data)-1)
                
    def get_sample(self, idx):
        # Implemented for each specific dataset
        pass

    def sample_timestep(self, max_step =1000):
        if np.random.rand() < 0.3:
            step = np.random.randint(0,max_step)
            return np.array([step])

        if self.dynamic == 1:
            # coarse videos
            step_start = max_step // 2
            step_end = max_step
        elif self.dynamic == 0:
            # static images
            step_start = 0 
            step_end = max_step // 2
        else:
            # fine multi-view images/videos/3Ds
            step_start = 0
            step_end = max_step
        step = np.random.randint(step_start, step_end)
        return np.array([step])

    def check_mask_area(self, mask):
        H,W = mask.shape[0], mask.shape[1]
        ratio = mask.sum() / (H * W)
        if ratio > 0.8 * 0.8  or ratio < 0.1 * 0.1:
            return False
        else:
            return True 
    

    def process_pairs(self, ref_image, ref_mask, tar_image, tar_mask, max_ratio = 0.8):
        assert mask_score(np.max(ref_mask, axis=0)) > 0.90
        assert self.check_mask_area(np.max(ref_mask, axis=0)) == True
        assert self.check_mask_area(np.max(tar_mask, axis=0)) == True

        '''
        inputs:
            ref_image: (H, W, 3)
            ref_mask: (S, H, W)
            tar_image: (H, W, 3)
            tar_mask: (S, H, W)
            
        outputs:
            masked_ref_image_aug: (H, S * W, 3)
            cropped_target_image: (512, 512, 3)
            collage: (512, 512, 3)
        '''
        # ========= Reference ===========
        '''
        # similate the case that the mask for reference object is coarse. Seems useless :(

        if np.random.uniform(0, 1) < 0.7: 
            ref_mask_clean = ref_mask.copy()
            ref_mask_clean = np.stack([ref_mask_clean,ref_mask_clean,ref_mask_clean],-1)
            ref_mask = perturb_mask(ref_mask, 0.6, 0.9)
            
            # select a fake bg to avoid the background leakage
            fake_target = tar_image.copy()
            h,w = ref_image.shape[0], ref_image.shape[1]
            fake_targe = cv2.resize(fake_target, (w,h))
            fake_back = np.fliplr(np.flipud(fake_target))
            fake_back = self.aug_data_back(fake_back)
            ref_image = ref_mask_clean * ref_image + (1-ref_mask_clean) * fake_back
        '''

        # Get the outline Box of the reference image
        multi_subject_ref_image = []
        multi_subject_ref_mask = []
        
        for single_mask in ref_mask:
            ref_box_yyxx = get_bbox_from_mask(single_mask)
            assert self.check_region_size(single_mask, ref_box_yyxx, ratio = 0.10, mode = 'min') == True
        
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
            masked_ref_image_compose, ref_mask_compose =  self.aug_data_mask(masked_ref_image, single_mask) 
            masked_ref_image_aug = masked_ref_image_compose.copy()
            multi_subject_ref_image.append(masked_ref_image_aug)
            multi_subject_ref_mask.append(ref_mask_compose)

        ref_mask_compose = np.concatenate(multi_subject_ref_mask, axis=1)
        ref_mask_3 = np.stack([ref_mask_compose, ref_mask_compose, ref_mask_compose], -1)
        masked_ref_image_compose = np.concatenate(multi_subject_ref_image, axis=1)
        masked_ref_image_aug = masked_ref_image_compose.copy()
        ref_image_collage = sobel(masked_ref_image_compose, ref_mask_compose / 255) # (224, 448, 3)

        

        # ========= Training Target ===========
        multi_subject_bbox = []
        multi_subject_bbox_crop = []
        
        for single_mask in tar_mask:
            tar_box_yyxx = get_bbox_from_mask(single_mask)
            tar_box_yyxx = expand_bbox(single_mask, tar_box_yyxx, ratio=[1.1, 1.2]) #1.1  1.3
            multi_subject_bbox.append(tar_box_yyxx)
            assert self.check_region_size(single_mask, tar_box_yyxx, ratio = max_ratio, mode = 'max') == True
            
            # Cropping around the target object 
            tar_box_yyxx_crop = expand_bbox(tar_image, tar_box_yyxx, ratio=[1.3, 3.0])   
            tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop) # crop box
            y1, y2, x1, x2 = tar_box_yyxx_crop
            multi_subject_bbox_crop.append(tar_box_yyxx_crop)
        
        y1, x1 = min([[bbox[0], bbox[2]] for bbox in multi_subject_bbox_crop])
        y2, x2 = max([[bbox[1], bbox[3]] for bbox in multi_subject_bbox_crop])
        tar_box_yyxx_crop = (y1, y2, x1, x2)
        cropped_target_image = tar_image[y1: y2, x1: x2, :]
        tar_mask = np.max(tar_mask, axis=0)
        cropped_tar_mask = tar_mask[y1: y2, x1: x2]
        
        y1, x1 = min([[bbox[0], bbox[2]] for bbox in multi_subject_bbox])
        y2, x2 = max([[bbox[1], bbox[3]] for bbox in multi_subject_bbox])
        tar_box_yyxx = (y1, y2, x1, x2)
        tar_box_yyxx = box_in_box(tar_box_yyxx, tar_box_yyxx_crop)
        y1, y2, x1, x2 = tar_box_yyxx

        # Prepairing collage image
        ref_image_collage = cv2.resize(ref_image_collage.astype(np.uint8), (x2-x1, y2-y1))
        ref_mask_compose = cv2.resize(ref_mask_compose.astype(np.uint8), (x2-x1, y2-y1))
        ref_mask_compose = (ref_mask_compose > 128).astype(np.uint8)

        collage = cropped_target_image.copy() 
        collage[y1: y2, x1: x2, :] = ref_image_collage

        collage_mask = cropped_target_image.copy() * 0.0
        collage_mask[y1: y2, x1: x2, :] = 1.0

        if np.random.uniform(0, 1) < 0.7: 
            cropped_tar_mask = perturb_mask(cropped_tar_mask)
            collage_mask = np.stack([cropped_tar_mask, cropped_tar_mask, cropped_tar_mask],-1)

        H1, W1 = collage.shape[0], collage.shape[1]

        cropped_target_image = pad_to_square(cropped_target_image, pad_value = 0, random = False).astype(np.uint8)
        collage = pad_to_square(collage, pad_value = 0, random = False).astype(np.uint8)
        collage_mask = pad_to_square(collage_mask, pad_value = 2, random = False).astype(np.uint8)
        H2, W2 = collage.shape[0], collage.shape[1]

        cropped_target_image = cv2.resize(cropped_target_image.astype(np.uint8), (512, 512)).astype(np.float32)
        collage = cv2.resize(collage.astype(np.uint8), (512, 512)).astype(np.float32)
        collage_mask = cv2.resize(collage_mask.astype(np.uint8), (512, 512), interpolation = cv2.INTER_NEAREST).astype(np.float32)
        collage_mask[collage_mask == 2] = -1
        
        # Prepairing dataloader items
        masked_ref_image_aug = masked_ref_image_aug  / 255 
        cropped_target_image = cropped_target_image / 127.5 - 1.0
        collage = collage / 127.5 - 1.0 
        collage = np.concatenate([collage, collage_mask[:, :, :1]] , -1)
        
        item = dict(
                ref=masked_ref_image_aug.copy(), 
                jpg=cropped_target_image.copy(), 
                hint=collage.copy(), 
                extra_sizes=np.array([H1, W1, H2, W2]), 
                tar_box_yyxx_crop=np.array(tar_box_yyxx_crop) 
                ) 
        return item





