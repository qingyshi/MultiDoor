import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from transformers import CLIPTokenizer
from PIL import Image
import cv2
from .data_utils import * 
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import albumentations as A


class BaseDataset(Dataset):
    def __init__(
        self,
        max_num_objects=2,
        image_token="<|image|>", 
        pretrained_model_name_or_path="/data00/sqy/checkpoints/stable-diffusion-2-1-base"
    ):
        self.max_num_objects = max_num_objects
        self.image_token = image_token
        self.tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=None,
        )
        self.tokenizer.add_tokens([image_token], special_tokens=True)
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(image_token)
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
            A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT, value=(0,0,0)),
            ])

        transformed = transform(image=image.astype(np.uint8), mask=mask)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        return transformed_image, transformed_mask


    def check_region_size(self, image, yyxx, ratio, mode='max'):
        pass_flag = True
        H, W = image.shape[0], image.shape[1]
        H, W = H * ratio, W * ratio
        y1, y2, x1, x2 = yyxx
        h, w = y2 - y1, x2 - x1
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
    
    def load_caption(self, id):
        if id not in self.caption:
            raise Exception
        else:
            anno = self.caption[id]
            caption = anno['caption']
            chosen_objs = anno.get('chosen_objs', None)
            nouns: list = anno.get('nouns', None)
            predicate: str = anno.get('predicate', None)
        return caption, chosen_objs, nouns, predicate
    
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
    

    def process_pairs(self, ref_image, ref_mask, tar_image, tar_mask, max_ratio=0.9):
        assert max(mask_score(ref_mask[0]), mask_score(ref_mask[1])) > 0.90
        assert self.check_mask_area(ref_mask[0]) == True
        assert self.check_mask_area(tar_mask[0]) == True
        assert self.check_mask_area(ref_mask[1]) == True
        assert self.check_mask_area(tar_mask[1]) == True
        '''
        inputs:
            ref_image: (H, W, 3)
            ref_mask: [(H, W), (H, W)]
            tar_image: (H, W, 3)
            tar_mask: [(H, W), (H, W)]
            
        outputs:
            masked_ref_image_aug: (2, 224, 224, 3)
            cropped_target_image: (512, 512, 3)
            collage: (512, 512, 4)
            target_masks: (2, 512, 512)
        '''   
        # Get the outline Box of the reference image
        multi_subject_ref_image = []
        multi_subject_ref_mask = []
        
        for single_mask in ref_mask:
            ref_box_yyxx = get_bbox_from_mask(single_mask)
            assert self.check_region_size(single_mask, ref_box_yyxx, ratio=0.10, mode='min') == True
        
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
            masked_ref_image_compose, ref_mask_compose = self.aug_data_mask(masked_ref_image, single_mask) 
            masked_ref_image_aug = masked_ref_image_compose.copy()
            multi_subject_ref_image.append(masked_ref_image_aug)
            multi_subject_ref_mask.append(ref_mask_compose)

        masked_ref_image_compose = np.stack(multi_subject_ref_image, axis=0)
        masked_ref_image_aug = masked_ref_image_compose.copy() # as ref image, shape: (2, 224, 244, 3)
        # Getting for high-freqency map
        multi_ref_image_collage = [sobel(masked_ref_image_compose, ref_mask_compose / 255) 
                                        for masked_ref_image_compose, ref_mask_compose in 
                                        zip(multi_subject_ref_image, multi_subject_ref_mask)]



        # ========= Training Target ===========
        multi_subject_bbox = []
        multi_subject_bbox_crop = []
        # tar_masks = []
        for single_mask in tar_mask:
            tar_box_yyxx = get_bbox_from_mask(single_mask)
            tar_box_yyxx = expand_bbox(single_mask, tar_box_yyxx, ratio=[1.1, 1.2]) # 1.1, 1.3
            multi_subject_bbox.append(tar_box_yyxx)
            assert self.check_region_size(single_mask, tar_box_yyxx, ratio=max_ratio, mode='max') == True
            
            # Cropping around the target object 
            tar_box_yyxx_crop = expand_bbox(tar_image, tar_box_yyxx, ratio=[1.3, 3.0])   
            tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop) # crop box
            multi_subject_bbox_crop.append(tar_box_yyxx_crop)
        
        # Bbox which contains multi-subjects
        y1 = min([bbox[0] for bbox in multi_subject_bbox_crop])
        x1 = min([bbox[2] for bbox in multi_subject_bbox_crop])
        y2 = max([bbox[1] for bbox in multi_subject_bbox_crop])
        x2 = max([bbox[3] for bbox in multi_subject_bbox_crop])
        tar_box_yyxx_crop = (y1, y2, x1, x2)
        cropped_target_image = tar_image[y1: y2, x1: x2, :]
        tar_masks = [mask[y1: y2, x1: x2] for mask in tar_mask]
        collage = cropped_target_image.copy()
        collage_mask = cropped_target_image.copy() * 0.0
        tar_mask = np.max(tar_mask, axis=0)
        cropped_tar_mask = tar_mask[y1: y2, x1: x2]
        
        for single_bbox in multi_subject_bbox:
            tar_box_yyxx = box_in_box(single_bbox, tar_box_yyxx_crop)
            y1, y2, x1, x2 = tar_box_yyxx
            collage[y1: y2, x1: x2, :] = 0
            
        for single_bbox, ref_image_collage in zip(multi_subject_bbox, multi_ref_image_collage):
            single_mask = cropped_target_image.copy() * 0.0
            tar_box_yyxx = box_in_box(single_bbox, tar_box_yyxx_crop)
            y1, y2, x1, x2 = tar_box_yyxx
            
            # Prepairing collage image
            ref_image_collage = cv2.resize(ref_image_collage.astype(np.uint8), (x2-x1, y2-y1))

            # stitch the hf map into the target image
            collage[y1: y2, x1: x2, :] += ref_image_collage
            collage_mask[y1: y2, x1: x2, :] = 1.0
            single_mask[y1: y2, x1: x2, :] = 1.0

        if np.random.uniform(0, 1) < 0.7: 
            cropped_tar_mask = perturb_mask(cropped_tar_mask)
            collage_mask = np.stack([cropped_tar_mask, cropped_tar_mask, cropped_tar_mask], -1)

        H1, W1 = collage.shape[0], collage.shape[1]

        cropped_target_image = pad_to_square(cropped_target_image, pad_value = 0, random = False).astype(np.uint8)
        collage = pad_to_square(collage, pad_value = 0, random = False).astype(np.uint8)
        collage_mask = pad_to_square(collage_mask, pad_value = 2, random = False).astype(np.uint8)
        tar_masks = [pad_to_square(single_mask[:, :, None], pad_value = 0, random = False).astype(np.uint8)[:, :, 0] 
                        for single_mask in tar_masks]
        H2, W2 = collage.shape[0], collage.shape[1]

        cropped_target_image = cv2.resize(cropped_target_image.astype(np.uint8), (512, 512)).astype(np.float32)
        collage = cv2.resize(collage.astype(np.uint8), (512, 512)).astype(np.float32)
        collage_mask = cv2.resize(collage_mask.astype(np.uint8), (512, 512), interpolation = cv2.INTER_NEAREST).astype(np.float32)
        tar_masks = [cv2.resize(single_mask.astype(np.uint8), (512, 512), interpolation = cv2.INTER_NEAREST).astype(np.float32) 
                        for single_mask in tar_masks]
        tar_masks = np.stack(tar_masks, axis=0)
        collage_mask[collage_mask == 2] = -1
        
        # Prepairing dataloader items
        masked_ref_image_aug = masked_ref_image_aug / 255 
        cropped_target_image = cropped_target_image / 127.5 - 1.0
        collage = collage / 127.5 - 1.0 
        collage = np.concatenate([collage, collage_mask[:, :, :1]], -1)
        
        item = dict(
                ref=masked_ref_image_aug.copy(), 
                jpg=cropped_target_image.copy(), 
                hint=collage.copy(), 
                extra_sizes=np.array([H1, W1, H2, W2]), 
                tar_box_yyxx_crop=np.array(tar_box_yyxx_crop),
                target_masks=tar_masks, # (n, 512, 512)
                ) 
        return item

    def process_nouns_in_caption(self, nouns, caption):
        nouns = sorted(nouns, key=lambda x: x["end"], reverse=True)
        for noun in nouns:
            end = noun["end"]
            caption = caption[:end] + self.image_token + caption[end:]

        input_ids = self.tokenizer(
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
            if idx == self.image_token_id:
                noun_phrase_end_mask[track_idx - 1] = True
            else:
                track_idx += 1
                clean_input_ids.append(idx)
                
        max_len = self.tokenizer.model_max_length
        if len(clean_input_ids) > max_len:
            clean_input_ids = clean_input_ids[:max_len]
        else:
            clean_input_ids = clean_input_ids + [self.tokenizer.pad_token_id] * (
                max_len - len(clean_input_ids)
            )
        noun_phrase_end_mask = torch.tensor(noun_phrase_end_mask, dtype=torch.bool)
        assert noun_phrase_end_mask.sum() == self.max_num_objects
        clean_input_ids = torch.tensor(clean_input_ids)
        
        image_token_ids = torch.nonzero(noun_phrase_end_mask[None], as_tuple=True)[1]
        image_token_ids_mask = torch.ones_like(image_token_ids, dtype=torch.bool)
        if len(image_token_ids) < self.max_num_objects:
            image_token_ids = torch.cat(
                [
                    image_token_ids,
                    torch.zeros(self.max_num_objects - len(image_token_ids), dtype=torch.long),
                ]
            )
            image_token_ids_mask = torch.cat(
                [
                    image_token_ids_mask,
                    torch.zeros(
                        self.max_num_objects - len(image_token_ids_mask),
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
    
    def check_names_in_nouns(self, names, nouns, caption):
        for name, noun in zip(names, nouns):
            if name == noun["word"]:
                pass
            elif name in caption:
                noun["word"] = name
                start = caption.index(name)
                noun["start"] = start
                noun["end"] = start + len(name)
            else:
                raise Exception
        return nouns