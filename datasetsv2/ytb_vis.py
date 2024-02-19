import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import cv2
from .data_utils import * 
from .base import BaseDataset

class YoutubeVISDataset(BaseDataset):
    def __init__(self, image_dir, anno, meta, caption):
        self.image_root = image_dir
        self.anno_root = anno 
        self.meta_file = meta

        video_dirs = []
        with open(self.meta_file) as f:
            records = json.load(f)
            records = records["videos"]
            for video_id in records:
                video_dirs.append(video_id)
        
        with open(caption) as f:
            caption = json.load(f)

        self.records = records
        self.caption = caption
        self.data = video_dirs
        self.size = (512,512)
        self.clip_size = (224,224)
        self.dynamic = 1

    def __len__(self):
        return 40000

    def check_region_size(self, image, yyxx, ratio, mode = 'max'):
        pass_flag = True
        H,W = image.shape[0], image.shape[1]
        H,W = H * ratio, W * ratio
        y1,y2,x1,x2 = yyxx
        h,w = y2-y1,x2-x1
        if mode == 'max':
            if h > H and w > W:
                pass_flag = False
        elif mode == 'min':
            if h < H and w < W:
                pass_flag = False
        return pass_flag

    def get_sample(self, idx):
        video_id = list(self.records.keys())[idx]
        if video_id not in self.caption:
            raise Exception
        else:
            caption_ann = self.caption[video_id]
            caption = caption_ann['caption']
            class_token_ids = caption_ann['class_token_ids']
        
        # obj_list = list(self.records[video_id]["objects"].keys())
        # if len(obj_list) <= 1:
        #     raise Exception
        # objects_id = np.random.choice(obj_list, 2, replace=False)
        objects_ids = caption_ann['obj_ids']
        frames = [self.records[video_id]["objects"][str(single_id)]["frames"] for single_id in objects_ids]
        names = [self.records[video_id]["objects"][str(single_id)]["category"] for single_id in objects_ids]
        frames = np.intersect1d(*frames)

        # Sampling frames
        min_interval = len(frames) // 10
        start_frame_index = np.random.randint(low=0, high=len(frames) - min_interval)
        end_frame_index = start_frame_index + np.random.randint(min_interval,  len(frames) - start_frame_index)
        end_frame_index = min(end_frame_index, len(frames) - 1)

        # Get image path
        ref_image_name = frames[start_frame_index]
        tar_image_name = frames[end_frame_index]
        ref_image_path = os.path.join(self.image_root, video_id, ref_image_name) + '.jpg'
        tar_image_path = os.path.join(self.image_root, video_id, tar_image_name) + '.jpg'
        ref_mask_path = ref_image_path.replace('JPEGImages','Annotations').replace('.jpg', '.png')
        tar_mask_path = tar_image_path.replace('JPEGImages','Annotations').replace('.jpg', '.png')

        # Read Image and Mask
        ref_image = cv2.imread(ref_image_path)
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

        tar_image = cv2.imread(tar_image_path)
        tar_image = cv2.cvtColor(tar_image, cv2.COLOR_BGR2RGB)

        ref_mask = Image.open(ref_mask_path).convert('P')
        ref_mask= np.array(ref_mask)
        ref_mask = np.stack([ref_mask == int(single_id) for single_id in objects_ids], 0)

        tar_mask = Image.open(tar_mask_path).convert('P')
        tar_mask= np.array(tar_mask)
        tar_mask = np.stack([tar_mask == int(single_id) for single_id in objects_ids], 0)

        item_with_collage = self.process_pairs(ref_image, ref_mask, tar_image, tar_mask)
        sampled_time_steps = self.sample_timestep()
        
        item_with_collage['time_steps'] = sampled_time_steps
        # item_with_collage['names'] = names
        # item_with_collage['obj_ids'] = objects_ids
        # item_with_collage['img_path'] = tar_image_path
        # item_with_collage['video_id'] = video_id
        item_with_collage['caption'] = caption
        item_with_collage['class_token_ids'] = torch.tensor(class_token_ids)
        return item_with_collage

