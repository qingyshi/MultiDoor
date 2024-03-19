import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import cv2
from .data_utils import * 
from .base import BaseDataset

class MVImageNetDataset(BaseDataset):
    def __init__(self, txt, image_dir, caption):
        with open(txt,"r") as f:
            data = f.read().split('\n')[:-1]    
        self.image_dir = image_dir 
        self.data = data
        
        with open(caption, 'r') as f:
            self.caption = json.load(f)
        
        self.size = (512,512)
        self.clip_size = (224,224)
        self.dynamic = 2

    def __len__(self):
        return 10000
    
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

    def get_alpha_mask(self, mask_path):
        image = cv2.imread( mask_path, cv2.IMREAD_UNCHANGED)
        mask = (image[:,:,-1] > 128).astype(np.uint8)
        return mask
        
    def get_sample(self, idx):
        object_dir = self.data[idx].replace('./', self.image_dir)
        _object_dir = os.path.dirname(object_dir)
        caption = self.load_caption(_object_dir)
        
        frames = os.listdir(object_dir)
        frames = [i for i in frames if '.png' in i]

        # Sampling frames
        min_interval = len(frames)  // 8
        start_frame_index = np.random.randint(low=0, high=len(frames) - min_interval)
        end_frame_index = start_frame_index + np.random.randint(min_interval,  len(frames) - start_frame_index )
        end_frame_index = min(end_frame_index, len(frames) - 1)

        # Get image path
        ref_mask_name = frames[start_frame_index]
        tar_mask_name = frames[end_frame_index]

        ref_image_name = ref_mask_name.split('_')[0] + '.jpg'
        tar_image_name = tar_mask_name.split('_')[0] + '.jpg'

        ref_mask_path = os.path.join(object_dir, ref_mask_name)
        tar_mask_path = os.path.join(object_dir, tar_mask_name)
        ref_image_path = os.path.join(object_dir, ref_image_name)
        tar_image_path = os.path.join(object_dir, tar_image_name) 

        # Read Image and Mask
        ref_image = cv2.imread(ref_image_path).astype(np.uint8)
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

        tar_image = cv2.imread(tar_image_path).astype(np.uint8)
        tar_image = cv2.cvtColor(tar_image, cv2.COLOR_BGR2RGB)

        ref_mask = self.get_alpha_mask(ref_mask_path)
        tar_mask = self.get_alpha_mask(tar_mask_path)

        item_with_collage = self.process_pairs(ref_image, ref_mask, tar_image, tar_mask)
        sampled_time_steps = self.sample_timestep()
        item_with_collage['time_steps'] = sampled_time_steps
        item_with_collage['caption'] = caption
        item_with_collage['img_path'] = tar_image_path
        # item_with_collage['video_id'] = os.path.dirname(object_dir)

        return item_with_collage

