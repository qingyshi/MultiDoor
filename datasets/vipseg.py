import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import cv2
from .data_utils import * 
from panopticapi.utils import rgb2id
from PIL import Image
from .base import BaseDataset

class VIPSegDataset(BaseDataset):
    def __init__(self, image_dir, caption):
        super().__init__()
        self.image_root = image_dir
        video_dirs = []
        video_dirs = os.listdir(self.image_root)
        self.data = video_dirs
        self.size = (512,512)
        self.clip_size = (224,224)
        self.caption = json.load(open(caption, "r"))
        self.dynamic = 1   
    
    def __len__(self):
        return 10000

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

    def get_sample(self, idx):
        video_name = self.data[idx]  
        caption, chosen_objs, start_end_frames, nouns = self.load_caption(video_name)
        batch = self.process_nouns_in_caption(nouns, caption)

        video_path = os.path.join(self.image_root, video_name)
        frames = os.listdir(video_path)
        
        # Get image path
        start_end_frames = np.random.randint(min(start_end_frames), max(start_end_frames), 2).tolist()
        start_frame_index, end_frame_index = start_end_frames
        ref_image_name = frames[start_frame_index]
        tar_image_name = frames[end_frame_index]
        ref_image_path = os.path.join(self.image_root, video_name, ref_image_name)
        tar_image_path = os.path.join(self.image_root, video_name, tar_image_name)

        ref_mask_path = ref_image_path.replace('images','panomasksRGB').replace('.jpg', '.png')
        tar_mask_path = tar_image_path.replace('images','panomasksRGB').replace('.jpg', '.png')

        # Read Image and Mask
        ref_image = cv2.imread(ref_image_path)
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

        tar_image = cv2.imread(tar_image_path)
        tar_image = cv2.cvtColor(tar_image, cv2.COLOR_BGR2RGB)

        ref_mask = np.array(Image.open(ref_mask_path).convert('RGB'))
        ref_mask = rgb2id(ref_mask)

        tar_mask = np.array(Image.open(tar_mask_path).convert('RGB'))
        tar_mask = rgb2id(tar_mask)
        
        ref_mask = [ref_mask == single_id for single_id in chosen_objs]
        tar_mask = [tar_mask == single_id for single_id in chosen_objs]
        item_with_collage = self.process_pairs(ref_image, ref_mask, tar_image, tar_mask)
        sampled_time_steps = self.sample_timestep()
        
        item_with_collage['time_steps'] = sampled_time_steps
        item_with_collage.update(batch)
        return item_with_collage

    def check_connect(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt_area = [cv2.contourArea(cnt) for cnt in contours]
        return cnt_area

    def load_caption(self, idx):
        if idx not in self.caption:
            raise Exception
        else:
            anno = self.caption[idx]
            caption = anno['caption']
            chosen_objs = anno['chosen_objs']
            start_end_frames = anno['start_end_frames']
            nouns = anno['nouns']
            reverse_mask = anno['reverse_mask']
            if reverse_mask:
                chosen_objs = list(reversed(chosen_objs))
        return caption, chosen_objs, start_end_frames, nouns