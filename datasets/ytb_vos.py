import json
import cv2
import numpy as np
import os
from PIL import Image
import cv2
from .data_utils import * 
from .base import BaseDataset

class YoutubeVOSDataset(BaseDataset):
    def __init__(self, image_dir, anno, meta, caption):
        super().__init__()
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
            self.caption = json.load(f)
        self.records = records
        self.data = video_dirs
        self.size = (512,512)
        self.clip_size = (224,224)
        self.dynamic = 1

    def __len__(self):
        return 20000

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
        caption, chosen_objs, nouns = self.load_caption(video_id)
        batch = self.process_nouns_in_caption(nouns, caption)
        
        frames = [self.records[video_id]["objects"][str(objects_id)]["frames"] for objects_id in chosen_objs]
        frames = np.intersect1d(*frames)

        # Sampling frames
        min_interval = len(frames) // 10
        start_frame_index = np.random.randint(low=0, high=len(frames) - min_interval)
        end_frame_index = start_frame_index + np.random.randint(min_interval, len(frames) - start_frame_index)
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
        ref_mask = [ref_mask == int(objects_id) for objects_id in chosen_objs]

        tar_mask = Image.open(tar_mask_path).convert('P')
        tar_mask= np.array(tar_mask)
        tar_mask = [tar_mask == int(objects_id) for objects_id in chosen_objs]

        item_with_collage = self.process_pairs(ref_image, ref_mask, tar_image, tar_mask)
        sampled_time_steps = self.sample_timestep()
        
        item_with_collage['time_steps'] = sampled_time_steps
        item_with_collage.update(batch)
        return item_with_collage


