import torch
import json
import numpy as np
import os
from typing import List, Dict
import torchvision.transforms.functional as F
from .burstapi.dataset import BURSTDataset
from .base import BaseDataset


class BurstDataset(BaseDataset):
    def __init__(self, annotations_file, images_base_dir, caption):
        super().__init__()
        self.data = BURSTDataset(annotations_file, images_base_dir)
        self.id2name = self.data.category_names
        self.caption = json.load(open(caption, "r"))
        self.dynamic = 2

    def get_sample(self, index):
        burst_video = self.data[index]
        video_id = burst_video.id
        caption, chosen_objs, nouns, frame_indices = self.load_caption(str(video_id))
        batch = self.process_nouns_in_caption(nouns, caption)
        
        # num_frames = burst_video.num_annotated_frames
        _track_category_ids = burst_video._track_category_ids
        # frame_indices = np.random.choice(num_frames, 2, replace=False).tolist()

        images = burst_video.load_images(frame_indices)
        images_path = burst_video.get_image_paths(frame_indices)
        masks: List[Dict] = burst_video.load_masks(frame_indices)
    
        names = [self.id2name[_track_category_ids[obj]] for obj in chosen_objs]
        ref_mask = [masks[0][obj] for obj in chosen_objs]
        tar_mask = [masks[1][obj] for obj in chosen_objs]
        ref_image = images[0]
        tar_image = images[1]
        
        item_with_collage = self.process_pairs(ref_image=ref_image, ref_masks=ref_mask, 
                                               tar_image=tar_image, tar_masks=tar_mask)
        sampled_time_steps = self.sample_timestep()
        
        item_with_collage['time_steps'] = sampled_time_steps
        item_with_collage.update(batch)
        return item_with_collage

    def __len__(self):
        return 20000
    
    def load_caption(self, idx):
        if idx not in self.caption:
            raise Exception
        else:
            anno = self.caption[idx]
            caption = anno['caption']
            chosen_objs = anno['chosen_objs']
            nouns: list = anno['nouns']
            reverse_mask: bool = anno['reverse_mask']
            frame_indices = anno['frame_indices']
            if reverse_mask:
                chosen_objs = list(reversed(chosen_objs))
        return caption, chosen_objs, nouns, frame_indices