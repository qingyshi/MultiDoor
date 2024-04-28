import numpy as np
import json
import os.path as osp
import cv2
from pycocotools.ovis import OVIS
import pycocotools.mask as mask_utils
from .base import BaseDataset


class LVVISDataset(BaseDataset):
    def __init__(
        self,
        ann_file,
        img_prefix,
        caption,
    ):
        super().__init__()
        # prefix of images path
        self.img_prefix = img_prefix

        # load annotations (and proposals)
        self.vid_infos = self.load_annotations(ann_file)
        
        # load relation caption
        self.caption = json.load(open(caption, 'r'))
        self.dynamic = 1

    def load_annotations(self, ann_file):
        self.ovis = OVIS(ann_file)
        self.cat_ids = self.ovis.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.data = self.ovis.getVidIds()
        vid_infos = {}
        for i in self.data:
            info: dict = self.ovis.loadVids([i])[0]
            info['filenames'] = info['file_names']
            vid_infos[i] = info
        return vid_infos

    def __len__(self):
        return 2000
    
    def get_sample(self, idx):
        video_id = self.data[idx]
        caption, chosen_objs, start_end_frames, nouns = self.load_caption(str(video_id))
        if nouns[0]["end"] == nouns[1]["end"]:
            raise Exception
        batch = self.process_nouns_in_caption(nouns, caption)
        
        vid_info = self.vid_infos[video_id]
        frames = vid_info["filenames"]
        start_frame_idx = start_end_frames[0]
        end_frame_idx = start_end_frames[1]
        start_frame = frames[start_frame_idx]
        end_frame = frames[end_frame_idx]
        
        annos = self.ovis.vidToAnns[video_id]  # list of dict            
        annos = sorted(annos, key=lambda x: mask_utils.decode(x["segmentations"][end_frame_idx]).sum(), reverse=True)
                
        objects = [annos[obj_id] for obj_id in chosen_objs]
        names = [self.ovis.cats[obj['category_id']]['name'].lower() for obj in objects]
        objects_masks = [obj["segmentations"] for obj in objects]
        
        ref_image_path = osp.join(self.img_prefix, start_frame)
        tar_image_path = osp.join(self.img_prefix, end_frame)
        ref_image = cv2.imread(ref_image_path)
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
        tar_image = cv2.imread(tar_image_path)
        tar_image = cv2.cvtColor(tar_image, cv2.COLOR_BGR2RGB)
        
        ref_mask = [mask_utils.decode(object_masks[start_frame_idx]) for object_masks in objects_masks]
        tar_mask = [mask_utils.decode(object_masks[end_frame_idx]) for object_masks in objects_masks]
        item_with_collage = self.process_pairs(ref_image, ref_mask, tar_image, tar_mask)
        sampled_time_steps = self.sample_timestep()
        item_with_collage['time_steps'] = sampled_time_steps
        item_with_collage.update(batch)
        return item_with_collage

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