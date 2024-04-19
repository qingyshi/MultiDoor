import numpy as np
import json
import os.path as osp
import random
import cv2
from pycocotools.ovis import OVIS
import pycocotools.mask as mask_utils
from .base import BaseDataset


class OVISDataset(BaseDataset):
    
    CLASSES=['Person', 'Bird', 'Cat', 'Dog', 'Horse', 'Sheep', 'Cow', 'Elephant', 'Bear', 'Zebra', 'Giraffe',
                  'Poultry', 'Giant_panda', 'Lizard', 'Parrot', 'Monkey', 'Rabbit', 'Tiger', 'Fish', 'Turtle', 'Bicycle',
               'Motorcycle', 'Airplane', 'Boat', 'Vehical']

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
        self.dynamic = 2

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
        return 20000
    
    def get_sample(self, idx):
        video_id = self.data[idx]
        caption, chosen_objs, nouns = self.load_caption(str(video_id))
        batch = self.process_nouns_in_caption(nouns, caption)
        
        vid_info = self.vid_infos[video_id]
        annnos = self.ovis.vidToAnns[video_id]  # list of dict
        # num_objects = len(annnos)
        # if num_objects >= 2:
        #     chosen_objs = np.random.choice(num_objects, 2, replace=False).tolist()
        # else:
        #     raise Exception
        objects = [annnos[obj_id] for obj_id in chosen_objs]
        names = [self.ovis.cats[obj['category_id']]['name'].lower() for obj in objects]
        objects_masks = [obj["segmentations"] for obj in objects]
        
        frames = vid_info["filenames"]
        num_frames = len(frames)
        start_end_frames = np.random.choice(num_frames, 2, replace=False).tolist()
        start_frame = frames[start_end_frames[0]]
        end_frame = frames[start_end_frames[1]]
        ref_image_path = osp.join(self.img_prefix, start_frame)
        tar_image_path = osp.join(self.img_prefix, end_frame)
        ref_image = cv2.imread(ref_image_path)
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
        tar_image = cv2.imread(tar_image_path)
        tar_image = cv2.cvtColor(tar_image, cv2.COLOR_BGR2RGB)
        
        ref_mask = [mask_utils.decode(object_masks[start_end_frames[0]]) for object_masks in objects_masks]
        tar_mask = [mask_utils.decode(object_masks[start_end_frames[1]]) for object_masks in objects_masks]
        item_with_collage = self.process_pairs(ref_image, ref_mask, 
                                               tar_image, tar_mask)
        sampled_time_steps = self.sample_timestep()
        item_with_collage['time_steps'] = sampled_time_steps
        item_with_collage.update(batch)
        return item_with_collage