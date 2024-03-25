import json
import os
import cv2
import numpy as np
from copy import deepcopy
from .base import BaseDataset


class VGDataset(BaseDataset):
    def __init__(self, data_root="/data00/VisualGenome/", image_dir="VG_100K", mask_dir="relationships", anno_file="relationships.json"):
        self.data_root = data_root
        self.image_dir = os.path.join(data_root, image_dir)
        self.mask_dir = os.path.join(data_root, mask_dir)
        with open(os.path.join(data_root, anno_file)) as file:
            self.data = json.load(file)
        self.dynamic = 0
    
    def __len__(self):
        return len(self.data)
    
    def get_sample(self, index):
        data = self.data[index]
        relations = data['relationships']
        image_id = data['image_id']
        num_relation = len(relations)
        relation = relations[np.random.choice(num_relation)]
        relation_id = relation['relationship_id']
        
        # prepare caption
        subject = relation['subject']['name']
        object = relation['object']['name']
        predicate = relation['predicate'].lower()
        caption = f"The {subject} is {predicate} the {object}."
        
        image_path = os.path.join(self.image_dir, str(image_id) + '.jpg')
        sub_mask_path = os.path.join(self.mask_dir, str(image_id), str(relation_id), "subject.png")
        obj_mask_path = sub_mask_path.replace('subject.png', 'object.png')
        
        assert os.path.exists(image_path) and os.path.exists(sub_mask_path) and os.path.exists(obj_mask_path)
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        sub_mask = cv2.imread(sub_mask_path, cv2.IMREAD_GRAYSCALE) == 255
        obj_mask = cv2.imread(obj_mask_path, cv2.IMREAD_GRAYSCALE) == 255
        mask = [sub_mask.astype(np.uint8), obj_mask.astype(np.uint8)]
        
        item_with_collage = self.process_pairs(ref_image=image, ref_mask=mask,
                                               tar_image=image.copy, tar_mask=deepcopy(mask), max_ratio=1.)
        sampled_time_steps = self.sample_timestep()
        
        item_with_collage['time_steps'] = sampled_time_steps
        item_with_collage['img_path'] = image_path
        item_with_collage['caption'] = caption
        return item_with_collage