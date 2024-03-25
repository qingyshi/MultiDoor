from .base import BaseDataset
import numpy as np
import cv2
import json
import os
from copy import deepcopy


class HICODataset(BaseDataset):
    def __init__(self, root, image_dir, mask_dir, annotation):
        self.root = root
        self.image_dir = os.path.join(root, image_dir)
        self.mask_dir = os.path.join(root, mask_dir)
        
        annotation_path = os.path.join(root, annotation)
        with open(annotation_path, "r") as f:
            ann = json.load(f)
        
        self.data = ann['filenames']
        self.annotations = ann['annotation']
        self.ids2verb: list = ann['verbs']
        self.ids2obj: list = ann['objects']
        self.dynamic = 0
    
    def __len__(self):
        return 20000
    
    def sample_timestep(self, max_step=1000):
        step_start = 0
        step_end = max_step
        step = np.random.randint(step_start, step_end)
        return np.array([step])

    def get_sample(self, index):
        filename = self.data[index]
        relationships = self.annotations[index]
        verbs = relationships['verb']
        objs = relationships['object']
        num_relationship = len(verbs)
        chosen_ids = int(np.random.choice(num_relationship, 1))
        verb: str = self.ids2verb[verbs[chosen_ids]].replace("_", "ing ")
        obj: str = self.ids2obj[objs[chosen_ids]].replace("_", " ")
        caption = f"The person {verb} the {obj}"
        
        image_path = os.path.join(self.image_dir, filename)
        subject_path = os.path.join(self.mask_dir, filename, str(chosen_ids), "subject.png")
        object_path = subject_path.replace("subject", "object")
        
        ref_image = cv2.imread(image_path)
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
        tar_image = ref_image.copy()
        
        subject_mask = cv2.imread(subject_path, cv2.IMREAD_GRAYSCALE) == 255
        object_mask = cv2.imread(object_path, cv2.IMREAD_GRAYSCALE) == 255
        ref_mask = [subject_mask, object_mask]
        tar_mask = deepcopy(ref_mask)
        
        item_with_collage = self.process_pairs(ref_image, ref_mask, tar_image, tar_mask)
        sampled_time_steps = self.sample_timestep()
        
        item_with_collage['time_steps'] = sampled_time_steps
        item_with_collage['img_path'] = image_path
        item_with_collage['caption'] = caption
        return item_with_collage