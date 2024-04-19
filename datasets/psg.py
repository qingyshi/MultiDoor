import torch
import json
import numpy as np
import os
from PIL import Image
from pycocotools.coco import COCO
import panopticapi.utils as utils
import torchvision.transforms.functional as F
from .base import BaseDataset


class PSGDataset(BaseDataset):
    def __init__(self, root="/data00/datasets/coco", relation="/data00/psg/psg.json"):
        super().__init__()
        self.root = root
        self.ann = json.load(open(relation, "r"))
        self.data = self.ann.get('data')
        self.id2predicate: list = self.ann.get('predicate_classes')
        self.id2class: list = self.ann.get('thing_classes')
        self.dynamic = 2

    def __len__(self):
        return 40000
    
    def get_sample(self, index):
        ann = self.data[index]
        file_name = ann.get('file_name')
        pan_seg_file_name = ann.get('pan_seg_file_name')
        image_path = os.path.join(self.root, file_name)
        mask_path = os.path.join(self.root, "annotations", pan_seg_file_name)
        img = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("RGB"))
        mask = utils.rgb2id(mask)
        
        relations = ann.get('relations')
        segments_info = ann.get('segments_info')
        relations = self.check_isthing(relations, segments_info)
        num_relation = len(relations)
        assert num_relation >= 1
        
        relation = relations[np.random.choice(num_relation)]
        objects = [self.id2class[segments_info[relation[0]]['category_id']], 
                   self.id2class[segments_info[relation[1]]['category_id']]]
        objects_id = [segments_info[relation[0]]['id'], 
                      segments_info[relation[1]]['id']]
        predicate = self.id2predicate[relation[2]]
        caption = f'The {objects[0]} is {predicate} the {objects[1]}.'
        names = objects
        nouns = []
        for name in names:
            start = caption.index(name)
            end = start + len(name)
            noun = dict(word=name, start=start, end=end)
            nouns.append(noun)
        batch = self.process_nouns_in_caption(nouns, caption)
        
        masks = [mask == id for id in objects_id]
        item_with_collage = self.process_pairs(ref_image=img, ref_masks=masks, 
                                               tar_image=img.copy(), tar_masks=masks.copy())
        sampled_time_steps = self.sample_timestep()
        
        item_with_collage['time_steps'] = sampled_time_steps
        # item_with_collage['image_path'] = image_path
        item_with_collage.update(batch)
        return item_with_collage
    
    def check_isthing(self, relations, segments_info):
        thing_relations = []
        for relation in relations:
            if segments_info[relation[0]]['isthing'] and segments_info[relation[1]]['isthing']:
                thing_relations.append(relation)
        return thing_relations