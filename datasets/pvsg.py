import os
import json
import cv2
import numpy as np
from PIL import Image
from .base import BaseDataset


class PVSGDataset(BaseDataset):
    def __init__(self, data_root):
        super().__init__()
        self.data_root = data_root
        with open(os.path.join(data_root, 'pvsg.json'), 'r') as f:
            anno = json.load(f)
        self.anno = anno
        self.data = anno['data']
        self.dynamic = 2

    def __len__(self):
        return 20000
    
    def get_sample(self, index):
        data = self.data[index]
        video_id = data['video_id']
        objects = data['objects']
        relations = data['relations']
        relations = self.check_isthing(relations, objects)
        num_relation = len(relations)
        assert num_relation >= 1
        relation = relations[np.random.choice(num_relation)]
        
        # prepare caption
        objects_ids = relation[:2]
        object_classes = [objects[ids - 1]['category'] for ids in objects_ids]
        predicate = relation[2]
        caption = f'The {object_classes[0]} is {predicate} the {object_classes[1]}.'
        names = object_classes
        nouns = []
        for name in names:
            if len(nouns) > 0:
                start = caption.index(name, nouns[0]["end"])
            else:
                start = caption.index(name)
            end = start + len(name)
            noun = dict(word=name, start=start, end=end)
            nouns.append(noun)
        batch = self.process_nouns_in_caption(nouns, caption)
        # random pick two frames
        num_clip = len(relation[3])
        clip = relation[3][np.random.choice(num_clip)]
        frames = np.random.randint(low=clip[0], high=clip[1], size=(2,))
        start_frame, end_frame = frames

        if '-' in video_id:
            dir_name = 'ego4d'
            raise ValueError
        elif 'P' in video_id:
            dir_name = 'epic_kitchen'
            raise ValueError
        else:
            dir_name = 'vidor' 
        video_dir = os.path.join(self.data_root, dir_name, 'frames', video_id)
        assert os.path.isdir(video_dir)

        ref_image_path = os.path.join(video_dir, f'{str(start_frame).zfill(4)}.png')
        ref_mask_path = ref_image_path.replace('frames', 'masks')
        tar_image_path = os.path.join(video_dir, f'{str(end_frame).zfill(4)}.png')
        tar_mask_path = tar_image_path.replace('frames', 'masks')
        
        ref_image = np.array(Image.open(ref_image_path).convert('RGB'))
        ref_mask = np.array(Image.open(ref_mask_path))
        ref_mask = [ref_mask == ids for ids in objects_ids]
        
        tar_image = np.array(Image.open(tar_image_path).convert('RGB'))
        tar_mask = np.array(Image.open(tar_mask_path))
        tar_mask = [tar_mask == ids for ids in objects_ids]
        
        item_with_collage = self.process_pairs(ref_image=ref_image, ref_masks=ref_mask, 
                                               tar_image=tar_image, tar_masks=tar_mask)
        sampled_time_steps = self.sample_timestep()
        
        item_with_collage['time_steps'] = sampled_time_steps
        item_with_collage.update(batch)
        return item_with_collage
        
    def check_isthing(self, relations, objects):
        thing_relations = []
        for relation in relations:
            if objects[relation[0] - 1]['is_thing'] and objects[relation[1] - 1]['is_thing']:
                thing_relations.append(relation)
        return thing_relations