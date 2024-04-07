import torch
import json
import numpy as np
import os
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
import torchvision.transforms.functional as F
from .base import BaseDataset


class CocoDataset(BaseDataset):
    def __init__(self, root, annotation, caption, transforms=None):
        self.root = root
        self.coco = COCO(annotation)
        self.data = list(sorted(self.coco.imgs.keys())) 
        with open(caption) as f:
            self.caption = json.load(f)
        
        self.transforms = transforms
        self.cat_to_names = {cat_id: cat['name'] for cat_id, cat in self.coco.cats.items()}
        self.dynamic = 0
    
    def get_sample(self, index):
        coco = self.coco
        img_id = self.data[index]        
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']
        caption, chosen_objs = self.load_caption(path)
        image_path = os.path.join(self.root, path)
        img = np.array(Image.open(image_path).convert("RGB"))
        sorted_annotation = sorted(coco_annotation, key=lambda x: x["area"], reverse=True)

        num_objs = len(sorted_annotation)
        if num_objs >= 2:
            chosen_objs = [0, 1]
        else:
            raise Exception
        
        names = []
        masks = []
        for i, idx in enumerate(chosen_objs):
            ann = sorted_annotation[idx]
            mask = self.coco.annToMask(ann)
            masks.append(mask)
            names.append(self.cat_to_names[ann['category_id']])
        assert len(names) == 2
        item_with_collage = self.process_pairs(ref_image=img, ref_mask=masks,               
                                               tar_image=img.copy(), tar_mask=masks.copy())
        sampled_time_steps = self.sample_timestep()
        
        item_with_collage['time_steps'] = sampled_time_steps
        # item_with_collage['chosen_objs'] = chosen_objs.tolist()
        # item_with_collage['names'] = names
        # item_with_collage['image_path'] = image_path
        item_with_collage['caption'] = caption
        return item_with_collage

    def __len__(self):
        return 40000