import torch
import json
import numpy as np
import os
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
import torchvision.transforms.functional as F
from .base import BaseDataset


class CocoValDataset(BaseDataset):
    def __init__(self, root, annotation, transforms=None):
        super().__init__()
        self.root = root
        self.coco = COCO(annotation)
        self.data = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms
        self.cat_to_names = {cat_id: cat['name'] for cat_id, cat in self.coco.cats.items()}
        self.dynamic = 0

    def __getitem__(self, idx):
        while(True):
            try:
                # idx = np.random.randint(0, len(self.data)-1)
                item = self.get_sample(idx)
                return item
            except:
                idx = np.random.randint(0, len(self.data)-1)

    def get_sample(self, index):
        coco = self.coco
        img_id = self.data[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']
        image_path = os.path.join(self.root, path)
        img = np.array(Image.open(image_path).convert("RGB"))

        num_objs = len(coco_annotation)
        if num_objs >= 2:
            chosen_objs = np.random.choice(list(range(num_objs)), 2, replace=False)
        else:
            raise Exception
        
        names = []
        masks = []
        bboxes = []
        for idx in chosen_objs:
            ann = coco_annotation[idx]
            mask = self.coco.annToMask(ann)
            bboxes.append(self.mask2bbox(mask))
            masks.append(mask)
            names.append(self.cat_to_names[ann['category_id']])

        caption = "_".join(names)
        item_with_collage = self.process_pairs(ref_image=img, ref_masks=masks, 
                                               tar_image=img.copy(), tar_masks=masks.copy())
        sampled_time_steps = self.sample_timestep()
        
        item_with_collage['time_steps'] = sampled_time_steps
        item_with_collage['image_path'] = image_path
        item_with_collage['caption'] = caption
        item_with_collage['chosen_objs'] = chosen_objs
        return item_with_collage

    def __len__(self):
        return len(self.data)
    
    def mask2bbox(self, mask):
        bbox = np.zeros_like(mask)
        y1 = np.nonzero(np.max(mask, axis=1) == 1)[0].min()
        y2 = np.nonzero(np.max(mask, axis=1) == 1)[0].max()
        x1 = np.nonzero(np.max(mask, axis=0) == 1)[0].min()
        x2 = np.nonzero(np.max(mask, axis=0) == 1)[0].max()
        bbox[y1: y2, x1: x2] = 1
        return bbox