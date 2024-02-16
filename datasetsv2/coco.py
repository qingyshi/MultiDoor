import torch
import numpy as np
import os
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
import torchvision.transforms.functional as F
from .base import BaseDataset


class CocoDataset(BaseDataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.coco = COCO(annotation)
        self.data = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms
        self.cat_to_names = {cat_id: cat['name'] for cat_id, cat in self.coco.cats.items()}
        self.dynamic = 0

    def get_sample(self, index):
        coco = self.coco
        img_id = self.data[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']
        image_path = os.path.join(self.root, path)
        img = np.array(Image.open(image_path).convert("RGB"))

        num_objs = len(coco_annotation)

        masks = []
        names = []
        
        if num_objs >= 2:
            chosen_objs = np.random.choice(list(range(num_objs)), 2, replace=False)
        else:
            chosen_objs = [num_objs]
        
        for i, idx in enumerate(chosen_objs):
            ann = coco_annotation[idx]
            if 'segmentation' in ann:
                mask = self.coco.annToMask(ann)
                masks.append(mask)
            else:
                masks.append(np.zeros((img.height, img.width)))

            names.append(self.cat_to_names[ann['category_id']])

        # 将掩码列表转换为一个多维数组
        if len(masks) == 2:
            masks = np.stack(masks, axis=0)
        else:
            masks.append(np.zeros((img.height, img.width)))
            masks = np.stack(masks, axis=0)
            names.append("none")

        # labels = torch.tensor([ann['category_id'] for ann in coco_annotation], dtype=torch.int64)
        item_with_collage = self.process_pairs(ref_image=img, ref_mask=masks, 
                                  tar_image=img.copy(), tar_mask=masks.copy())
        sampled_time_steps = self.sample_timestep()
        item_with_collage['time_steps'] = sampled_time_steps
        item_with_collage["names"] = names
        item_with_collage['img_path'] = image_path
        return item_with_collage

    def __len__(self):
        return len(self.data)