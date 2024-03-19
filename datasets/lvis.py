import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import cv2
from .data_utils import * 
from .base import BaseDataset
from pycocotools import mask as mask_utils
from lvis import LVIS

class LvisDataset(BaseDataset):
    def __init__(self, image_dir, json_path, caption):
        self.image_dir = image_dir
        self.json_path = json_path
        lvis_api = LVIS(json_path)
        img_ids = sorted(lvis_api.imgs.keys())
        imgs = lvis_api.load_imgs(img_ids)
        anns = [lvis_api.img_ann_map[img_id] for img_id in img_ids]
        self.data = imgs
        self.annos = anns
        self.caption = json.load(open(caption, 'r'))
        self.lvis_api = lvis_api
        self.size = (512,512)
        self.clip_size = (224,224)
        self.dynamic = 0

    def register_subset(self, path):
        data = os.listdir(path)
        data = [ os.path.join(path, i) for i in data if '.json' in i]
        self.data = self.data + data

    def get_sample(self, idx):
        # ==== get pairs =====
        image_name = self.data[idx]['coco_url'].split('/')[-1]
        caption = self.load_caption(image_name)
            
        image_path = os.path.join(self.image_dir, image_name)
        image = cv2.imread(image_path)
        ref_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        anno = self.annos[idx]
        anno = sorted(anno, key=lambda x: x['area'], reverse=True)
        obj_ids = []    # obj_ids: valid indices of anno
        names = []      # names: valid names in anno
        for i in range(len(anno)):
            obj = anno[i]
            area = obj['area']
            category_id = obj['category_id']
            category_info = self.lvis_api.cats[category_id]
            name = category_info['name']
            if area > 3600:
                obj_ids.append(i)
                # names.append(name)
                if len(obj_ids) == 2:
                    break
        assert len(anno) > 0
        annos = [anno[obj_id] for obj_id in obj_ids]
        ref_mask = [self.lvis_api.ann_to_mask(anno) for anno in annos]

        tar_image, tar_mask = ref_image.copy(), ref_mask.copy()
        item_with_collage = self.process_pairs(ref_image, ref_mask, tar_image, tar_mask)
        sampled_time_steps = self.sample_timestep()
        item_with_collage['time_steps'] = sampled_time_steps
        item_with_collage['img_path'] = image_path
        item_with_collage['caption'] = caption
        # item_with_collage['class_token_ids'] = torch.tensor(class_token_ids)
        return item_with_collage

    def __len__(self):
        return 10000

    def check_region_size(self, image, yyxx, ratio, mode = 'max'):
        pass_flag = True
        H,W = image.shape[0], image.shape[1]
        H,W = H * ratio, W * ratio
        y1,y2,x1,x2 = yyxx
        h,w = y2-y1,x2-x1
        if mode == 'max':
            if h > H or w > W:
                pass_flag = False
        elif mode == 'min':
            if h < H or w < W:
                pass_flag = False
        return pass_flag



        
