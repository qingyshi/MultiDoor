import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import cv2
from .data_utils import * 
from panopticapi.utils import rgb2id
from PIL import Image
from .base import BaseDataset

class VIPSegDataset(BaseDataset):
    def __init__(self, image_dir, anno, meta, caption):
        self.image_root = image_dir
        self.anno_root = anno
        video_dirs = []
        video_dirs = os.listdir(self.image_root)
        self.data = video_dirs
        self.size = (512,512)
        self.clip_size = (224,224)
        self.dynamic = 1
        
        with open(caption, 'r') as file:
            caption = json.load(file)
            
        self.caption = caption
        # with open(meta, 'r') as file:
        #     ann = json.load(file)
        
        # self.vid_name2ann_per_frame = {}
        # for video_ann in ann['annotations']:
        #     self.vid_name2ann_per_frame[video_ann['video_id']] = video_ann['annotations']   # list of annotations per frame
            
        # self.id2name = {}
        # self.id2is_thing = {}
        # for cat in ann['categories']:
        #     self.id2name[cat['id']] = cat['name']
        #     self.id2is_thing[cat['id']] = cat['isthing']

    def __len__(self):
        return 40000

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

    def get_sample(self, idx):
        video_name = self.data[idx]
        
        caption = self.load_caption(video_name)
            
        # ann_per_frame = self.vid_name2ann_per_frame[video_name]     # list
        video_path = os.path.join(self.image_root, video_name)
        frames = os.listdir(video_path)

        # Sampling frames
        min_interval = len(frames) // 100
        start_frame_index = np.random.randint(low=0, high=len(frames) - min_interval)
        end_frame_index = start_frame_index + np.random.randint(min_interval, len(frames) - start_frame_index)
        end_frame_index = min(end_frame_index, len(frames) - 1)
        # end_frame_ann = ann_per_frame[end_frame_index]
        # start_end_frame_index = [start_frame_index, end_frame_index]
        
        # Get image path
        ref_image_name = frames[start_frame_index]
        tar_image_name = frames[end_frame_index]
        ref_image_path = os.path.join(self.image_root, video_name, ref_image_name)
        tar_image_path = os.path.join(self.image_root, video_name, tar_image_name)

        ref_mask_path = ref_image_path.replace('images','panomasksRGB').replace('.jpg', '.png')
        tar_mask_path = tar_image_path.replace('images','panomasksRGB').replace('.jpg', '.png')

        # Read Image and Mask
        ref_image = cv2.imread(ref_image_path)
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

        tar_image = cv2.imread(tar_image_path)
        tar_image = cv2.cvtColor(tar_image, cv2.COLOR_BGR2RGB)

        ref_mask = np.array(Image.open(ref_mask_path).convert('RGB'))
        ref_mask = rgb2id(ref_mask)

        tar_mask = np.array(Image.open(tar_mask_path).convert('RGB'))
        tar_mask = rgb2id(tar_mask)
        
        ref_ids = np.unique(ref_mask)
        tar_ids = np.unique(tar_mask)

        common_ids = list(np.intersect1d(ref_ids, tar_ids))
        common_ids = [i for i in common_ids if i != 0]
        
        if len(common_ids) >= 2:
            chosen_id = np.random.choice(common_ids, 2, replace=False)
        else:
            chosen_id = np.array(common_ids)
        # chosen_cat_id = []
        # for single_id in chosen_id:
        #     obj_ids = [obj['id'] for obj in end_frame_ann['segments_info']]
        #     chosen_cat_id.append(obj_ids.index(single_id))
        
        # chosen_id = obj_ids        
        # names = [self.id2name[cat_id] for cat_id in chosen_cat_id]
        
        ref_mask = [ref_mask == single_id for single_id in chosen_id]
        tar_mask = [tar_mask == single_id for single_id in chosen_id]

        len_mask = len(self.check_connect(ref_mask[0].astype(np.uint8)))
        assert len_mask == 1

        item_with_collage = self.process_pairs(ref_image, ref_mask, tar_image, tar_mask)
        sampled_time_steps = self.sample_timestep()
        
        item_with_collage['time_steps'] = sampled_time_steps
        # item_with_collage['video_id'] = video_name
        item_with_collage['img_path'] = tar_image_path
        item_with_collage['caption'] = caption
        return item_with_collage

    def check_connect(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt_area = [cv2.contourArea(cnt) for cnt in contours]
        return cnt_area

