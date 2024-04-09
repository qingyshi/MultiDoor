from datasets.ytb_vos import YoutubeVOSDataset
from datasets.ytb_vis import YoutubeVISDataset
from datasets.saliency_modular import SaliencyDataset
from datasets.vipseg import VIPSegDataset
from datasets.mvimagenet import MVImageNetDataset
from datasets.sam import SAMDataset
from datasets.dreambooth import DreamBoothDataset
from datasets.uvo import UVODataset
from datasets.uvo_val import UVOValDataset
from datasets.mose import MoseDataset
from datasets.vitonhd import VitonHDDataset
from datasets.fashiontryon import FashionTryonDataset
from datasets.lvis import LvisDataset
from datasets.coco import CocoDataset
from datasets.coco_val import CocoValDataset
from datasets.hico import HICODataset
from datasets.psg import PSGDataset
from datasets.pvsg import PVSGDataset
from datasets.ytb_vis import YoutubeVIS21Dataset
from datasets.vg import VGDataset
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
import numpy as np 
import cv2
import torch
import json
import os
from tqdm import tqdm
from omegaconf import OmegaConf

# Datasets
DConf = OmegaConf.load('./configs/datasets.yaml')
# dataset1 = YoutubeVOSDataset(**DConf.Train.YoutubeVOS)  
# dataset2 = SaliencyDataset(**DConf.Train.Saliency) 
# dataset3 = VIPSegDataset(**DConf.Train.VIPSeg) 
# dataset4 = YoutubeVISDataset(**DConf.Train.YoutubeVIS) 
# dataset5 = MVImageNetDataset(**DConf.Train.MVImageNet)
# dataset6 = SAMDataset(**DConf.Train.SAM)
# dataset7 = UVODataset(**DConf.Train.UVO.train)
# dataset8 = VitonHDDataset(**DConf.Train.VitonHD)
# dataset9 = UVOValDataset(**DConf.Train.UVO.val)
# dataset10 = MoseDataset(**DConf.Train.Mose)
# dataset11 = FashionTryonDataset(**DConf.Train.FashionTryon)
# dataset12 = LvisDataset(**DConf.Train.Lvis)
# dataset13 = CocoDataset(**DConf.Train.COCO)
dataset14 = HICODataset(**DConf.Train.HICO)
dataset15 = PSGDataset()
dataset16 = PVSGDataset(data_root="/data00/OpenPVSG/data")
# dataset17 = VGDataset()
# dataset18 = YoutubeVIS21Dataset(**DConf.Train.YoutubeVIS21)
# dataset19 = CocoValDataset(**DConf.Train.COCOVal)
dataset = dataset16

def find_save_path(is_mask):
    image_dir = "examples/cocoval/ref"
    image_name = 0
    while True:
        if not is_mask:
            image_path = os.path.join(image_dir, str(image_name) + ".jpg")
        else:
            image_path = os.path.join(image_dir, str(image_name) + ".png")
        if os.path.exists(image_path):
            image_name += 1
        else:
            break
    return image_path

def save_image_mask_pair(image, mask):
    image_path = find_save_path(is_mask=False)
    mask_path = find_save_path(is_mask=True)
    cv2.imwrite(image_path, image)
    cv2.imwrite(mask_path, mask)

def make_test_case(item):
    ref = torch.cat([item['ref'][:, i] for i in range(2)], dim=2) * 255
    tar = item['jpg'] * 127.5 + 127.5
    hint = item['hint'] * 127.5 + 127.5
    tar_masks = item['tar_masks']
    step = item['time_steps']

    ref1, ref2 = torch.chunk(ref[0], 2, dim=1)
    mask1 = (ref1.sum(-1) != 255 * 3).to(torch.uint8) * 255
    mask2 = (ref2.sum(-1) != 255 * 3).to(torch.uint8) * 255
    ref1, ref2, mask1, mask2 = ref1.numpy()[:, :, ::-1], ref2.numpy()[:, :, ::-1], mask1.numpy(), mask2.numpy()
    save_image_mask_pair(ref1, mask1)
    save_image_mask_pair(ref2, mask2)
    
    hint_image = hint[0, :, :, :-1].numpy()
    hint_mask = hint[0, :, :, -1].numpy()
    hint_mask = np.stack([hint_mask, hint_mask, hint_mask],-1)
    ref1 = cv2.resize(ref1.astype(np.uint8), (512, 512))
    ref2 = cv2.resize(ref2.astype(np.uint8), (512, 512))
    tar = tar[0].numpy()
    vis = cv2.hconcat([ref1.astype(np.float32), ref2.astype(np.float32), hint_image.astype(np.float32), hint_mask.astype(np.float32), tar.astype(np.float32)])
    tar_mask1, tar_mask2 = tar_masks[0] * 255, tar_masks[1] * 255
    bg = "examples/cocoval/bg"
    bg_dir = 0
    while True:
        dir_path = os.path.join(bg, str(bg_dir))
        if os.path.exists(dir_path):
            bg_dir += 1
            continue
        else:
            os.makedirs(dir_path)
            bg_path = os.path.join(dir_path, "bg.jpg")
            bg_mask_path1 = os.path.join(dir_path, "0.png")
            bg_mask_path2 = os.path.join(dir_path, "1.png")
            cv2.imwrite(bg_path, tar[:, :, ::-1])
            cv2.imwrite(bg_mask_path1, tar_mask1[0].numpy())
            cv2.imwrite(bg_mask_path2, tar_mask2[0].numpy())
            break        
    # cv2.imwrite('sample_vis.jpg', vis[:, :, ::-1])
    # anno = {}
    # anno['image_path'] = item['image_path'][0]
    # anno['caption'] = item['caption'][0]
    # anno['chosen_objs'] = item['chosen_objs'].tolist()[0]
    # annotations.append(anno)
    
def vis_sample(item):
    ref: torch.Tensor = item['ref'] # (1, 2, 224, 224, 3)
    ref = ref.transpose(1, 2).flatten(2, 3) * 255
    tar = item['jpg'] * 127.5 + 127.5
    hint = item['hint'] * 127.5 + 127.5
    step = item['time_steps']
    print(item['ref'].shape, tar.shape, hint.shape, step.shape)

    ref = ref[0].numpy()
    tar = tar[0].numpy()
    hint_image = hint[0, :, :, :-1].numpy()
    hint_mask = hint[0, :, :, -1].numpy()
    hint_mask = np.stack([hint_mask, hint_mask, hint_mask],-1)
    ref = cv2.resize(ref.astype(np.uint8), (1024, 512))
    vis = cv2.hconcat([ref.astype(np.float32), hint_image.astype(np.float32), hint_mask.astype(np.float32), tar.astype(np.float32)])
    cv2.imwrite('sample_vis.jpg', vis[:, :, ::-1])
    print(item['caption'][0])
    
dataloader = DataLoader(dataset, num_workers=8, batch_size=1, shuffle=True)
print('len dataloader: ', len(dataloader))
annotations = []
for data in tqdm(dataloader):
    vis_sample(data)