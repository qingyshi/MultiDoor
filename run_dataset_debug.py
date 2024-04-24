from datasets.ytb_vos import YoutubeVOSDataset
from datasets.ytb_vis import YoutubeVISDataset
from datasets.vipseg import VIPSegDataset
from datasets.coco_val import CocoValDataset
from datasets.hico import HICODataset, HICOTestDataset
from datasets.psg import PSGDataset
from datasets.pvsg import PVSGDataset
from datasets.ytb_vis import YoutubeVIS21Dataset
from datasets.vg import VGDataset
from datasets.burst import BurstDataset
from datasets.ovis import OVISDataset
from datasets.lvvis import LVVISDataset
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
import numpy as np
import time
import cv2
import torch
import os
from tqdm import tqdm
from omegaconf import OmegaConf

# Datasets
DConf = OmegaConf.load('./configs/datasets.yaml')
dataset1 = HICODataset(**DConf.Train.HICO.Train)
dataset2 = HICOTestDataset(**DConf.Train.HICO.Test)
dataset3 = PSGDataset(**DConf.Train.PSG)
dataset4 = BurstDataset(**DConf.Train.Burst.Train)
dataset5 = BurstDataset(**DConf.Train.Burst.Val)
dataset6 = BurstDataset(**DConf.Train.Burst.Test)
dataset7 = LVVISDataset(**DConf.Train.LVVIS)
dataset8 = OVISDataset(**DConf.Train.OVIS)
dataset9 = PVSGDataset(**DConf.Train.PVSG)
dataset10 = YoutubeVIS21Dataset(**DConf.Train.YoutubeVIS21)
dataset11 = YoutubeVISDataset(**DConf.Train.YoutubeVIS)
dataset12 = YoutubeVOSDataset(**DConf.Train.YoutubeVOS)
tokenizer = dataset1.tokenizer
image_data = [dataset1, dataset2, dataset3]
video_data = [dataset9, dataset10, dataset11, dataset12]

# The ratio of each dataset is adjusted by setting the __len__ 
dataset = ConcatDataset(image_data + video_data)
dataset = dataset1
dataloader = DataLoader(dataset, num_workers=0, batch_size=1, shuffle=True)

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
    ref1, ref2 = item['ref'][0, 0] * 255, item['ref'][0, 1] * 255
    tar = item['jpg'][0] * 127.5 + 127.5
    hint = item['hint'][0] * 127.5 + 127.5
    tar_masks = item['target_masks'][0] * 255
    caption = item['caption'][0]

    mask1 = (ref1.sum(-1) != 255 * 3).to(torch.uint8) * 255
    mask2 = (ref2.sum(-1) != 255 * 3).to(torch.uint8) * 255
    ref1, ref2 = ref1.numpy()[:, :, ::-1], ref2.numpy()[:, :, ::-1]
    mask1, mask2 = mask1.numpy(), mask2.numpy()
    # save_image_mask_pair(ref1, mask1)
    # save_image_mask_pair(ref2, mask2)
    
    tar = tar.numpy()[:, :, ::-1]
    tar_mask1, tar_mask2 = tar_masks[0].numpy(), tar_masks[1].numpy()
    bg = f"examples/cocovalv2/{caption}"
    count = 0
    while True:
        dir_path = os.path.join(bg, str(count))
        if os.path.exists(dir_path):
            count += 1
            continue
        else:
            os.makedirs(dir_path)
            bg_path = os.path.join(dir_path, "bg.jpg")
            bg_mask_path1 = os.path.join(dir_path, "0.png")
            bg_mask_path2 = os.path.join(dir_path, "1.png")
            ref1_path = os.path.join(dir_path, "ref1.jpg")
            ref2_path = os.path.join(dir_path, "ref2.jpg")
            cv2.imwrite(bg_path, tar)
            cv2.imwrite(bg_mask_path1, tar_mask1)
            cv2.imwrite(bg_mask_path2, tar_mask2)
            cv2.imwrite(ref1_path, ref1)
            cv2.imwrite(ref2_path, ref2)
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
    target_masks = item['target_masks'][0][..., None].repeat(1, 1, 1, 3) * 255
    target_masks = target_masks.transpose(0, 1).flatten(1, 2).numpy()

    ref = ref[0].numpy()
    tar = tar[0].numpy()
    hint_image = hint[0, :, :, :-1].numpy()
    hint_mask = hint[0, :, :, -1].numpy()
    hint_mask = np.stack([hint_mask, hint_mask, hint_mask], -1)
    ref = cv2.resize(ref.astype(np.uint8), (1024, 512))
    vis = cv2.hconcat([ref.astype(np.float32), hint_image.astype(np.float32), target_masks.astype(np.float32), tar.astype(np.float32)])
    cv2.imwrite('sample_vis.jpg', vis[:, :, ::-1])
    input_ids = item["caption"][0]
    decode_string = tokenizer.decode(input_ids)
    print(decode_string)
    print(item['names'])

count = 0
start = time.time()
for data in tqdm(dataloader):
    break