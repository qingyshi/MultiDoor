import sys
import numpy as np
from PIL import Image
import torch
import os
from tqdm import tqdm
import cv2
import clip
from eval_dataset import CustomDataset
from utils import load_clip_model
from einops import rearrange
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--result_dir', type=str, default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument("--model_name", type=str, default=None)
parser.add_argument('--device', type=str, default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')

opt = parser.parse_args()
args = {}
test_dataset = CustomDataset()
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=1, 
    num_workers=4, 
    pin_memory=True, 
    shuffle=False,#sampler=train_sampler, 
    drop_last=True
)
model, processor = load_clip_model(opt.model_name, opt.device)
        
sum=0
count=0
for crop_tensor, ref_image_tensor in tqdm(test_dataloader):
    crop_tensor = crop_tensor.to('cuda')
    ref_image_tensor = ref_image_tensor.to('cuda')
    result_feat = clip_model.encode_image(crop_tensor)
    ref_feat = clip_model.encode_image(ref_image_tensor)
    result_feat = result_feat.to('cpu')
    ref_feat = ref_feat.to('cpu')
    result_feat = result_feat / result_feat.norm(dim=-1, keepdim=True)
    ref_feat = ref_feat / ref_feat.norm(dim=-1, keepdim=True)
    similarity = (100.0 * result_feat @ ref_feat.T)
    sum = sum + similarity.item()
    count = count + 1
    
print(sum / count)