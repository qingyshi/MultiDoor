"""Calculates the CLIP Scores

The CLIP model is a contrasitively learned language-image model. There is
an image encoder and a text encoder. It is believed that the CLIP model could 
measure the similarity of cross modalities. Please find more information from 
https://github.com/openai/CLIP.

The CLIP Score measures the Cosine Similarity between two embedded features.
This repository utilizes the pretrained CLIP Model to calculate 
the mean average of cosine similarities. 

See --help to see further details.

Code apapted from https://github.com/mseitzer/pytorch-fid and https://github.com/openai/CLIP.

Copyright 2023 The Hong Kong Polytechnic University

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import os.path as osp
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import torch
from PIL import Image
from eval_dataset import CustomDataset
from utils import load_clip_model
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--clip-model', type=str, default='ViT-B/32',
                    help='CLIP model to use')
parser.add_argument('--num-workers', type=int,
                    help=('Number of processes to use for data loading. '
                          'Defaults to `min(8, num_cpus)`'))
parser.add_argument('--device', type=str, default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--real_flag', type=str, default='img',
                    help=('The modality of real path. '
                          'Default to img'))
parser.add_argument('--fake_flag', type=str, default='txt',
                    help=('The modality of real path. '
                          'Default to txt'))
parser.add_argument('real_path', type=str, 
                    help=('Paths to the generated images or '
                          'to .npz statistic files'))
parser.add_argument('fake_path', type=str,
                    help=('Paths to the generated images or '
                          'to .npz statistic files'))

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}

TEXT_EXTENSIONS = {'txt'}



@torch.no_grad()
def calculate_clip_score(dataloader, model, device):
    model.eval()
    logit_scale = model.logit_scale.exp()
    score_acc = 0.
    sample_num = 0
    
    for batch in tqdm(dataloader):
        if batch['real'].dim() == 3:  # Check if input is text (input_ids)
            real_features = model.get_text_features(input_ids=batch['real'].to(device))
        else:
            real_features = model.get_image_features(pixel_values=batch['real'].to(device))

        if batch['fake'].dim() == 3:
            fake_features = model.get_text_features(input_ids=batch['fake'].to(device))
        else:
            fake_features = model.get_image_features(pixel_values=batch['fake'].to(device))

        real_features = real_features / real_features.norm(dim=1, keepdim=True)
        fake_features = fake_features / fake_features.norm(dim=1, keepdim=True)
        
        score = logit_scale * (fake_features * real_features).sum(dim=1)
        score_acc += score.sum().item()
        sample_num += real_features.shape[0]

    return score_acc / sample_num


def main():
    args = parser.parse_args()

    device = torch.device(args.device if args.device else 'cuda' if torch.cuda.is_available() else 'cpu')
    model, processor = load_clip_model(args.clip_model, device)

    dataset = CustomDataset(args.fake_path, args.real_flag, args.fake_flag, processor=processor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers or 0, pin_memory=True)

    clip_score = calculate_clip_score(dataloader, model, device)
    print('CLIP Score:', clip_score)


if __name__ == '__main__':
    main()