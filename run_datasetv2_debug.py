from datasetsv2.ytb_vos import YoutubeVOSDataset
from datasetsv2.ytb_vis import YoutubeVISDataset
from datasetsv2.saliency_modular import SaliencyDataset
from datasetsv2.vipseg import VIPSegDataset
from datasetsv2.mvimagenet import MVImageNetDataset
from datasetsv2.sam import SAMDataset
from datasetsv2.dreambooth import DreamBoothDataset
from datasetsv2.uvo import UVODataset
from datasetsv2.uvo_val import UVOValDataset
from datasetsv2.mose import MoseDataset
from datasetsv2.vitonhd import VitonHDDataset
from datasetsv2.fashiontryon import FashionTryonDataset
from datasetsv2.lvis import LvisDataset
from datasetsv2.coco import CocoDataset
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
import numpy as np 
import cv2
import torch
from omegaconf import OmegaConf

# Datasets
DConf = OmegaConf.load('./configs/datasetsv2.yaml')
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
dataset13 = CocoDataset(**DConf.Train.COCO)

dataset = dataset13


def vis_sample(item):
    ref = torch.cat([item['ref'][:, i] for i in range(2)], dim=2) * 255
    tar = item['jpg'] * 127.5 + 127.5
    hint = item['hint'] * 127.5 + 127.5
    step = item['time_steps']
    print(ref.shape, tar.shape, hint.shape, step.shape)

    ref = ref[0].numpy()
    tar = tar[0].numpy()
    hint_image = hint[0, :, :, :-1].numpy()
    hint_mask = hint[0, :, :, -1].numpy()
    hint_mask = np.stack([hint_mask, hint_mask,hint_mask],-1)
    ref = cv2.resize(ref.astype(np.uint8), (512, 512))
    vis = cv2.hconcat([ref.astype(np.float32), hint_image.astype(np.float32), hint_mask.astype(np.float32), tar.astype(np.float32)])
    cv2.imwrite('sample_vis.jpg', vis[:, :, ::-1])
    # print(item['caption'][0])
    print(item['names'][0])


dataloader = DataLoader(dataset, num_workers=8, batch_size=2, shuffle=True)
print('len dataloader: ', len(dataloader))
for data in dataloader:
    vis_sample(data) 
    continue