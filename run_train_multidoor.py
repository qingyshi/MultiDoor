import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasetsv2.ytb_vos import YoutubeVOSDataset
from datasetsv2.ytb_vis import YoutubeVISDataset
from datasetsv2.saliency_modular import SaliencyDataset
from datasetsv2.vipseg import VIPSegDataset
from datasetsv2.mvimagenet import MVImageNetDataset
from datasetsv2.sam import SAMDataset
from datasetsv2.uvo import UVODataset
from datasetsv2.uvo_val import UVOValDataset
from datasetsv2.mose import MoseDataset
from datasetsv2.vitonhd import VitonHDDataset
from datasetsv2.fashiontryon import FashionTryonDataset
from datasetsv2.lvis import LvisDataset
from datasetsv2.coco import CocoDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from torch.utils.data import ConcatDataset
from cldm.hack import disable_verbosity, enable_sliced_attention
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint

save_memory = False
disable_verbosity()
if save_memory:
    enable_sliced_attention()

# Configs
resume_path = 'checkpoints/epoch=10.ckpt'
batch_size = 6
logger_freq = 1000
learning_rate = 1e-5
sd_locked = False
only_mid_control = False
n_gpus = 4
accumulate_grad_batches = 1

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./configs/multidoorv2.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# datasetsv2
DConf = OmegaConf.load('./configs/datasetsv2.yaml')
dataset1 = YoutubeVOSDataset(**DConf.Train.YoutubeVOS)  
# dataset2 =  SaliencyDataset(**DConf.Train.Saliency) 
dataset3 = VIPSegDataset(**DConf.Train.VIPSeg) 
dataset4 = YoutubeVISDataset(**DConf.Train.YoutubeVIS) 
# dataset5 = MVImageNetDataset(**DConf.Train.MVImageNet)
# dataset6 = SAMDataset(**DConf.Train.SAM)
# dataset7 = UVODataset(**DConf.Train.UVO.train)
# dataset8 = VitonHDDataset(**DConf.Train.VitonHD)
# dataset9 = UVOValDataset(**DConf.Train.UVO.val)
# dataset10 = MoseDataset(**DConf.Train.Mose)
# dataset11 = FashionTryonDataset(**DConf.Train.FashionTryon)
dataset12 = LvisDataset(**DConf.Train.Lvis)
dataset13 = CocoDataset(**DConf.Train.COCO)

# image_data = [dataset2, dataset6, dataset12]
# video_data = [dataset1, dataset3, dataset4, dataset7, dataset9, dataset10 ]
# video_data = [dataset1, dataset3, dataset4, dataset7, dataset10]
# tryon_data = [dataset8, dataset11]
# threed_data = [dataset5]
image_data = [dataset12, dataset13]
video_data = [dataset1, dataset3, dataset4]

# The ratio of each dataset is adjusted by setting the __len__ 
dataset = ConcatDataset(image_data + video_data + video_data)
dataloader = DataLoader(dataset, num_workers=8, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints/",
    filename="{epoch}",
    save_top_k=-1,
    every_n_epochs=1,
    verbose=True,
)
trainer = pl.Trainer(
    gpus=n_gpus,
    strategy="ddp",
    precision=16,
    accelerator="gpu",
    callbacks=[logger, checkpoint_callback],
    progress_bar_refresh_rate=1,
    accumulate_grad_batches=accumulate_grad_batches,
    max_epochs=12
)


# Train!
trainer.fit(model, dataloader)
