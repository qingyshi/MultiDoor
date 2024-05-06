import pytorch_lightning as pl
from datasets.hico import HICODataset, HICOTestDataset
from datasets.psg import PSGDataset
from datasets.pvsg import PVSGDataset
from datasets.coco import CocoDataset
from datasets.sam import SAMDataset
from datasets.vipseg import VIPSegDataset
from datasets.ytb_vis import YoutubeVIS21Dataset, YoutubeVISDataset
from datasets.ytb_vos import YoutubeVOSDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from cldm.hack import disable_verbosity, enable_sliced_attention
from omegaconf import OmegaConf

save_memory = False
disable_verbosity()
if save_memory:
    enable_sliced_attention()

# Configs
resume_path = 'checkpoints/control_sd21_ini.ckpt'
batch_size = 4
logger_freq = 2000
learning_rate = 1e-5
sd_locked = False
only_mid_control = False
n_gpus = 4
accumulate_grad_batches = 2
max_epochs = 12

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./configs/multidoor.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# datasets
DConf = OmegaConf.load('./configs/datasets.yaml')
dataset1 = HICODataset(**DConf.Train.HICO.Train)
dataset2 = HICOTestDataset(**DConf.Train.HICO.Test)
dataset3 = PSGDataset(**DConf.Train.PSG)
dataset4 = PVSGDataset(**DConf.Train.PVSG)

image_data = [dataset1, dataset2, dataset3]
video_data = [dataset4]

# The ratio of each dataset is adjusted by setting the __len__ 
dataset = ConcatDataset(image_data + video_data + video_data)
dataloader = DataLoader(dataset, num_workers=8, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq, split="v4")
trainer = pl.Trainer(
    gpus=n_gpus,
    strategy="ddp",
    precision=16,
    accelerator="gpu",
    callbacks=[logger],
    progress_bar_refresh_rate=1,
    accumulate_grad_batches=accumulate_grad_batches,
    max_epochs=max_epochs
)


# Train!
trainer.fit(model, dataloader)