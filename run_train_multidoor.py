import pytorch_lightning as pl
from datasets.hico import HICODataset
from datasets.psg import PSGDataset
from datasets.pvsg import PVSGDataset
from datasets.vg import VGDataset
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
resume_path = 'lightning_logs/version_5/checkpoints/epoch=11-step=104999.ckpt'
batch_size = 4
logger_freq = 2000
learning_rate = 1e-5
sd_locked = False
only_mid_control = False
n_gpus = 4
accumulate_grad_batches = 1
max_epochs = 12

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./configs/multidoor.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# datasets
DConf = OmegaConf.load('./configs/datasets.yaml')
dataset1 = HICODataset(**DConf.Train.HICO)
dataset2 = PSGDataset(**DConf.Train.PSG)
dataset3 = PVSGDataset(**DConf.Train.PVSG)

image_data = [dataset1, dataset2]
video_data = [dataset3]

# The ratio of each dataset is adjusted by setting the __len__ 
dataset = ConcatDataset(image_data + video_data)
dataloader = DataLoader(dataset, num_workers=8, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq, split="psg_train")
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
