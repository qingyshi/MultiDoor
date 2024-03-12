<p align="center">

  <h2 align="center">MultiDoor</h2>
  <p align="center">
    <a href="https://scholar.google.com.hk/citations?user=VpSqhJAAAAAJ&hl=zh-CN"><strong>Qingyu Shi</strong></a>
    ·
    <a href="https://scholar.google.com.hk/citations?user=SSI90d4AAAAJ&hl=zh-CN&oi=ao"><strong>Lu Qi</strong></a>
    <br>
    <b>Peking University &nbsp; | &nbsp;  UC Merced  | &nbsp; Skywork AI </b>
  </p>
  
  <!-- <table align="center">
    <tr>
    <td>
      <img src="assets/Figures/Teaser.png">
    </td>
    </tr>
  </table> -->



## Installation
Install with `conda`: 
```bash
conda env create -f environment.yaml
conda activate anydoor
```
or `pip`:
```bash
pip install -r requirements.txt
```
Additionally, for training, you need to install panopticapi, pycocotools, and lvis-api.
```bash
pip install git+https://github.com/cocodataset/panopticapi.git

pip install pycocotools -i https://pypi.douban.com/simple

pip install lvis
```
## Training
<!-- Download AnyDoor checkpoint: 
* [ModelScope](https://modelscope.cn/models/damo/AnyDoor/files)
* [HuggingFace](https://huggingface.co/spaces/xichenhku/AnyDoor/tree/main) -->

<!-- **Note:** We include all the optimizer params for Adam, so the checkpoint is big. You could only keep the "state_dict" to make it much smaller. -->

### Download checkpoints

Download DINOv2 checkpoint and revise `./configs/multidoor.yaml` for the path (line 84).
* URL: https://github.com/facebookresearch/dinov2?tab=readme-ov-file

Download CLIP-ViT-H-14 and revise `./configs/multidoor.yaml` for the path (line 89).
* URL: https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/tree/main

Download Stable Diffusion V2.1 base for training from scratch.
* URL: https://huggingface.co/stabilityai/stable-diffusion-2-1-base/tree/main

### Convert checkpoints for initialization

If your would like to train from scratch, convert the downloaded checkpoints to control copy by running:
```bash
sh ./scripts/convert_weight.sh  
```
There will be a checkpoint at `./checkpoints/control_sd21_ini.ckpt`

### Prepare the datasets

* download datasets and revise `./configs/datasetsv2.yaml` for the path
* You could prepare you own datasets according to the formates of files in `./datasetsv2`.
* If you use UVO dataset, you need to process the json following `./datasetsv2/Preprocess/uvo_process.py`
* You could refer to `run_datasetv2_debug.py` to verify you data is correct.

### Start training

```bash
sh ./scripts/train.sh
```


## Inference
We provide inference code in `run_multidoor_inference.py` (from Line 222 - ) for both inference images provided by users and inference a dataset (VITON-HD Test). You should modify the data path and run the following code. The generated results are provided in `examples/TestDreamBooth/GEN` for single image, and `VITONGEN` for VITON-HD Test.

```bash
python run_multidoor_inference.py
```
<!-- The inferenced results on VITON-Test would be like [garment, ground truth, generation].

*Noticing that AnyDoor does not contain any specific design/tuning for tryon, we think it would be helpful to add skeleton infos or warped garment, and tune on tryon data to make it better :)*
  <table align="center">
    <tr>
    <td>
      <img src="assets/Figures/tryon.png">
    </td>
    </tr>
  </table>


Our evaluation data for DreamBooth an COCOEE coud be downloaded at Google Drive:
* URL: [to be released] -->


## Acknowledgements
This project is developped on the codebase of [AnyDoor](https://github.com/ali-vilab/AnyDoor). We  appreciate this great work! 


## Citation
If you find this codebase useful for your research, please use the following entry.
```BibTeX
```
