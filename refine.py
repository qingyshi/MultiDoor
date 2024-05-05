from diffusers import StableDiffusionXLImg2ImgPipeline
import torch
from PIL import Image
import os

# load both base & refiner
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "/data00/sqy/checkpoints/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe = pipe.to("cuda:4")



img_name = str(int(0))+'.png'
sample = Image.open("examples/GEN/cat_surfboard/1/gen.jpg").convert("RGB")
strength, steps = 0.35, 20

prompt = "A cat is surfing on the white surfboard."
refined_image = pipe(prompt, image=sample, strength=strength, num_inference_steps=steps).images[0]
refined_image.save(os.path.join(img_name.replace('.png', '_xl_s{}_n{}.png'.format(strength, steps))))
sample.save(os.path.join(img_name))