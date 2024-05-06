import os
import os.path as osp
from torch.utils.data import Dataset
from PIL import Image



class CustomDataset(Dataset):
    def __init__(self, fake_path, real_flag='txt', fake_flag='img', processor=None):
        self.fake_folder = [osp.join(fake_path, f) for f in os.listdir(fake_path) if not f.startswith('.')]
        self.data = []
        for fake_folder in self.fake_folder:
            for image_name in os.listdir(fake_folder):
                self.data.append(osp.join(fake_folder, image_name))
        self.real_flag = real_flag
        self.fake_flag = fake_flag
        self.processor = processor  # Use the CLIP processor for all input transformations

    def __len__(self):
        return len(self.real_folder)

    def __getitem__(self, index):
        fake_path = self.data[index]
        caption = osp.basename(osp.dirname(fake_path)).replace("_", " ")

        if self.real_flag == 'txt':
            real_data = caption
            real_data = self.processor(text=real_data, return_tensors='pt', padding=True, truncation=True)['input_ids']
        else:
            raise ValueError

        if self.fake_flag == 'img':
            fake_data = Image.open(fake_path).convert('RGB')
            fake_data = self.processor(images=fake_data, return_tensors='pt')['pixel_values'].squeeze(0)
        else:
            raise ValueError

        return {'real': real_data, 'fake': fake_data}