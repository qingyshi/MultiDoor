import os
import os.path as osp
import json
from torch.utils.data import Dataset
from PIL import Image



class CustomDataset(Dataset):
    def __init__(self, test_dataset_json, real_flag='txt', fake_flag='img', processor=None):
        with open(test_dataset_json, "r") as file:
            test_cases = json.load(file)["cases"]
        self.data = test_cases
        self.real_flag = real_flag
        self.fake_flag = fake_flag
        self.processor = processor  # Use the CLIP processor for all input transformations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        test_case = self.data[index]
        caption = test_case["caption"]
        fake_path = test_case["save_path"]

        if self.real_flag == 'txt':
            real_data = caption
            real_data = self.processor(text=real_data, return_tensors='pt', padding="max_length", max_length=77, truncation=True)['input_ids']
        else:
            raise ValueError

        if self.fake_flag == 'img':
            fake_data = Image.open(fake_path).convert('RGB')
            fake_data = self.processor(images=fake_data, return_tensors='pt')['pixel_values'].squeeze(0)
        else:
            raise ValueError

        return {'real': real_data, 'fake': fake_data}