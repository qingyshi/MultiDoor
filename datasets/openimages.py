from .base import BaseDataset
import numpy as np
import cv2
import os


class OpenImagesDataset(BaseDataset):
    def __init__(self, root, image_dir, mask_dir, annotations, id2name):
        super().__init__()
        self.root = root
        self.image_dir = os.path.join(root, image_dir)
        self.mask_dir = os.path.join(root, mask_dir)
        
        annotations_path = os.path.join(root, annotations)
        with open(annotations_path, "r") as f:
            annos = f.readlines()[1:]
        id2name_path = os.path.join(root, id2name)
        with open(id2name_path, "r") as f:
            id2name = f.readlines()[1:]
            id2name = [idname.split(",")[:2] for idname in id2name]
            
        self.data = annos
        self.id2name = {}
        for idname in id2name:
            id = idname[0]
            if len(idname) == 2:
                self.id2name[id] = idname[1][:-1]
            else:
                name = " ".join(idname[1:])[:-1]
                self.id2name[id] = name
        self.dynamic = 1
    
    def __len__(self):
        return len(self.data)
    
    def sample_timestep(self, max_step=1000):
        step_start = 0
        step_end = max_step
        step = np.random.randint(step_start, step_end)
        return np.array([step])

    def get_sample(self, index):
        anno = self.data[index].split(",")[:-1]
        mask_path = anno[0]
        image_id = anno[1]
        label_id = anno[2]
        
        name = self.id2name[label_id]
        caption = f"This is a photo of {name}."
        names = [name]
        nouns = []
        for name in names:
            if len(nouns) > 0:
                start = caption.index(name, nouns[0]["end"])
            else:
                start = caption.index(name)
            end = start + len(name)
            noun = dict(word=name, start=start, end=end)
            nouns.append(noun)
        batch = self.process_nouns_in_caption(nouns, caption)
        
        image_path = os.path.join(self.image_dir, image_id + ".jpg")
        mask_path = os.path.join(self.mask_dir, mask_path)
        
        ref_image = cv2.imread(image_path)
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
        tar_image = ref_image.copy()
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) == 255
        
        item_with_collage = self.process_pairs(ref_image, [mask], tar_image, [mask.copy()])
        sampled_time_steps = self.sample_timestep()
        
        item_with_collage['time_steps'] = sampled_time_steps
        item_with_collage.update(batch)
        return item_with_collage