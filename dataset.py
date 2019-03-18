import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

def extract_target(file_name):
    return file_name.split('_')[1]

class WordImageDataset(Dataset):
    """ Some documantation"""

    def __init__(self, root_dir, annotation_file, transform=None, preload=None,
                 size=(32, 100)):
        """
        Args:
        """
        self.root_dir = root_dir
        self.path = []
        self.transform = transform

        with open(os.path.join(root_dir, annotation_file), "r") as f:
            for line in f:
                self.path.append(line.split(" ")[0])
        
        self.target = [extract_target(pth) for pth in self.path]
        self.preload = preload
        self.size = size
        # Store all the data in memory to speed up computations. HUGE MEMORY CONSUMPTION!!!
        if preload:
            idx = np.random.choice(len(self.path), preload, replace=False)
            self.images = [Image.open(pth).resize(self.size) for pth in self.path[idx]]
            self.target = self.target[idx]


    def __getitem__(self, idx):

        if self.preload:
            img = self.images[idx]
        else:
            img = Image.open(self.path[idx]).resize(self.size)
        
        y = list(self.targets[idx])
        
        if not self.transform is None:
            img = self.transform(img)

        return img, y

    
    def __len__(self):
        if self.preload:
            return len(self.images)
        else:
            return len(self.path)


if __name__ == "__main__":
    root_dir = "../synth90k"
    dataset = WordImageDataset(root_dir, "annotation_test.txt")
    print(len(dataset))