import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

def extract_target(file_name):
    return file_name.split('_')[1]

class WordImageDataset(Dataset):
    """ Some documantation"""

    def __init__(self, root_dir, annotation_file, transform=None, preload=None,
                 size=(124, 68)):
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
            self.path = [self.path[i] for i in idx]
            self.target = [self.target[i] for i in idx]
            self.images = [Image.open(os.path.join(root_dir, pth)).resize(self.size) for pth in self.path]


    def __getitem__(self, idx):

        if self.preload:
            img = self.images[idx]
        else:
            img = Image.open(os.path.join(self.root_dir, self.path[idx])).resize(self.size)
        
        y = list(self.target[idx])
        
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
    data = dataset.__getitem__(150)
    
    print(data[0], data[1])
    plt.imshow(data[0])
    plt.show()