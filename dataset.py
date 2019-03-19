import os
import glob
from PIL import Image
import string
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

def extract_target(file_name):
    return file_name.split('_')[1]

class WordImageDataset(Dataset):
    """ Some documantation"""

    def __init__(self, root_dir, annotation_file, transform=None, max_samples=None, preload=False,
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
        self.max_len = max([len(t) for t in self.target]) + 1
        #self.target = [list(t.lower()) + ['<EOW>'] + ([-100] * (self.max_len - len(t) - 1)) for t in self.target]
        self.target = [list(t.lower()) + (['<EOW>'] * (self.max_len - len(t))) for t in self.target]
        self.preload = preload
        self.size = size
        self.vocab = ['<SOW>', '<EOW>'] + list(string.printable[:36])
        self.inv_vocab = {word: i for i, word in enumerate(self.vocab)}
        self.inv_vocab[-100] = -100
        # Store all the data in memory to speed up computations. HUGE MEMORY CONSUMPTION!!!
        if max_samples:
            idx = np.random.choice(len(self.path), max_samples, replace=False)
            self.path = [self.path[i] for i in idx]
            self.target = [self.target[i] for i in idx]
            
            if preload:
                self.images = [Image.open(os.path.join(root_dir, pth)).resize(self.size) for pth in self.path]


    def __getitem__(self, idx):
        if self.preload:
            img = self.images[idx]
        else:
            img = Image.open(os.path.join(self.root_dir, self.path[idx])).resize(self.size)
        
        y = torch.tensor([self.inv_vocab[ch] for ch in self.target[idx]], dtype=torch.long)
        
        if not self.transform is None:
            img = self.transform(img)

        return img, y

    
    def __len__(self):
        return len(self.path)


    def get_max_len(self):
        return self.max_len