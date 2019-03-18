import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from model import OCR_model
from train import train, validate, load_model
from dataset import WordImageDataset

# cuda device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# create model instance
model = OCR_model().to(device)

# optimizer instance
# Reduce learning rate by a factor of 5 after 2 epochs without decrease of error
optimizer = optim.SGD(model.parameters(), lr=2e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2)

# Loss function
criterion = nn.CrossEntropyLoss()

# Load data
train_transform = transforms.Compose([
    transforms.Grayscale(),
    #transforms.RandomVerticalFlip(),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
    ])

test_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
    ])

bs = 32
n_workers = 6

train_dataset = WordImageDataset("../synth90k", "annotation_train.txt", train_transform)
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=n_workers)

val_dataset = WordImageDataset("../synth90k", "annotation_val.txt", test_transform)
val_loader = DataLoader(val_dataset, batch_size=bs, num_workers=n_workers)

test_dataset = WordImageDataset("../synth90k", "annotation_test.txt", test_transform)
test_loader = DataLoader(test_dataset, batch_size=bs, num_workers=n_workers)

# Train
SAVE_BEST = "best_model.pt"
SAVE_CKPT = "last_model.pt"
curves = train(model, train_loader, val_loader, criterion, optimizer, 30,
               SAVE_BEST, SAVE_CKPT, device, val_each=1, print_each=1)