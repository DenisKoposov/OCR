import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from model import OCR_model
from train import train, validate, load_model
from dataset import WordImageDataset

# cuda device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# create model instance
model = OCR_model().to(device)
# optimizer instance
# Reduce lr by a factor of 5 after 2 epochs without decrease of metrics
optimizer = optim.SGD(model.parameters(), lr=2e-3)
scheduler = optim.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2)
# Loss function
criterion = nn.CrossEntropyLoss()
# Train
curves = train(model, train_loader, val_loader, criterion, optimizer, 300,
               SAVE_BEST, SAVE_CKPT, val_each, print_each)