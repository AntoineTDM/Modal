#ce fichier comporte toutes les fonctions classiques nécessaires 

import os
from PIL import Image
import torch
from torch import nn
from tqdm.notebook import tqdm
from torchvision import transforms,datasets
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader,ConcatDataset
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np 

#with PIL images
def show_images(images):
    # Create a grid of images with 2 rows and 5 columns
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 4))

    # Flatten the axes array so we can easily iterate over it
    axes = axes.ravel()

    # Loop over the images in the tensor
    for i in np.arange(0, images.shape[0]):
        # Get the i-th image from the tensor
        image = images[i]

        # Transpose the dimensions of the image tensor to match the expected format
        image = np.transpose(image, (1, 2, 0))

        # Rescale the pixel values to be between 0 and 1
        image = (image - image.min()) / (image.max() - image.min())

        # Display the image on the corresponding axis
        axes[i].imshow(image)
        axes[i].axis('off')

    # Adjust the spacing between the images and the layout of the subplots
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.tight_layout()
    plt.show()

#now with tensors
def show_tensor_images(image_tensor, num_images=8, size=(3, 224, 224)):
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

def train(model,device, optimizer, train_loader, criterion,  n_epoch = 5,cuda=True):
  #train_loader=train_loader.to(device)
  model=model.to(device)
  total = 0
  for epoch in tqdm(range(n_epoch)):  # loop over the dataset multiple times
      model.train()
      running_loss = 0.0
      running_acc = 0.0
      for batch_idx, (data, target) in enumerate(train_loader):
          data, target = data.to(device), target.to(device)
          
          # zero the parameter gradients
          optimizer.zero_grad()
          output = model(data)
          loss = criterion(output, target)
          loss.backward()
          optimizer.step()       
      
  print('Finished Training')

