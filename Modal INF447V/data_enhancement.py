import os
from PIL import Image
import torch
from torch import nn
from tqdm.notebook import tqdm
from torchvision import transforms,datasets
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader,ConcatDataset,Dataset
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np 

class CustomDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.dataset[index]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.dataset)

transformations=[]
#vu les couleurs du dataset, modifiés les couleurs n'aura pas énormément d'impact

#première transformation possible
transform = transforms.Compose([
   transforms.RandomHorizontalFlip(p=0.5), 
   transforms.RandomAffine(degrees=10, translate=(0.1,0.1), scale=(0.9,1.1)), 
   transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), 
   transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
])
transformations.append(transform)


#deuxième transformation possible
transform = transforms.Compose([
   transforms.RandomHorizontalFlip(p=0.5), 
   transforms.RandomRotation(degrees=20), 
   transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
   transforms.RandomResizedCrop(size=(224, 224), scale=(0.7, 1.0), ratio=(0.9, 1.1), interpolation=2)
])
transformations.append(transform)


#troisième transformation possible
transform = transforms.Compose([
   transforms.RandomHorizontalFlip(p=0.5), 
   transforms.RandomRotation(degrees=10), 
   transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), 
   transforms.RandomAffine(degrees=10, translate=(0.1,0.1), scale=(0.9,1.1))
])
transformations.append(transform)


#quatrieme transformation possible
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15), 
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5)
])
transformations.append(transform)


#cinquième transformation possible
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5), 
    transforms.RandomRotation(degrees=5), 
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05), 
    transforms.RandomErasing(p=0.5, scale=(0.02,0.1), ratio=(0.3, 3.3))
])
transformations.append(transform)

def augmenter_data(dataloader_,transformations,batch_size=64):
  resultat=dataloader_
  for transform in transformations:
      transformed_dataset = CustomDataset(dataloader_.dataset, transform) 
      # On concatène le résultat 
      concatenated_dataset = ConcatDataset([resultat.dataset, transformed_dataset])
      resultat = DataLoader(concatenated_dataset, batch_size=batch_size, shuffle=True)
  return resultat
