#entrainement avec la classe

import data_enhancement as de
import fonction_utiles as fu

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


#import of data

# Define the transform for the input images
transform = transforms.Compose([transforms.ToTensor(),# convert image to tensor
    transforms.Resize((224, 224))   # resize image to 400x400
    ])

#for the data to train with
# Create a dataset object from the folder containing the images
folder_path = '/Users/s.radziszewski/Desktop/Modal_INF447V/dataset_train/train'
# Create a list of all image file paths in the folder
image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
# Create a dataset object from the image file paths
labeled_dataset = ImageFolder(folder_path, transform=transform, loader=lambda x: Image.open(x))
# Load the dataset into a dataloader for batching and shuffling
training_dataloader = torch.utils.data.DataLoader(labeled_dataset, batch_size=64, shuffle=True)

#for the data to explore
# Create a dataset object from the folder containing the images
folder_path = '/Users/s.radziszewski/Desktop/Modal_INF447V/dataset_train/to_explore'
# Create a list of all image file paths in the folder
image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
# Create a dataset object from the image file paths
explore_dataset = ImageFolder(folder_path, transform=transform, loader=lambda x: Image.open(x))
# Load the dataset into a dataloader for batching and shuffling 
test_dataloader = torch.utils.data.DataLoader(explore_dataset, batch_size=64, shuffle=True)


#augmenation des datas 
augmented_train_dataloader=de.augmenter_data(training_dataloader,de.transformations)

#test de vérification
'''
print("test des dimensions init")
print(len(training_dataloader))#nb batch
images,labels= next(iter(training_dataloader))
print(images.size())#(size_batch,channel,h,w)
print(labels.size())#(size_batch)

print("test des dimensions augmenter")
print(len(augmented_train_dataloader))#nb batch
images,labels= next(iter(augmented_train_dataloader))
print(images.size())#(size_batch,channel,h,w)
print(labels.size())#(size_batch)'''