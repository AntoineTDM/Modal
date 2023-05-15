import fonction_utiles as fu
import data_management as dm
import data_enhancement as de 
import model 


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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#défintion de l'entrainement 
learning_rate = 1e-3
net=model.net
optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)
criterion=nn.CrossEntropyLoss()

fu.train(net, device,optimizer, dm.augmented_train_dataloader, criterion)

torch.save(net.state_dict(), 'net.pt')