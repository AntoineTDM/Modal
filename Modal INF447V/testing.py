#test sur le jeu de données on garde que au-dessus de 90% 

import fonction_utiles as fu
import data_management as dm
import data_enhancement as de 
import model 
#import training

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
net=model.net
net.load_state_dict(torch.load('net.pt'))
net.eval()
treshold=0.99

max=0
count=0
for batch_idx, (data, target) in tqdm(enumerate(dm.augmented_train_dataloader)):
          net.eval()
          data, target = data.to(device), target.to(device)
          outputs = torch.argmax(net(data), dim=1)
          for i in range(target.size()[0]):
            if outputs[i]==target[i]:
              count+=1
          max+=1
print('train accuracy: ',count/max)

max=0
count=0
for batch_idx, (data, target) in tqdm(enumerate(dm.test_dataloader)):
          if max>30:
               break
          net.eval()
          data, target = data.to(device), target.to(device)
          res=net(data)
          output1=torch.argmax(res, dim=1)
          outputs = torch.max(res, dim=1)
          '''if max==1: 
               for i in range(target.size()[0]):
                    fu.show_tensor_images(data[i])
                    print(output1[i])
                    '''
          for i in range(target.size()[0]):
            if outputs[1][i].item()>=treshold:
              count+=1
          max+=1
print('test accuracy: (treshold: 0.99) ',count/max)