import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torchvision import datasets,transforms 
from torch.utils.data.sampler import SubsetRandomSampler

import dataset_tools as dtools

dtools.download(dataset='TBX11K', dst_dir='dataset-ninja/')

file_ext = "png"
clf_result = "normal"

base_path = 'dataset-ninja/tbx11k/'
normal_destination_path = 'drive/MyDrive/tb-detection/data-normal/'
tb_destination_path = 'drive/MyDrive/tb-detection/data-tb/'

test_ann_path = base_path + 'test/ann/'
test_img_path = base_path + 'test/img/'

train_ann_path = base_path + 'train/ann/'
train_img_path = base_path + 'train/img/'

val_ann_path = base_path + 'val/ann/'
val_img_path = base_path + 'val/img/'

ann_directory = [test_ann_path, train_ann_path, val_ann_path]
img_directory = [test_img_path, train_img_path, val_img_path]

# Collect data

for i in range(len(ann_directory)):
    for file in os.listdir(ann_directory[i]):
        imgname = file[:-len(file_ext)] + file_ext
        with open(os.path.join(ann_directory[i],file)) as report:
            for line in report:
                if clf_result in line:
                    shutil.copy(img_directory[i] + imgname, normal_destination_path + imgname)
                    break
            else:
                shutil.copy(img_directory[i] + imgname, tb_destination_path + imgname)

#Load and preprocess data
#Splitting the dataset into train and test using the SubsetSampler class
#To apply two different sets of transforms to train and test, I'm loading the dataset twice.
#I've applied resize so as to make the images smaller and thus model training faster. Models trained on smaller images tend to not overfit and generalize well for larger images.
#I've also added a random crop and random flip to augment the dataset further.

#The mean and standard deviation of the ImageNet data Alexnet was trained on.
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225] 
test_size = 0.30
random_seed = 24
num_workers = 0
batch_size = 8
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean,
                         std=std)
])
test_transform = transforms.Compose([
    transforms.Resize(256),    
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean,
                         std=std)
])
train_dataset = datasets.ImageFolder(root=dataset_dir,
                                  transform=train_transform)

test_dataset = datasets.ImageFolder(root=dataset_dir,
                                  transform=test_transform)
dataset_size = len(train_dataset)
indices = list(range(dataset_size))
split = int(np.floor(test_size * dataset_size))
np.random.seed(random_seed)
np.random.shuffle(indices)
train_idx, test_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
test_sampler = SubsetRandomSampler(test_idx)

train_loader = torch.utils.data.DataLoader(train_dataset, 
                                           batch_size=batch_size, 
                                           sampler=train_sampler,
                                           num_workers=num_workers)

test_loader = torch.utils.data.DataLoader(test_dataset, 
                                          batch_size=batch_size, 
                                          sampler=test_sampler,
                                          num_workers=num_workers)
dataloaders = {
    'train': train_loader,
    'test': test_loader
}
