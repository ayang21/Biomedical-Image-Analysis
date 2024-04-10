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
