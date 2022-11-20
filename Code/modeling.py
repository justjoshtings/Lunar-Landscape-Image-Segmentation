"""
modeling.py
Script to do modeling

author: @saharae, @justjoshtings
created: 11/17/2022
"""

from LunarModules.ImageProcessor import ImageProcessor
from LunarModules.CustomDataLoader import CustomDataLoader
from LunarModules.Model import Model
from torch.utils.data import Dataset, DataLoader
import os

CODE_PATH = os.getcwd()
os.chdir('..')
BASE_PATH = os.getcwd()
os.chdir(CODE_PATH)
DATA_PATH = os.path.join(BASE_PATH, 'Data')

# Will need to split into train/test/val later
train_img_folder = DATA_PATH + '/images/render/'
mask_img_folder = DATA_PATH + '/images/clean/'
batch_size = 8
imsize = 256
num_classes = 4

'''
Create dataloader for train, validation, and testing dataset
'''
train_data = CustomDataLoader(img_folder=train_img_folder, mask_folder=mask_img_folder, batch_size=batch_size, imsize=imsize, num_classes=num_classes)
train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# tuples of 2 tensor arrays
# Image element is tensor array of size [imsize, imsize, RGB channel]
# Mask element is tensor array of size [imsize, imsize, class channel]
sample_data = next(iter(train_data))

# list of len 2 (1 for image, 1 for mask). 
# Image element in list is torch tensor of size [batch size, imsize, imsize, RGB channel]
# Mask element in list is torch tensor of size [batch size, imsize, imsize, class channel]
batch_data = next(iter(train_data_loader))

print(sample_data[0].shape)
print(sample_data[1].shape)

print(batch_data[0].shape)
print(batch_data[1].shape)

# NEED TO DO AN IMAGE PLOT CHECK TO SEE IF EVERYTHING LOOKS GOOD OUT OF CUSTOM DATALOADER

'''
Load Model(s)
'''


'''
Train
'''

'''
Evaluate Model(s)
'''