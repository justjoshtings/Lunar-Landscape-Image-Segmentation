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

train_img_folder = DATA_PATH + '/images/render/'
mask_img_folder = DATA_PATH + '/images/clean/'
batch_size = 8
imsize = 256
num_classes = 4

train_data = CustomDataLoader(img_folder=train_img_folder, mask_folder=mask_img_folder, batch_size=batch_size, imsize=imsize, num_classes=num_classes)
train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

print(next(iter(train_data)))
print(next(iter(train_data_loader)))