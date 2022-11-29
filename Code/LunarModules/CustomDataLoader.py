"""
CustomDataLoader.py
Object to handle data generators.

author: @saharae, @justjoshtings
created: 11/12/2022
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import cv2
import copy
from LunarModules.ImageProcessor import ImageProcessor
from LunarModules.Plotter import Plotter
import torch, gc
from torch.utils.data import Dataset, DataLoader

# WILL NEED TO CLEAN THIS WHOLE MESS UP LATER!!!

class CustomDataLoader:
    '''
    Object to handle data generator.
    '''
    def __init__(self, img_folder, mask_folder, batch_size, imsize, num_classes, first_n=None, log_file=None):
        '''
        Params:
            self: instance of object
            img_folder: directory path to images folder
            mask_folder: directory path to masks folder
            batch_size: batch size to use 
            imsize: image height and width and n channels (image height, image width, n of channels)
            first_n: optional, set to some int to choose only first n data points
            log_file (str): default is None to not have logging, otherwise, specify logging path ../filepath/log.log
        '''
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        # We'll do actual batch size in the dataloader not here, here we set to 1 
        self.batch_size = 1
        self.imsize = imsize
        self.num_classes = num_classes
        self.first_n = first_n
        self.log_file = log_file

        self.element_counter = 0 

        self.images_list = os.listdir(self.img_folder) #List of training images
        self.masks_list = os.listdir(self.mask_folder) #List of Mask images

        if self.first_n is None:
            self.images_list = sorted(self.images_list)
            self.masks_list= sorted(self.masks_list)
        else:
            self.images_list = sorted(self.images_list)[:self.first_n]
            self.masks_list= sorted(self.masks_list)[:self.first_n]

    def __len__(self):
        '''
        Params:
            self: instance of object
        Returns:
            number of images
        '''
        return len(self.images_list)

    def __getitem__(self, idx):
        '''
		Params:
			self: instance of object
			idx (int): index of iteration
		Returns:
			input_ids (pt tensors): encoded text as tensors
			attn_masks (pt tensors): attention masks as tensors
		'''
        images = self.images_list[idx]
        masks = self.masks_list[idx]

        # does preprocessing

        # Read an image from folder and resize
        # plt.imread() loads as (imsize, imsize, channels)
        # Original image and mask seems to be 480x720x3
        img_loaded = plt.imread(self.img_folder+'/'+images)
        img_loaded =  cv2.resize(img_loaded, (self.imsize, self.imsize))

        # Read corresponding mask from folder and resize
        mask_loaded = plt.imread(self.mask_folder+'/'+masks)
        mask_loaded = cv2.resize(mask_loaded, (self.imsize, self.imsize))


        # plt.imread() Loads as value between 0 and 1, cv2.imread() loads values between 0-255
        # test= cv2.imread(self.img_folder+'/'+images)
        # test_mask = cv2.imread(self.mask_folder+'/'+masks)
        # print('Image Max/Min Val: ', np.max(img_loaded), np.min(img_loaded))
        # print('Mask Max/Min Val: ', np.max(mask_loaded), np.min(mask_loaded))
        # print('Image CV2 Max/Min Val: ', np.max(test), np.min(test))
        # print('Mask CV2 Max/Min Val: ', np.max(test_mask), np.min(test_mask))
        # print(test.shape, test_mask.shape)

        img_mask_processor = ImageProcessor()
        # check_plotter = Plotter()

        #Data Augmentation steps
        img_loaded, mask_loaded = img_mask_processor.data_augmentation(img_loaded, mask_loaded)

        # check_plotter.peek_images(sample_images=img_loaded,sample_masks=mask_loaded,file_name='current_test_2.png')

        #Pre-processing steps
        img_loaded = img_mask_processor.preprocessor_images(img_loaded)
        mask_loaded = img_mask_processor.preprocessor_masks(mask_loaded)

        # Check and save images as plots
        # sample_mask = img_mask_processor.rescale(mask_loaded)
        # Reverse one hot encode predicted mask
        # sample_mask_decoded = img_mask_processor.reverse_one_hot_encode(sample_mask)
        # sample_mask_decoded = img_mask_processor.rescale(sample_mask_decoded)
        # check_plotter.peek_images(sample_images=img_loaded,sample_masks=sample_mask_decoded,file_name='current_test.png')

        img_tensor = torch.from_numpy(img_loaded)
        mask_tensor = torch.from_numpy(mask_loaded)

        # Change ordering, channels first then img size
        img_tensor = img_tensor.permute(2, 0, 1)
        mask_tensor = mask_tensor.permute(2, 0, 1)

        plt.close('all')
        # returns as a tuple of tensors
        return img_tensor, mask_tensor
