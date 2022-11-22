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
from ImageProcessor import ImageProcessor
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
        img_loaded = plt.imread(self.img_folder+'/'+images)
        img_loaded =  cv2.resize(img_loaded, (self.imsize, self.imsize))

        # Read corresponding mask from folder and resize
        mask_loaded = plt.imread(self.mask_folder+'/'+masks)
        mask_loaded = cv2.resize(mask_loaded, (self.imsize, self.imsize))

        #Add pre-processing steps
        img_mask_processor = ImageProcessor()
        img_loaded = img_mask_processor.preprocessor_images(img_loaded)
        mask_loaded = img_mask_processor.preprocessor_masks(mask_loaded)

        img_tensor = torch.from_numpy(img_loaded)
        mask_tensor = torch.from_numpy(mask_loaded)
        
        # returns as a tuple of tensors
        return img_tensor, mask_tensor

    # def data_generator(self, img_folder, mask_folder, batch_size, imsize, num_classes, first_n=None):
    #     """
    #     Function to create data generator object of images and masks.

    #     Parameters:
    #         img_folder: directory path to images folder
    #         mask_folder: directory path to masks folder
    #         batch_size: batch size to use 
    #         imsize: image height and width and n channels (image height, image width, n of channels)
    #         first_n: optional, set to some int to choose only first n data points

    #     Yields:
    #         (img, mask): tuple of image and mask in np arrays of shape (batch size, image height, image width, n of channels)
    #     """
    #     element_counter = 0 
        
    #     images_list = os.listdir(img_folder) #List of training images
    #     masks_list = os.listdir(mask_folder) #List of Mask images

    #     if first_n is None:
    #         images_list = sorted(images_list)
    #         masks_list= sorted(masks_list)
    #     else:
    #         images_list = sorted(images_list)[:first_n]
    #         masks_list= sorted(masks_list)[:first_n]
        
    #     while (True):
    #         channel_num = 3
    #         img = np.zeros((batch_size, imsize, imsize, channel_num)).astype('float')
    #         mask = np.zeros((batch_size, imsize, imsize, num_classes)).astype('int')

    #         for i in range(element_counter, element_counter+batch_size):

    #             # Read an image from folder and resize
    #             train_img = plt.imread(img_folder+'/'+images_list[i])
    #             train_img =  cv2.resize(train_img, (imsize, imsize))

    #             # Read corresponding mask from folder and resize
    #             train_mask = plt.imread(mask_folder+'/'+masks_list[i])
    #             train_mask = cv2.resize(train_mask, (imsize, imsize))

    #             #Add pre-processing steps
    #             train_img = ImageProcessor.preprocessor_images(train_img)
    #             train_mask = ImageProcessor.preprocessor_masks(train_mask)
                
    #             #add to array - img[0], img[1], and so on to created np framework
    #             mask[i-element_counter] = train_mask
    #             img[i-element_counter] = train_img 
                
    #         element_counter+=batch_size

    #         # If we've reached the end of the batch, set element_counter back to 0 to start next batch
    #         if(element_counter+batch_size>=len(images_list)):
    #             element_counter=0
            
    #         yield (img, mask)