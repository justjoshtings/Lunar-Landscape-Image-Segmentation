"""
DataLoader.py
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

# WILL NEED TO CLEAN THIS WHOLE MESS UP LATER!!!

class DataLoader:
    '''
    Object to handle data generator.
    '''
    def __init__(self, log_file=None):
        '''
        Params:
            self: instance of object
            log_file (str): default is None to not have logging, otherwise, specify logging path ../filepath/log.log

        '''
    def data_generator(img_folder, mask_folder, batch_size, imsize, num_classes, first_n=None):
        """
        Function to create data generator object of images and masks.

        Parameters:
            img_folder: directory path to images folder
            mask_folder: directory path to masks folder
            batch_size: batch size to use 
            imsize: image height and width and n channels (image height, image width, n of channels)
            first_n: optional, set to some int to choose only first n data points

        Yields:
            (img, mask): tuple of image and mask in np arrays of shape (batch size, image height, image width, n of channels)
        """
        element_counter = 0 
        
        images_list = os.listdir(img_folder) #List of training images
        masks_list = os.listdir(mask_folder) #List of Mask images

        if first_n is None:
            images_list = sorted(images_list)
            masks_list= sorted(masks_list)
        else:
            images_list = sorted(images_list)[:first_n]
            masks_list= sorted(masks_list)[:first_n]
        
        while (True):
            channel_num = 3
            img = np.zeros((batch_size, imsize, imsize, channel_num)).astype('float')
            mask = np.zeros((batch_size, imsize, imsize, num_classes)).astype('int')

            for i in range(element_counter, element_counter+batch_size):

                # Read an image from folder and resize
                train_img = plt.imread(img_folder+'/'+images_list[i])
                train_img =  cv2.resize(train_img, (imsize, imsize))

                # Read corresponding mask from folder and resize
                train_mask = plt.imread(mask_folder+'/'+masks_list[i])
                train_mask = cv2.resize(train_mask, (imsize, imsize))

                #Add pre-processing steps
                train_img = preprocessor_images(train_img)
                train_mask = preprocessor_masks(train_mask)
                
                #add to array - img[0], img[1], and so on to created np framework
                mask[i-element_counter] = train_mask
                img[i-element_counter] = train_img 
                
            element_counter+=batch_size

            # If we've reached the end of the batch, set element_counter back to 0 to start next batch
            if(element_counter+batch_size>=len(images_list)):
                element_counter=0
            
            yield (img, mask)