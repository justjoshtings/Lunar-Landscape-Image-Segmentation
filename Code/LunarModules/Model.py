"""
Model.py
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
# optimizer = adam?
# loss = pixel wise cross entropy, jacard something
# metric = mean IOU
# pretrained models = unet, resnet backbone with upsampling, ... lots of others

class Model:
    '''
    Object to handle model and related methods.
    '''
    def __init__(self, log_file=None):
        '''
        Params:
            self: instance of object
            log_file (str): default is None to not have logging, otherwise, specify logging path ../filepath/log.log

        '''