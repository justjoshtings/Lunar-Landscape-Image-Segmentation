from LunarModules.ImageProcessor import ImageProcessor
from LunarModules.CustomDataLoader import CustomDataLoader
from LunarModules.Plotter import Plotter
from LunarModules.Model import *
from modeling import *
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW
from transformers import get_scheduler
import os
import gc
from tqdm.auto import tqdm
import time
from datetime import datetime
import datetime as dt
from torchmetrics import JaccardIndex
from torchmetrics import Dice
from torchvision import models
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smp_utils
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default = 'test', type = str, required = False)
    args = parser.parse_args()
    print('RUNNING WITH METHOD: ', args.method)

    TRAIN = False
    debug = False
    plot = False

    if args.method == 'test':
        TRAIN = False
        debug = False
        plot = True
    if args.method == 'train':
        TRAIN = True
        debug = False
        plot = True
    if args.method == 'debug':
        TRAIN = True
        debug = True
        plot = True

    # download data from google drive
    # download trained models from google drive
    # do EDA
    # do traintestsplit

    # Run Modeling and Evaluation
    RUN_MODEL_LOOP(TRAIN = TRAIN, debug = debug, plot = plot)
    print("EXITING")


