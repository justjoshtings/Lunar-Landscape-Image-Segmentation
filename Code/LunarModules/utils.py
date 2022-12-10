from LunarModules.ImageProcessor import ImageProcessor
from LunarModules.CustomDataLoader import CustomDataLoader
from LunarModules.Plotter import Plotter
from LunarModules.Model import *
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import torch
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

'''
Utility Functions
'''
def test(loader):
    for step, batch in enumerate(loader):
        print("testing loader")
        if step == 0:
            x_test, y_test = batch[0], batch[1]

            y_test_reorder = y_test.permute(0, 2, 3, 1)
            x_test_reorder = x_test.permute(0, 2, 3, 1)

            img = x_test_reorder.cpu().detach().numpy()[10]
            img_processor = ImageProcessor()

            predicted_image_decoded_mask = img_processor.reverse_one_hot_encode(y_test_reorder.cpu().detach().numpy()[10])
            fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (8, 6))

            axes[0].imshow(predicted_image_decoded_mask)
            axes[0].set_title(f'clean_mask')
            axes[1].imshow(img)
            axes[1].set_title(f'clean_actual')
            fig.savefig(f'clean_testing_loader.png')
            print('done')
            return

def do_preprocessing_checks(train_data, train_data_loader, train_img_folder, train_mask_folder, real_test_img_folder, real_test_mask_folder):
    # Check endcoding of images in each step

    # tuples of 2 tensor arrays
    # Image element is tensor array of size [RGB channel, imsize, imsize]
    # Mask element is tensor array of size [class channel, imsize, imsize]
    sample_data = next(iter(train_data))

    # list of len 2 (1 for image, 1 for mask).
    # Image element in list is torch tensor of size [batch size, RGB channel, imsize, imsize]
    # Mask element in list is torch tensor of size [batch size, class channel, imsize, imsize]
    batch_data = next(iter(train_data_loader))

    img_processor = ImageProcessor()

    print(sample_data[0].shape)
    print(sample_data[1].shape)

    print(batch_data[0].shape)
    print(batch_data[1].shape)

    # Reorder images for plotting
    sample_image = sample_data[0].permute(1,2,0)
    sample_mask = sample_data[1].permute(1,2,0)
    sample_image = sample_image.numpy()
    sample_mask = sample_mask.numpy()

    sample_mask = img_processor.rescale(sample_mask)
    # Reverse one hot encode predicted mask
    sample_mask_decoded = img_processor.reverse_one_hot_encode(sample_mask)
    sample_mask_decoded = img_processor.rescale(sample_mask_decoded)

    # Plot images to ensure correct processing steps
    check_plotter = Plotter()
    check_plotter.peek_images(sample_images=sample_image,sample_masks=sample_mask_decoded,file_name='current_test.png')
    check_plotter.sanity_check(train_img_folder+'/' , train_mask_folder+'/')

    # Plot real images
    check_plotter.sanity_check(real_test_img_folder+'/' , real_test_mask_folder+'/')

    print(sample_mask_decoded.shape)

def update_results(model, RESULTS, RESULT_PATH):
    for metric in model.history.keys():
        for epoch, val in model.history[metric]:
            RESULTS.append([model.name, epoch, metric, val])

    if os.path.exists(os.path.join(RESULT_PATH, 'RESULTS.csv')):
        current_results = pd.read_csv(os.path.join(RESULT_PATH, 'RESULTS.csv'))
    else:
        current_results = pd.DataFrame(columns = ['model_name', 'epoch', 'metric', 'value'])
    new_res = pd.DataFrame(RESULTS, columns = ['model_name', 'epoch', 'metric', 'value'])
    to_save = pd.concat([current_results, new_res])
    to_save.to_csv(os.path.join(RESULT_PATH, 'RESULTS.csv'), index = False)
    print("results updated")
    return RESULTS

def plot_prediction(model, test_data_loader, device):
    print('plotting...')
    for step, batch in enumerate(test_data_loader):
        if step == 0:
            x_test, y_test = batch[0], batch[1]
            y_pred = model.model(x_test.to(device))
            # print(f'TEST: {y_test.shape}', y_test)
            # print(f'PRED: {y_pred.shape}', y_pred)
            np.save('y_test_batch.npy', y_test.cpu().detach().numpy())
            np.save('y_pred_batch.npy', y_pred.cpu().detach().numpy())

            if 'scratch' not in model.name:
                y_pred_OHE = torch.softmax(y_pred, dim = 1)
            else:
                y_pred_OHE = y_pred
            #print(f'PRED OHE: {y_pred_OHE.shape}', y_pred_OHE)

            y_pred_reorder = y_pred_OHE.permute(0, 2, 3, 1)
            y_test_reorder = y_test.permute(0, 2, 3, 1)
            x_test_reorder = x_test.permute(0, 2, 3, 1)

            img = x_test_reorder.cpu().detach().numpy()[10]
            img_processor = ImageProcessor()

            argmaxed_pred = img_processor.mask_argmax(y_pred_reorder.cpu().detach().numpy()[10])
            argmaxed_test = y_test_reorder.cpu().detach().numpy()[10]

            predicted_image_decoded = img_processor.reverse_one_hot_encode(argmaxed_pred)
            predicted_image_decoded_mask = img_processor.reverse_one_hot_encode(argmaxed_test)

            fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (10, 8))

            axes[0].imshow(predicted_image_decoded)
            axes[0].set_title(f'{model.name} - predicted')
            axes[1].imshow(predicted_image_decoded_mask)
            axes[1].set_title(f'{model.name} - mask')
            axes[2].imshow(img)
            axes[2].set_title(f'{model.name} - actual')
            fig.savefig(f'prayers_{model.name}.png')
            print('done')
            return
