from LunarModules.ImageProcessor import ImageProcessor
from LunarModules.CustomDataLoader import CustomDataLoader
from LunarModules.Plotter import Plotter
from LunarModules.Model import *
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import torch
import cv2
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

def get_random_prediction(test_data_loader, device):
    print('random ...')
    running_iou = 0
    for step, batch in enumerate(test_data_loader):
        x_test, y_test = batch[0].to(device), batch[1].to(device)
        y_pred = torch.randint(low = 0, high = 2, size = y_test.shape)
        iou = JaccardIndex(num_classes = 4)(torch.argmax(y_pred.float(), dim = 1).cpu(), torch.argmax(y_test.float(), dim = 1).cpu())
        running_iou += iou
    print('random iou: ', (running_iou/len(test_data_loader)).numpy()+0)
    return

def get_real_stats(test_data_loader, device, DATA_PATH):
    CODE_PATH = os.getcwd()
    os.chdir('..')
    BASE_PATH = os.getcwd()
    os.chdir(CODE_PATH)
    all_models = []
    Closs = smp.utils.losses.CrossEntropyLoss()
    metrics = [
        smp_utils.metrics.IoU(threshold = 0.5),
    ]

    Unet = UNet_scratch(verbose = False).to(device)
    model = Model(Unet, loss = None, opt = None, scheduler = None, metrics = {'no':'no'}, random_seed = 42, train_data_loader = None, val_data_loader = None, test_data_loader = test_data_loader, real_test_data_loader = None, device = device, base_loc = BASE_PATH, name = f"Unet_scratch_ground", log_file=None)

    _ = model.load()
    all_models.append(model)

    backbone = 'vgg11_bn'
    encoder_weights = 'imagenet'
    activation = None
    Closs = smp.utils.losses.CrossEntropyLoss()
    pretrained_vgg = Pretrained_Model(backbone = backbone, train_data_loader = None, val_data_loader = None, test_data_loader = test_data_loader, real_test_data_loader = None, encoder_weights = encoder_weights, activation = activation, metrics = metrics, LR = 0.01, loss = Closs, device = device, base_loc = BASE_PATH, name = f'VGG11_BN_ground')
    _ = pretrained_vgg.load()
    all_models.append(pretrained_vgg)

    backbone = 'resnet18'
    pretrained_resnet = Pretrained_Model(backbone = backbone, train_data_loader = None, val_data_loader = None, test_data_loader = test_data_loader, real_test_data_loader = None, encoder_weights = encoder_weights, activation = activation, metrics = metrics, LR = 0.01, loss = Closs, device = device, base_loc = BASE_PATH, name = f'RESNET18_ground')
    _ = pretrained_resnet.load()
    all_models.append(pretrained_resnet)

    backbone = 'timm-mobilenetv3_large_100'
    pretrained_mobilenet = Pretrained_Model(backbone = backbone, train_data_loader = None, val_data_loader = None, test_data_loader = test_data_loader, real_test_data_loader = None, encoder_weights = encoder_weights, activation = activation, metrics = metrics, LR = 0.01, loss = Closs, device = device, base_loc = BASE_PATH, name = f'mobilenetv3_large_100_ground')
    _ = pretrained_mobilenet.load()
    all_models.append(pretrained_mobilenet)

    iou = JaccardIndex(num_classes = 4)
    running = {}
    for model in all_models:
        running[model.name] = 0

    data_path = os.path.join(DATA_PATH, 'images', 'real')
    imgs = os.listdir(os.path.join(data_path, 'real_img'))
    img_mask_processor = ImageProcessor()
    count = 0
    res = []
    with torch.no_grad():
        for img in imgs:
            img_loaded = plt.imread(data_path + '/real_img/' + img)
            img_loaded = cv2.resize(img_loaded, (256, 256))

            mask_loaded = plt.imread(data_path + '/real_mask/g_' + img)
            mask_loaded = cv2.resize(mask_loaded, (256, 256))

            img_loaded = img_mask_processor.preprocessor_images(img_loaded)
            mask_loaded = img_mask_processor.preprocessor_masks(mask_loaded)

            img_tensor = torch.from_numpy(img_loaded).float()
            mask_tensor = torch.from_numpy(mask_loaded).float()

            # Change ordering, channels first then img size
            x_test = img_tensor.permute(2, 0, 1).to(device)
            x_test = torch.unsqueeze(x_test, 0)
            y_test = mask_tensor.permute(2, 0, 1).to(device)
            y_test = torch.unsqueeze(y_test, 0)

            if x_test.shape[1] == 4:
                continue
            for model in all_models:
                model.model.eval()
                y_pred = model.model(x_test.float())
                if 'scratch' not in model.name:
                    y_pred = torch.softmax(y_pred, dim = 1)
                iou_score = iou(torch.argmax(y_pred.float(), dim = 1).cpu(), torch.argmax(y_test.float(), dim = 1).cpu())
                running[model.name] += iou_score
                res.append([model.name, count, 'real_test_iou', iou_score.numpy()+0])
            count += 1
    print(count, ' total images')
    # for model in all_models:
    #     final_iou = (running[model.name]/count).numpy()+0
    #     res.append([model.name, -1, 'real_test_iou', final_iou])
    df = pd.DataFrame(res, columns = ['model_name', 'epoch', 'metric', 'value'])
    df.to_csv(BASE_PATH + '/Results/real_data_results.csv')
    return