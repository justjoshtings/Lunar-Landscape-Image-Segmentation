"""
modeling.py
Script to do modeling

author: @saharae, @justjoshtings
created: 11/17/2022
"""
import pandas as pd

from LunarModules.ImageProcessor import ImageProcessor
from LunarModules.CustomDataLoader import CustomDataLoader
from LunarModules.Plotter import Plotter
from LunarModules.Model import *
from LunarModules.utils import *
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


def RUN_MODEL_LOOP(TRAIN = True, debug = False, plot = True, data_source = 'ground'):
    '''
    Main loop to run modeling code -- called from main function or from command line
    :param TRAIN: if True the loop will run the full training code
    :param debug:  if True the utility checking functions will be run in the begining before training
    :param plot:  if True the predictions will be plotted from the models as well as the training curves
    :return:
    '''

    '''
    Set paths
    '''
    CODE_PATH = os.getcwd()
    os.chdir('..')
    BASE_PATH = os.getcwd()
    os.chdir(CODE_PATH)
    DATA_PATH = os.path.join(BASE_PATH, 'Data')

    RESULT_PATH = os.path.join(BASE_PATH, 'Results')

    if not os.path.exists(RESULT_PATH):
        os.mkdir(RESULT_PATH)

    '''
    SET UP device
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device..', device)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    gc.collect()
    torch.cuda.empty_cache()

    '''
    Set parameters
    '''
    train_img_folder = DATA_PATH + '/images/train/render'
    train_mask_folder = DATA_PATH + '/images/train/mask'
    val_img_folder = DATA_PATH + '/images/val/render'
    val_mask_folder = DATA_PATH + '/images/val/mask'
    test_img_folder = DATA_PATH + '/images/test/render'
    test_mask_folder = DATA_PATH + '/images/test/mask'
    real_test_img_folder = DATA_PATH + '/images/real/real_img'
    real_test_mask_folder = DATA_PATH + '/images/real/real_mask'

    batch_size = 32
    imsize = 256
    num_classes = 4
    all_models = []

    '''
    Create dataloader for train, validation, and testing dataset
    '''
    train_data = CustomDataLoader(img_folder=train_img_folder, mask_folder=train_mask_folder, batch_size=batch_size, imsize=imsize, num_classes=num_classes, split='train', augmentation=True)
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    val_data = CustomDataLoader(img_folder=val_img_folder, mask_folder=val_mask_folder, batch_size=batch_size, imsize=imsize, num_classes=num_classes, split='validation', augmentation=False)
    val_data_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    test_data = CustomDataLoader(img_folder=test_img_folder, mask_folder=test_mask_folder, batch_size=batch_size, imsize=imsize, num_classes=num_classes, split='test', augmentation=False)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    real_test_data = CustomDataLoader(img_folder=real_test_img_folder, mask_folder=real_test_mask_folder, batch_size=batch_size, imsize=imsize, num_classes=num_classes, split='test', augmentation=False)
    real_test_data_loader = DataLoader(real_test_data, batch_size=batch_size, shuffle=True)

    '''
    Review and Check Preprocessing and DataLoader outputs are correctly performed
    '''
    if debug:
        do_preprocessing_checks(train_data, train_data_loader, train_img_folder, train_mask_folder, real_test_img_folder, real_test_mask_folder)
        test(test_data_loader)

    '''
    SET hyperparams
    '''
    n_epochs = 20
    LR = 0.001

    metrics = {
        "Dice": Dice(num_classes = 4),
        "IOU": JaccardIndex(num_classes = 4)
    }

    lossBCE = torch.nn.BCEWithLogitsLoss()
    num_training_steps = n_epochs * len(train_data_loader)

    total_t0 = time.time()

    # [model name, epoch, metric, value]
    RESULTS = []

    '''
    UNET SCRATCH
    '''
    Unet = UNet_scratch(verbose = False).to(device)
    opt = AdamW(Unet.parameters(), lr = LR)
    lr_scheduler = get_scheduler(name="linear", optimizer=opt, num_warmup_steps=0, num_training_steps=num_training_steps)
    model = Model(Unet, loss = lossBCE, opt = opt, scheduler = lr_scheduler, metrics = metrics, random_seed = 42, train_data_loader = train_data_loader, val_data_loader = val_data_loader, test_data_loader = test_data_loader, real_test_data_loader = real_test_data_loader, device = device, base_loc = BASE_PATH, name = f"Unet_scratch_{data_source}", log_file=None)

    if TRAIN:
        print('Training ', num_training_steps, 'steps!!')
        model.run_training(n_epochs = n_epochs, save_on = 'val_IOU', load = True)

    '''
    Evaluate model
    '''
    _ = model.load() # always load latest model
    model.run_test()
    _ = update_results(model, RESULTS, RESULT_PATH)

    if debug:
        plot_prediction(model, test_data_loader, device)
    all_models.append(model)


    '''
    PRETRAINED VGG
    '''
    RESULTS = []

    backbone = 'vgg11_bn'
    encoder_weights = 'imagenet'
    activation = None

    loss = smp.utils.losses.BCEWithLogitsLoss()
    metrics = [
        smp_utils.metrics.IoU(threshold=0.5)
    ]
    pretrained_vgg = Pretrained_Model(backbone = backbone, train_data_loader = train_data_loader, val_data_loader = val_data_loader, test_data_loader = test_data_loader, encoder_weights = encoder_weights, activation = activation, metrics = metrics, LR = LR, loss = loss, device = device, base_loc = BASE_PATH, name = f'VGG11_BN_{data_source}')

    if TRAIN:
        pretrained_vgg.run_training(n_epochs)

    '''
    Evaluate pretrained model
    '''
    _ = pretrained_vgg.load() # always load the best model
    pretrained_vgg.run_testing()
    _ = update_results(pretrained_vgg, RESULTS, RESULT_PATH)

    if debug:
        plot_prediction(pretrained_vgg, test_data_loader, device)
    all_models.append(pretrained_vgg)

    '''
    PRETRAINED RESNET
    '''

    RESULTS = []

    backbone = 'resnet18'
    encoder_weights = 'imagenet'
    activation = None

    loss = smp.utils.losses.BCEWithLogitsLoss()
    metrics = [
        smp_utils.metrics.IoU(threshold=0.5)
    ]
    pretrained_resnet = Pretrained_Model(backbone = backbone, train_data_loader = train_data_loader, val_data_loader = val_data_loader, test_data_loader = test_data_loader, encoder_weights = encoder_weights, activation = activation, metrics = metrics, LR = LR, loss = loss, device = device, base_loc = BASE_PATH, name = f'RESNET18_{data_source}')

    if TRAIN:
        pretrained_resnet.run_training(n_epochs)

    '''
    Evaluate pretrained model
    '''
    _ = pretrained_resnet.load() # always load the best model
    pretrained_resnet.run_testing()
    _ = update_results(pretrained_resnet, RESULTS, RESULT_PATH)

    if debug:
        plot_prediction(pretrained_resnet, test_data_loader, device)
    all_models.append(pretrained_resnet)

    '''
    PRETRAINED MOBILENET
    '''

    backbone = 'timm-mobilenetv3_large_100'
    encoder_weights = 'imagenet'
    activation = None

    loss = smp.utils.losses.BCEWithLogitsLoss()
    metrics = [
        smp_utils.metrics.IoU(threshold = 0.5),
    ]
    pretrained_mobilenet = Pretrained_Model(backbone = backbone, train_data_loader = train_data_loader, val_data_loader = val_data_loader, test_data_loader = test_data_loader, encoder_weights = encoder_weights, activation = activation, metrics = metrics, LR = LR, loss = loss, device = device, base_loc = BASE_PATH, name = f'mobilenetv3_large_100_{data_source}')

    if TRAIN:
        pretrained_mobilenet.run_training(n_epochs)
    '''
    Evaluate pretrained model
    '''
    _ = pretrained_mobilenet.load() # always load the best model
    pretrained_mobilenet.run_testing()
    _ = update_results(pretrained_mobilenet, RESULTS, RESULT_PATH)

    if debug:
        plot_prediction(pretrained_mobilenet, test_data_loader, device)
    all_models.append(pretrained_mobilenet)


    '''
    Plots on Test Data
    '''
    if plot:
    # Plot some test results' class channel breakdowns
        check_plotter_channels_breakdown = Plotter()
        for mod in all_models:
            for i in range(5):
                try:
                    print('Plotting breakdown channels')
                    # Plot every model made
                    check_plotter_channels_breakdown.sanity_check(test_img_folder+'/' , test_mask_folder+'/', predicted_breakdown=True, predict=True, imsize=imsize, model=mod, test_type=f'render_test_{mod.name}')
                    check_plotter_channels_breakdown.sanity_check(real_test_img_folder+'/' , real_test_mask_folder+'/', predicted_breakdown=True, predict=True, imsize=imsize, model=mod, test_type=f'real_test_{mod.name}')

                except RuntimeError:
                    continue

    total_t1 = time.time()
    total_time = (total_t1 - total_t0)/60
    print(f'TOTAL RUNTIME: {total_time} minutes')

if __name__ == '__main__':
    print('Running modeling.py')
    RUN_MODEL_LOOP(TRAIN = True, debug = False, plot = True, data_source = 'ground')
