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

def RUN_MODEL_LOOP(TRAIN = True, debug = False, plot = True):
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
    n_epochs = 2
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
    Unet = UNet_scratch().to(device)
    opt = AdamW(Unet.parameters(), lr = LR)
    lr_scheduler = get_scheduler(name="linear", optimizer=opt, num_warmup_steps=0, num_training_steps=num_training_steps)
    model = Model(Unet, loss = lossBCE, opt = opt, scheduler = lr_scheduler, metrics = metrics, random_seed = 42, train_data_loader = train_data_loader, val_data_loader = val_data_loader, test_data_loader = test_data_loader, device = device, base_loc = BASE_PATH, name = "Unet_scratch_ground", log_file=None)

    if TRAIN:
        print('Training ', num_training_steps, 'steps!!')
        model.run_training(n_epochs = n_epochs, device = device, save_on = 'val_IOU', load = True)
        if plot:
            model.plot_train(save_loc = RESULT_PATH)

    '''
    Evaluate pretrained
    '''
    _ = model.load() # always load latest model
    model.run_test()
    _ = update_results(model, RESULTS, RESULT_PATH)

    if debug:
        plot_prediction(model, test_data_loader, device)

    '''
    PRETRAINED RESNET
    '''
    RESULTS = []

    # backbone = 'resnet18'
    backbone = 'vgg11_bn'
    encoder_weights = 'imagenet'
    activation = None

    loss = smp.utils.losses.BCEWithLogitsLoss()
    metrics = [
        smp_utils.metrics.IoU(threshold=0.5),
    ]
    pretrained = Pretrained_Model(backbone = backbone, train_data_loader = train_data_loader, val_data_loader = val_data_loader, test_data_loader = test_data_loader, encoder_weights = encoder_weights, activation = activation, metrics = metrics, LR = LR, loss = loss, device = device, base_loc = BASE_PATH, name = 'VGG11_BN_Ground')

    if TRAIN:
        pretrained.run_training(n_epochs)

    '''
    Evaluate pretrained model
    '''
    _ = pretrained.load() # always load the best model
    pretrained.run_testing()
    _ = update_results(pretrained, RESULTS, RESULT_PATH)

    if debug:
        plot_prediction(pretrained, test_data_loader, device)

    '''
    Plots on Test Data
    '''
    if plot:
    # Plot some test results' class channel breakdowns
        check_plotter_channels_breakdown = Plotter()
        for i in range(5):
            try:
                print('Plotting breakdown channels')
                # Unet Scratch
                check_plotter_channels_breakdown.sanity_check(test_img_folder+'/' , test_mask_folder+'/', predicted_breakdown=True, predict=True, imsize=imsize, model=model, test_type=f'render_test_{model.name}')
                check_plotter_channels_breakdown.sanity_check(real_test_img_folder+'/' , real_test_mask_folder+'/', predicted_breakdown=True, predict=True, imsize=imsize, model=model, test_type=f'real_test_{model.name}')
                # Resnet18 Backbone
                check_plotter_channels_breakdown.sanity_check(test_img_folder+'/' , test_mask_folder+'/', predicted_breakdown=True, predict=True, imsize=imsize, model=pretrained, test_type=f'render_test_{pretrained.name}')
                check_plotter_channels_breakdown.sanity_check(real_test_img_folder+'/' , real_test_mask_folder+'/', predicted_breakdown=True, predict=True, imsize=imsize, model=pretrained, test_type=f'real_test_{pretrained.name}')
            except RuntimeError:
                continue

# backbone = 'resnet18'
# backbone = 'vgg11_bn'
backbone = 'timm-mobilenetv3_large_100'
encoder_weights = 'imagenet'
activation = None

loss = smp.utils.losses.BCEWithLogitsLoss()
metrics = [
    smp_utils.metrics.IoU(threshold=0.5),
]
pretrained = Pretrained_Model(backbone = backbone, train_data_loader = train_data_loader, val_data_loader = val_data_loader, test_data_loader = test_data_loader, encoder_weights = encoder_weights, activation = activation, metrics = metrics, LR = LR, loss = loss, device = device, base_loc = BASE_PATH, name = 'mobilenetv3_large_100')

n_epochs = 20

# RESULTS = update_results(pretrained, RESULTS, BASE_PATH)

if training_mode:
    pretrained.run_training(n_epochs)
    RESULTS = update_results(pretrained, RESULTS, RESULT_PATH)
else:
    last_epoch = pretrained.load() # only if not training

# plot_prediction(pretrained, test_data_loader)

# '''
# Plots on Test Data
# '''
# Plot some test results' class channel breakdowns
check_plotter_channels_breakdown = Plotter()
for i in range(5):
    try:
        print('Plotting breakdown channels')
        # Unet Scratch
        # check_plotter_channels_breakdown.sanity_check(test_img_folder+'/' , test_mask_folder+'/', predicted_breakdown=True, predict=True, imsize=imsize, model=model, test_type=f'render_test_{model.name}')
        # check_plotter_channels_breakdown.sanity_check(real_test_img_folder+'/' , real_test_mask_folder+'/', predicted_breakdown=True, predict=True, imsize=imsize, model=model, test_type=f'real_test_{model.name}')
        # Pretrained Model
        check_plotter_channels_breakdown.sanity_check(test_img_folder+'/' , test_mask_folder+'/', predicted_breakdown=True, predict=True, imsize=imsize, model=pretrained, test_type=f'render_test_{pretrained.name}')
        check_plotter_channels_breakdown.sanity_check(real_test_img_folder+'/' , real_test_mask_folder+'/', predicted_breakdown=True, predict=True, imsize=imsize, model=pretrained, test_type=f'real_test_{pretrained.name}')
    except RuntimeError:
        continue

# '''
# Evaluate Model(s) on Test Data
# '''

if __name__ == '__main__':
    print('Running modeling.py')
    RUN_MODEL_LOOP()

