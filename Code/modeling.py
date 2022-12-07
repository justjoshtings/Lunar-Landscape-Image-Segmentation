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

# test = models.resnet18()
# print(test)

CODE_PATH = os.getcwd()
os.chdir('..')
BASE_PATH = os.getcwd()
os.chdir(CODE_PATH)
DATA_PATH = os.path.join(BASE_PATH, 'Data')

RESULT_PATH = os.path.join(BASE_PATH, 'Results')

if not os.path.exists(RESULT_PATH):
    os.mkdir(RESULT_PATH)
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

#test(test_data_loader)

'''
Review and Check Preprocessing and DataLoader outputs are correctly performed
'''
def do_preprocessing_checks():
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

# do_preprocessing_checks()

def update_results(model, RESULTS, RESULT_PATH):
    for metric in model.history.keys():
        for epoch, val in enumerate(model.history[metric]):
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

def plot_prediction(model, test_data_loader):
    print('plotting...')
    for step, batch in enumerate(test_data_loader):
        if step == 0:
            x_test, y_test = batch[0], batch[1]
            y_pred = model.model(x_test.to(device))
            print(f'TEST: {y_test.shape}', y_test)
            print(f'PRED: {y_pred.shape}', y_pred)
            np.save('y_test_batch.npy', y_test.cpu().detach().numpy())
            np.save('y_pred_batch.npy', y_pred.cpu().detach().numpy())

            y_pred_OHE = torch.softmax(y_pred, dim = 1)
            print(f'PRED OHE: {y_pred_OHE.shape}', y_pred_OHE)

            y_pred_reorder = y_pred_OHE.permute(0, 2, 3, 1)
            y_test_reorder = y_test.permute(0, 2, 3, 1)
            x_test_reorder = x_test.permute(0, 2, 3, 1)

            img = x_test_reorder.cpu().detach().numpy()[10]
            img_processor = ImageProcessor()

            predicted_image_decoded = img_processor.reverse_one_hot_encode(y_pred_reorder.cpu().detach().numpy()[10])
            predicted_image_decoded_mask = img_processor.reverse_one_hot_encode(y_test_reorder.cpu().detach().numpy()[10])
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

# '''
# SET UP device
# '''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device..', device)
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# '''
# SET hyperparams
# '''

n_epochs = 2
LR = 0.001
training_mode = True
metrics = {
    "Dice": Dice(num_classes = 4),
    "IOU": JaccardIndex(num_classes = 4)
}

lossBCE = torch.nn.BCEWithLogitsLoss()
num_training_steps = n_epochs * len(train_data_loader)

print(num_training_steps, 'steps!!')
total_t0 = time.time()

gc.collect()
torch.cuda.empty_cache()

# [model name, epoch, metric, value]
RESULTS = []

# SCRATCH UNET
Unet = UNet_scratch().to(device)
opt = AdamW(Unet.parameters(), lr = LR)
lr_scheduler = get_scheduler(name="linear", optimizer=opt, num_warmup_steps=0, num_training_steps=num_training_steps)

model = Model(Unet, loss = lossBCE, opt = opt, metrics = metrics, random_seed = 42, train_data_loader = train_data_loader, val_data_loader = val_data_loader, test_data_loader = test_data_loader, device = device, base_loc = BASE_PATH, name = "Unet_scratch_ground", log_file=None)

if training_mode:
    print('not training scratch model rn')
    model.run_training(n_epochs = n_epochs, device = device, save_every = 2, load = True)
    RESULTS = update_results(model, RESULTS, RESULT_PATH)
    model.plot_train(save_loc = RESULT_PATH)
else:
    last_epoch = model.load() # only if not training

#plot_prediction(model, test_data_loader)


RESULTS = []
# RESNET

# pretrained_resnet = Unet_transfer()
#
# opt = AdamW(pretrained_resnet.parameters(), lr = 0.01)
# lr_scheduler = get_scheduler(name="linear", optimizer=opt, num_warmup_steps=0, num_training_steps=num_training_steps)

# model = Model(pretrained_resnet, loss = lossBCE, opt = opt, metric = metric, random_seed = 42, train_data_loader = train_data_loader, val_data_loader = val_data_loader, test_data_loader = test_data_loader, device = device, base_loc = BASE_PATH, name = "RESNET", log_file=None)
# print(f'Training: {model.name}')
#
# model.run_training(n_epochs = n_epochs, device = device, save_every = 2, load = True)
# model.plot_train(save_loc = RESULT_PATH)
#
# RESULTS = update_results(model, RESULTS, RESULT_PATH)

# backbone = 'resnet18'
backbone = 'vgg11_bn'
encoder_weights = 'imagenet'
activation = None

loss = smp.utils.losses.BCEWithLogitsLoss()
metrics = [
    smp_utils.metrics.IoU(threshold=0.5),
]
pretrained = Pretrained_Model(backbone = backbone, train_data_loader = train_data_loader, val_data_loader = val_data_loader, test_data_loader = test_data_loader, encoder_weights = encoder_weights, activation = activation, metrics = metrics, LR = LR, loss = loss, device = device, base_loc = BASE_PATH, name = 'VGG11_BN_Ground')

n_epochs = 2

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
        check_plotter_channels_breakdown.sanity_check(test_img_folder+'/' , test_mask_folder+'/', predicted_breakdown=True, predict=True, imsize=imsize, model=model, test_type=f'render_test_{model.name}')
        check_plotter_channels_breakdown.sanity_check(real_test_img_folder+'/' , real_test_mask_folder+'/', predicted_breakdown=True, predict=True, imsize=imsize, model=model, test_type=f'real_test_{model.name}')
        # Resnet18 Backbone
        check_plotter_channels_breakdown.sanity_check(test_img_folder+'/' , test_mask_folder+'/', predicted_breakdown=True, predict=True, imsize=imsize, model=pretrained, test_type=f'render_test_{pretrained.name}')
        check_plotter_channels_breakdown.sanity_check(real_test_img_folder+'/' , real_test_mask_folder+'/', predicted_breakdown=True, predict=True, imsize=imsize, model=pretrained, test_type=f'real_test_{pretrained.name}')
    except RuntimeError:
        continue

# '''
# Evaluate Model(s) on Test Data
# '''