"""
modeling.py
Script to do modeling

author: @saharae, @justjoshtings
created: 11/17/2022
"""

from ImageProcessor import ImageProcessor
from CustomDataLoader import CustomDataLoader
from Plotter import Plotter
from Model import *
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW
from transformers import get_scheduler
import os
import gc
from tqdm.auto import tqdm
import time
from datetime import datetime
import datetime as dt
from torchvision import models


CODE_PATH = os.getcwd()
os.chdir('../..')
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

batch_size = 8
imsize = 256
num_classes = 4

'''
Create dataloader for train, validation, and testing dataset
'''
train_data = CustomDataLoader(img_folder=train_img_folder, mask_folder=train_mask_folder, batch_size=batch_size, imsize=imsize, num_classes=num_classes)
train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

val_data = CustomDataLoader(img_folder=val_img_folder, mask_folder=val_mask_folder, batch_size=batch_size, imsize=imsize, num_classes=num_classes)
val_data_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

test_data = CustomDataLoader(img_folder=test_img_folder, mask_folder=test_mask_folder, batch_size=batch_size, imsize=imsize, num_classes=num_classes)
test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

'''
Review and Check Preprocessing and DataLoader outputs are correctly performed
'''
# tuples of 2 tensor arrays
# Image element is tensor array of size [RGB channel, imsize, imsize]
# Mask element is tensor array of size [class channel, imsize, imsize]
sample_data = next(iter(train_data))

# list of len 2 (1 for image, 1 for mask).
# Image element in list is torch tensor of size [batch size, RGB channel, imsize, imsize]
# Mask element in list is torch tensor of size [batch size, class channel, imsize, imsize]
batch_data = next(iter(train_data_loader))

img_processor = ImageProcessor()

# print(sample_data[0].shape)
# print(sample_data[1].shape)
#
# print(batch_data[0].shape)
# print(batch_data[1].shape)
#
# # Reorder images for plotting
# sample_image = sample_data[0].permute(1,2,0)
# sample_mask = sample_data[1].permute(1,2,0)
# sample_image = sample_image.numpy()
# sample_mask = sample_mask.numpy()
#
# sample_mask = img_processor.rescale(sample_mask)
# # Reverse one hot encode predicted mask
# sample_mask_decoded = img_processor.reverse_one_hot_encode(sample_mask)
# sample_mask_decoded = img_processor.rescale(sample_mask_decoded)
#
# # Plot images to ensure correct processing steps
# check_plotter = Plotter()
# check_plotter.peek_images(sample_images=sample_image,sample_masks=sample_mask_decoded,file_name='current_test.png')
# check_plotter.sanity_check(train_img_folder+'/' , train_mask_folder+'/')
#
# print(sample_mask_decoded.shape)

# NEED TO CHECK PREPROCESSING STEPS - WHAT IS THE SEQUENCE OF STEPS
# Check endcoding of images in each step
# Check max and min values for img and masks before processing

def update_results(model, RESULTS, BASE_PATH):
    for epoch, val in enumerate(model.history['train_loss']):
        RESULTS.append([model.name, epoch, 'train_loss', val])
    for epoch, val in enumerate(model.history['val_loss']):
        RESULTS.append([model.name, epoch, 'val_loss', val])
    for epoch, val in enumerate(model.history['train_iou']):
        RESULTS.append([model.name, epoch, 'train_iou', val])
    for epoch, val in enumerate(model.history['val_iou']):
        RESULTS.append([model.name, epoch, 'val_iou', val])

    if os.path.exists(os.path.join(BASE_PATH, 'RESULTS.csv')):
        current_results = pd.read_csv(os.path.join(BASE_PATH, 'RESULTS.csv'))
    else:
        current_results = pd.DataFrame(columns = ['model_name', 'epoch', 'metric', 'value'])

    to_save = pd.concat([current_results, RESULTS])
    to_save.to_csv(os.path.join(BASE_PATH, 'RESULTS.csv'), index = False)
    print("results updated")
    return RESULTS
# '''
# Load Model(s)
# '''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device..', device)
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# '''
# Train
# '''
n_epochs = 2
lossBCE = torch.nn.BCEWithLogitsLoss()
metric = Dice(num_classes = 4)

num_training_steps = n_epochs * len(train_data_loader)
print(num_training_steps, 'steps!!')
progress_bar = tqdm(range(num_training_steps))
total_t0 = time.time()
sample_every = 100


print('training')

gc.collect()
torch.cuda.empty_cache()

# model = Model(Unet, loss = lossBCE, opt = opt, metric = metric, random_seed = 42, train_data_loader = train_data_loader, val_data_loader = val_data_loader, test_data_loader = test_data_loader, device = device, base_loc = BASE_PATH, name = "Initial_model", log_file=None)
#
# model.run_training(n_epochs = n_epochs, device = device, save_every = 2, load = True)
# model.plot_train(save_loc = RESULT_PATH)

# [model name, epoch, metric, value]
RESULTS = []

# SCRATCH UNET
Unet = UNet_scratch().to(device)
opt = AdamW(Unet.parameters(), lr = 0.01)
lr_scheduler = get_scheduler(name="linear", optimizer=opt, num_warmup_steps=0, num_training_steps=num_training_steps)


model = Model(Unet, loss = lossBCE, opt = opt, metric = metric, random_seed = 42, train_data_loader = train_data_loader, val_data_loader = val_data_loader, test_data_loader = test_data_loader, device = device, base_loc = BASE_PATH, name = "Unet_scratch_noaugment", log_file=None)
print(f'Training: {model.name}')

model.run_training(n_epochs = n_epochs, device = device, save_every = 2, load = True)
model.plot_train(save_loc = RESULT_PATH)

RESULTS = update_results(model, RESULTS, BASE_PATH)

# RESNET
pretrained = models.resnet18(pretrained = True)
pretrained.fc = nn.Linear(pretrained.fc.in_features, 4)

pretrained = models.resnet18(pretrained = True)
pretrained.fc = nn.Linear(pretrained.fc.in_features, 4)
opt = AdamW(pretrained.parameters(), lr = 0.01)
lr_scheduler = get_scheduler(name="linear", optimizer=opt, num_warmup_steps=0, num_training_steps=num_training_steps)

model = Model(pretrained, loss = lossBCE, opt = opt, metric = metric, random_seed = 42, train_data_loader = train_data_loader, val_data_loader = val_data_loader, test_data_loader = test_data_loader, device = device, base_loc = BASE_PATH, name = "RESNET", log_file=None)
print(f'Training: {model.name}')

model.run_training(n_epochs = n_epochs, device = device, save_every = 2, load = True)
model.plot_train(save_loc = RESULT_PATH)

RESULTS = update_results(model, RESULTS, BASE_PATH)

# '''
# saving results
# '''


# '''
# Evaluate Model(s) on Test Data
# '''