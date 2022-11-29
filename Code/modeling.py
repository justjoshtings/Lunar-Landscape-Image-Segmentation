"""
modeling.py
Script to do modeling

author: @saharae, @justjoshtings
created: 11/17/2022
"""

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
train_data = CustomDataLoader(img_folder=train_img_folder, mask_folder=train_mask_folder, batch_size=batch_size, imsize=imsize, num_classes=num_classes)
train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

val_data = CustomDataLoader(img_folder=val_img_folder, mask_folder=val_mask_folder, batch_size=batch_size, imsize=imsize, num_classes=num_classes)
val_data_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

test_data = CustomDataLoader(img_folder=test_img_folder, mask_folder=test_mask_folder, batch_size=batch_size, imsize=imsize, num_classes=num_classes)
test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

real_test_data = CustomDataLoader(img_folder=real_test_img_folder, mask_folder=real_test_mask_folder, batch_size=batch_size, imsize=imsize, num_classes=num_classes)
real_test_data_loader = DataLoader(real_test_data, batch_size=batch_size, shuffle=True)

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


# '''
# Load Model(s)
# '''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device..', device)
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

Unet = UNet_scratch().to(device)


# '''
# Train
# '''
n_epochs = 10
lossBCE = torch.nn.BCEWithLogitsLoss()
opt = AdamW(Unet.parameters(), lr = 0.01)
metric = Dice(num_classes = 4)

num_training_steps = n_epochs * len(train_data_loader)
print(num_training_steps, 'steps!!')
lr_scheduler = get_scheduler(name="linear", optimizer=opt, num_warmup_steps=0, num_training_steps=num_training_steps)
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

model = Model(Unet, loss = lossBCE, opt = opt, metric = metric, random_seed = 42, train_data_loader = train_data_loader, val_data_loader = val_data_loader, test_data_loader = test_data_loader, device = device, base_loc = BASE_PATH, name = "Unet_scratch_noaugment", log_file=None)
print(f'Training: {model.name}')

model.run_training(n_epochs = n_epochs, device = device, save_every = 2, load = True)
model.plot_train(save_loc = RESULT_PATH)

# '''
# Validation Loop
# '''


# '''
# Save Model Weights
# '''


# '''
# Evaluate Model(s) on Test Data
# '''