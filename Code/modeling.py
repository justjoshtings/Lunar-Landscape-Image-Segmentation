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

CODE_PATH = os.getcwd()
os.chdir('..')
BASE_PATH = os.getcwd()
os.chdir(CODE_PATH)
DATA_PATH = os.path.join(BASE_PATH, 'Data')

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

# tuples of 2 tensor arrays
# Image element is tensor array of size [imsize, imsize, RGB channel]
# Mask element is tensor array of size [imsize, imsize, class channel]
sample_data = next(iter(train_data))

# list of len 2 (1 for image, 1 for mask). 
# Image element in list is torch tensor of size [batch size, imsize, imsize, RGB channel]
# Mask element in list is torch tensor of size [batch size, imsize, imsize, class channel]
batch_data = next(iter(train_data_loader))

img_processor = ImageProcessor()

print(sample_data[0].shape)
print(sample_data[1].shape)

print(batch_data[0].shape)
print(batch_data[1].shape)

sample_image = sample_data[0].permute(1,2,0)
sample_mask = sample_data[1].permute(1,2,0)
sample_image = sample_image.numpy()
sample_mask = sample_mask.numpy()

sample_mask = img_processor.rescale(sample_mask)
# Reverse one hot encode predicted mask
sample_mask_decoded = img_processor.reverse_one_hot_encode(sample_mask)
sample_mask_decoded = img_processor.rescale(sample_mask_decoded)

check_plotter = Plotter()
check_plotter.peek_images(sample_images=sample_image,sample_masks=sample_mask,file_name='current_test.png')
check_plotter.sanity_check(train_img_folder+'/' , train_mask_folder+'/')

# NEED TO DO AN IMAGE PLOT CHECK TO SEE IF EVERYTHING LOOKS GOOD OUT OF CUSTOM DATALOADER
    # mask is black and white need to change to RGB or check channels which one is first, and check 
# NEED TO CHECK PREPROCESSING STEPS - WHAT IS THE SEQUENCE OF STEPS
# Check endcoding of images in each step
# Check order of tensors channels first

'''
Load Model(s)
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device..', device)
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model = UNet_scratch().to(device)


'''
Train
'''
n_epochs = 10
lossBCE = torch.nn.BCEWithLogitsLoss()
opt = AdamW(model.parameters(), lr = 0.01)

num_training_steps = n_epochs * len(train_data_loader)
print(num_training_steps, 'steps!!')
lr_scheduler = get_scheduler(name="linear", optimizer=opt, num_warmup_steps=0, num_training_steps=num_training_steps)
progress_bar = tqdm(range(num_training_steps))
total_t0 = time.time()
sample_every = 100


print('training')

gc.collect()
torch.cuda.empty_cache()

for e in range(n_epochs):
    t0 = time.time()
    model.train()

    # for i in range(batch_data[0].shape[0]):
    for step, batch in enumerate(train_data_loader):
        print(batch[0].shape)
        print(batch[1].shape)
        x_train, y_train = batch[0].to(device), batch[1].to(device)
        model.zero_grad() 
        pred = model(x_train)
        print(pred.shape)
        loss = lossBCE(pred, y_train)

        print(pred, loss)

        # opt.zero_grad()
        loss.backward()
        opt.step()
        lr_scheduler.step()
        progress_bar.update(1)

        print(f'testing functionality: loss is sorta {loss}')

    # Measure how long this epoch took.
    print("")
    training_time = str(dt.timedelta(seconds=int(round((time.time() - t0)))))
    print(f"Training epoch took: {training_time}")

'''
Validation Loop
'''


'''
Save Model Weights
'''


'''
Evaluate Model(s)
'''