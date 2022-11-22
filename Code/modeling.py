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
import os

CODE_PATH = os.getcwd()
os.chdir('..')
BASE_PATH = os.getcwd()
os.chdir(CODE_PATH)
DATA_PATH = os.path.join(BASE_PATH, 'Data')

# Will need to split into train/test/val later
train_img_folder = DATA_PATH + '/images/render/'
mask_img_folder = DATA_PATH + '/images/clean/'
batch_size = 8
imsize = 256
num_classes = 4

'''
Create dataloader for train, validation, and testing dataset
'''
train_data = CustomDataLoader(img_folder=train_img_folder, mask_folder=mask_img_folder, batch_size=batch_size, imsize=imsize, num_classes=num_classes)
train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

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

sample_image = sample_data[0].numpy()
sample_mask = sample_data[1].numpy()

sample_mask = img_processor.rescale(sample_mask)
# Reverse one hot encode predicted mask
sample_mask_decoded = img_processor.reverse_one_hot_encode(sample_mask)
sample_mask_decoded = img_processor.rescale(sample_mask_decoded)

check_plotter = Plotter()
check_plotter.peek_images(sample_images=sample_image,sample_masks=sample_mask,file_name='current_test.png')
check_plotter.sanity_check(DATA_PATH + '/images/render/' , DATA_PATH + '/images/ground/')

# NEED TO DO AN IMAGE PLOT CHECK TO SEE IF EVERYTHING LOOKS GOOD OUT OF CUSTOM DATALOADER
    # mask is black and white need to change to RGB or check channels which one is first, and check 
# NEED TO CHECK PREPROCESSING STEPS - WHAT IS THE SEQUENCE OF STEPS
# Check endcoding of images in each step
# Check order of tensors channels first

'''
Load Model(s)
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
opt = Adam(model.parameters(), lr = 0.01)

print('training')
for e in range(n_epochs):
    model.train()

    for i in range(batch_data[0].shape[0]):
        x_train, y_train = batch_data[0][i], batch_data[1][i]
        pred = model(x_train)
        loss = lossBCE(pred, y_train)

        opt.zero_grad()
        lossBCE.backward()
        opt.step()

        print(f'testing functionality: loss is sorta {loss}')

'''
Evaluate Model(s)
'''