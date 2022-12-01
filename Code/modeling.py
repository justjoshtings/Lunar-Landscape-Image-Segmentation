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
    new_res = pd.DataFrame(RESULTS, columns = ['model_name', 'epoch', 'metric', 'value'])
    to_save = pd.concat([current_results, new_res])
    to_save.to_csv(os.path.join(BASE_PATH, 'RESULTS.csv'), index = False)
    print("results updated")
    return RESULTS

def plot_prediction(model, test_data_loader):
    for step, batch in enumerate(test_data_loader):
        if step == 0:
            x_test, y_test = batch[0], batch[1]
            y_pred = model.model(x_test.to(device).float())
            y_pred_OHE = torch.softmax(y_pred.float(), dim = 1)
            y_pred_reorder = y_pred_OHE.permute(0, 2, 3, 1)

            y_test_reorder = y_test.permute(0, 2, 3, 1)

            x_test_reorder = x_test.permute(0, 2, 3, 1)
            img = x_test_reorder.cpu().detach().numpy()[13]

            img_processor = ImageProcessor()
            predicted_image_decoded = img_processor.reverse_one_hot_encode(y_pred_reorder.cpu().detach().numpy()[13])
            predicted_image_decoded_mask = img_processor.reverse_one_hot_encode(y_test_reorder.cpu().detach().numpy()[13])
            fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (10, 8))

            axes[0].imshow(predicted_image_decoded)
            axes[0].set_title('help - predicted')
            axes[1].imshow(predicted_image_decoded_mask)
            axes[1].set_title('help - mask')
            axes[2].imshow(img)
            axes[2].set_title('help - actual')
            fig.savefig('prayers.png')
            print('done')
            return
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
n_epochs = 10
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

# [model name, epoch, metric, value]
RESULTS = []

# SCRATCH UNET
Unet = UNet_scratch().to(device)
opt = AdamW(Unet.parameters(), lr = 0.01)
lr_scheduler = get_scheduler(name="linear", optimizer=opt, num_warmup_steps=0, num_training_steps=num_training_steps)

model = Model(Unet, loss = lossBCE, opt = opt, metric = metric, random_seed = 42, train_data_loader = train_data_loader, val_data_loader = val_data_loader, test_data_loader = test_data_loader, device = device, base_loc = BASE_PATH, name = "Unet_scratch", log_file=None)
print(f'Training: {model.name}')

# model.run_training(n_epochs = n_epochs, device = device, save_every = 2, load = True)
# model.plot_train(save_loc = RESULT_PATH)

# RESULTS = update_results(model, RESULTS, BASE_PATH)

#last_epoch = model.load() # only if not training
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
# RESULTS = update_results(model, RESULTS, BASE_PATH)


ENCODER = 'resnet18'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation

# create segmentation model with pretrained encoder
model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=7,
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

loss = smp.utils.losses.DiceLoss()
metrics = [
    smp_utils.metrics.IoU(threshold=0.5),
]

optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=0.0001),
])

train_epoch = smp_utils.train.TrainEpoch(
    model,
    loss=lossBCE,
    metrics=metrics,
    optimizer=opt,
    device=device,
    verbose=True,
)

valid_epoch = smp_utils.train.ValidEpoch(
    model,
    loss=lossBCE,
    metrics=metrics,
    device=device,
    verbose=True,
)

best_iou_score = 0.0
train_logs_list, valid_logs_list = [], []

for i in range(0, n_epochs):

    # Perform training & validation
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_data_loader)
    valid_logs = valid_epoch.run(val_data_loader)
    train_logs_list.append(train_logs)
    valid_logs_list.append(valid_logs)

    # Save model if a better val IoU score is obtained
    # if best_iou_score < valid_logs['iou_score']:
    #     best_iou_score = valid_logs['iou_score']
    #     torch.save(model, './best_model.pth')
    #     print('Model saved!')

# '''
# Evaluate Model(s) on Test Data
# '''