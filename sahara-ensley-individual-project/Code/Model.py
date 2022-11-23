"""
Model.py
Object to handle data generators.

author: @saharae, @justjoshtings
created: 11/12/2022
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import copy
import torch
import torch.nn as nn
import torchvision
import time
import datetime as dt
from transformers import get_scheduler
from tqdm.auto import tqdm


# WILL NEED TO CLEAN THIS WHOLE MESS UP LATER!!!
# optimizer = adam?
# loss = pixel wise cross entropy, jacard something
# metric = mean IOU
# pretrained models = unet, resnet backbone with upsampling, ... lots of others

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class Encoder(nn.Module):
    def __init__(self, chs = (3, 16, 32, 64)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs

class Decoder(nn.Module):
    def __init__(self, chs = (64, 32, 16), verbose = False):
        super().__init__()
        self.verbose = verbose
        self.chs = chs
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i + 1], 2, 2) for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            if self.verbose:
                print(f'decoder forward: x shape: {x.shape}')
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim = 1)
            x = self.dec_blocks[i](x)
        return x

    def crop(self, enc_ftrs, x):
        if self.verbose:
            print(f'crop enc_ftrs shape: {enc_ftrs.shape}')
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs

class UNet_scratch(nn.Module):
    def __init__(self, enc_chs = (3, 16, 32, 64), dec_chs = (64, 32, 16), num_class = 4, retain_dim = True, out_sz = (256, 256), verbose = False):
        super().__init__()
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs, verbose = verbose)
        self.head = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim = retain_dim
        self.out_sz = out_sz

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        if self.retain_dim:
            out = torch.nn.functional.interpolate(out, self.out_sz)
        return out

class Model:
    '''
    Object to handle model and related methods.
    '''

    ## NEED TO ADD THIS
    def __init__(self, model, loss, opt, random_seed, train_data_loader, val_data_loader, test_data_loader, name = None, log_file=None):
        '''
        Params:
            self: instance of object
            log_file (str): default is None to not have logging, otherwise, specify logging path ../filepath/log.log

        '''
        self.log_file = log_file
        self.model = model
        self.random_seed = random_seed
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.name = name
        self.loss = loss
        self.opt = opt
        self.history = {
            "train_loss":[],
            "val_loss":[],
            "train_iou":[],
            "val_iou":[]
        }

    def run_training(self, n_epochs, device):
        num_training_steps = n_epochs * len(self.train_data_loader)
        progress_bar = tqdm(range(num_training_steps))

        lr_scheduler = get_scheduler(name = "linear", optimizer = self.opt, num_warmup_steps = 0, num_training_steps = num_training_steps)
        total_t0 = time.time()
        sample_every = 100


        for e in range(n_epochs):
            running_train_loss = 0
            running_val_loss = 0
            running_train_iou = 0
            running_val_iou = 0
            t0 = time.time()
            self.model.train()

            for step, batch in enumerate(self.train_data_loader):
                # print(batch[0].shape)
                # print(batch[1].shape)
                x_train, y_train = batch[0].to(device), batch[1].to(device)
                self.model.zero_grad()
                pred = self.model(x_train)
                # print(pred.shape)

                loss = self.loss(pred, y_train.float())
                running_train_loss += loss.item()
                #running_train_iou += self.IoU(pred, y_train.float())

                # print(pred, loss)
                # opt.zero_grad()
                loss.backward()
                self.opt.step()
                lr_scheduler.step()
                progress_bar.update(1)

            self.history["train_loss"].append((running_train_loss/len(self.train_data_loader)))
            #self.history["train_iou"].append((running_train_iou / len(self.train_data_loader)))

            self.model.eval()
            with torch.no_grad():
                for step, batch in enumerate(self.val_data_loader):
                    x_val, y_val = batch[0].to(device), batch[1].to(device)
                    y_val_pred = self.model(x_val)
                    loss = self.loss(y_val_pred, y_val.float())
                    #running_val_iou += self.IoU(y_val_pred, y_val.float())
                    running_val_loss += loss.item()
            self.history['val_loss'].append((running_val_loss/len(self.val_data_loader)))
            #self.history['val_iou'].append((running_val_iou/len(self.val_data_loader)))

            #print(f"EPOCH: {e} -- train_loss {self.history['train_loss'][-1]}, train_iou {self.history['train_iou'][-1]}, val_loss {self.history['val_loss'][-1]}, val_iou {self.history['val_iou'][-1]}")
            print(f"EPOCH: {e} -- train_loss {self.history['train_loss'][-1]}, val_loss {self.history['val_loss'][-1]}")
            # Measure how long this epoch took.
            print("")
            training_time = str(dt.timedelta(seconds = int(round((time.time() - t0)))))
            print(f"Training epoch took: {training_time}")

    def plot_train(self, save_loc):
        fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (8,8))
        axes[0].plot(self.history['train_loss'], color = "slate_grey", label = "Training Loss")
        axes[0].plot(self.history['val_loss'], color = "seagreen", label = "Training Loss")
        axes[0].legend()
        axes[0].title(f'MODEL: {self.name} LOSS')

        # axes[1].plot(self.history['train_iou'], color = "slate_grey", label = "Training IoU")
        # axes[1].plot(self.history['val_iou'], color = "seagreen", label = "Training IoU")
        # axes[1].legend()
        # axes[1].title(f'MODEL: {self.name} IoU')
        sns.despine()
        fig.save(os.path.join(save_loc, f'{self.name}_training_curves'))

    def IoU(self, y_pred, labels):
        y_pred = y_pred.squeeze(1)
        y_pred = y_pred.float().cpu().detach().numpy()
        labels = labels.float().cpu().detach().numpy()
        intersection = (y_pred & labels).sum((1,2))
        union = (y_pred | labels).sum((1,2))
        iou = (intersection + 1e-6) / (union + 1e-6) # to avoid divide by 0 error
        #thresholded = torch.clamp(20*(iou-0.5),0,10).ceil()/10
        thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
        return thresholded
