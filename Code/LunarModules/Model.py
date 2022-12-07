"""
Model.py
Object to handle data generators.

author: @saharae, @justjoshtings
created: 11/12/2022


CITATIONS:
Pretrained model backbone based on this example: https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb
Scratch model based on this code: https://amaarora.github.io/2020/09/13/unet.html
Weight initialization based on this code: https://wandb.ai/wandb_fc/tips/reports/How-to-Initialize-Weights-in-PyTorch--VmlldzoxNjcwOTg1
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
from torchmetrics import JaccardIndex
from torchmetrics import Dice
from torchvision import models
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smp_utils
from torch.optim import Adam, AdamW, SGD

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class Down(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3)
        self.convnorm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3)

        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3)
        self.convnorm2 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3)

        self.conv5 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3)
        self.convnorm3 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3)

        self.pool = nn.MaxPool2d(kernel_size = 2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))

    def forward(self, x):
        ft_maps = []

        x = self.conv2(self.convnorm1(self.relu(self.conv1(x))))
        ft_maps.append(x)
        x = self.pool(x)

        x = self.conv4(self.convnorm2(self.relu(self.conv3(x))))
        ft_maps.append(x)
        x = self.pool(x)

        x = self.conv6(self.convnorm3(self.relu(self.conv5(x))))
        ft_maps.append(x)
        x = self.pool(x)

        return ft_maps


class Up(nn.Module):
    def __init__(self, verbose = False):
        super().__init__()
        self.verbose = verbose

        self.conv_trans1 = nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = 2, stride = 2)
        self.conv_trans2 = nn.ConvTranspose2d(in_channels = 32, out_channels = 16, kernel_size = 2, stride = 2)

        self.conv1 = nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3)
        self.convnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3)

        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size = 3)
        self.convnorm2 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3)

        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))

    def forward(self, x, encoder_features):
        x = self.conv_trans1(x)
        ft = self.crop(encoder_features[0], x)
        x = torch.cat([x, ft], dim = 1)
        x = self.conv2(self.convnorm1(self.relu(self.conv1(x)))) # decoder block 1

        x = self.conv_trans2(x)
        ft = self.crop(encoder_features[1], x)
        x = torch.cat([x, ft], dim = 1)
        x = self.conv4(self.convnorm2(self.relu(self.conv3(x)))) # decoder block 2

        return x

    def crop(self, enc_ftrs, x):
        if self.verbose:
            print(f'crop enc_ftrs shape: {enc_ftrs.shape}')
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs

class UNet_scratch(nn.Module):
    def __init__(self, num_class = 4, retain_dim = True, out_sz = (256, 256), verbose = False):
        super().__init__()
        self.encoder = Down()
        self.decoder = Up(verbose = verbose)

        self.head = nn.Conv2d(in_channels = 16, out_channels = num_class, kernel_size = 1)
        self.retain_dim = retain_dim
        self.out_sz = out_sz

    def forward(self, x):
        ft_maps = self.encoder(x)
        out = self.decoder(ft_maps[::-1][0], ft_maps[::-1][1:])
        out = self.head(out)

        if self.retain_dim:
            out = torch.nn.functional.interpolate(out, self.out_sz)
        return out

class Model:
    '''
    Object to handle model and related methods.
    '''

    ## NEED TO ADD THIS
    def __init__(self, model, loss, opt, scheduler, metrics, random_seed, train_data_loader, val_data_loader, test_data_loader, device, base_loc = None, name = None, log_file=None):
        '''
        Params:
            self: instance of object
            log_file (str): default is None to not have logging, otherwise, specify logging path ../filepath/log.log

        '''
        self.log_file = log_file
        self.model = model.to(device)
        self.random_seed = random_seed
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.name = name
        self.loss = loss
        self.opt = opt
        self.scheduler = scheduler
        self.history = {
            "train_loss":[],
            "val_loss":[]
        }
        for metric in metrics.keys():
            self.history[f'train_{metric}'] = []
            self.history[f'val_{metric}'] = []

        self.metrics = metrics
        self.base_loc = base_loc
        self.device = device

    def load(self):
        last_e = self.load_latest_model(self.device)
        return last_e

    def run_training(self, n_epochs, device, save_on = 'val_IOU', load = False):
        print(f'Training: {self.name}')
        num_training_steps = n_epochs * len(self.train_data_loader)
        progress_bar = tqdm(range(num_training_steps))
        lr_scheduler = get_scheduler(name = "linear", optimizer = self.opt, num_warmup_steps = 0, num_training_steps = num_training_steps)

        if load:
            last_e = self.load_latest_model(device)
        else:
            last_e = 0

        if save_on not in self.history.keys():
            print('that save metric doesnt exist, make sure the metric is passed into the function')
            return

        best_met = 0
        for e in range(last_e, n_epochs):

            running_metrics = {
                'running_train_loss': 0,
                'running_val_loss': 0
            }
            for metric in self.metrics.keys():
                running_metrics[f'running_train_{metric}'] = 0
                running_metrics[f'running_val_{metric}'] = 0

            t0 = time.time()
            self.model.train()

            for step, batch in enumerate(self.train_data_loader):
                x_train, y_train = batch[0].to(device), batch[1].to(device)
                x_train.requires_grad = True

                self.model.zero_grad()
                pred = self.model(x_train.float())
                loss = self.loss(pred, y_train.float())

                pred_soft = torch.softmax(pred, dim = 1)
                pred_argmax = torch.argmax(pred_soft, dim = 1)

                running_metrics['running_train_loss'] += loss.item()
                for metric in self.metrics.keys():
                    m = self.metrics[metric]
                    running_metrics[f'running_train_{metric}'] += m(pred_argmax.cpu(), torch.argmax(torch.softmax(y_train.float(), dim = 1), dim = 1).cpu())

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                lr_scheduler.step()
                progress_bar.update(1)

            self.history["train_loss"].append((running_metrics['running_train_loss']/len(self.train_data_loader)))
            for metric in self.metrics.keys():
                self.history[f'train_{metric}'].append((running_metrics[f'running_train_{metric}'] / len(self.train_data_loader)).numpy())

            self.model.eval()
            with torch.no_grad():
                for step, batch in enumerate(self.val_data_loader):
                    x_val, y_val = batch[0].to(device), batch[1].to(device)
                    y_val_pred = self.model(x_val.float())
                    loss = self.loss(y_val_pred, y_val.float())

                    running_metrics['running_val_loss'] += loss.item()
                    for metric in self.metrics.keys():
                        m = self.metrics[metric]
                        running_metrics[f'running_val_{metric}'] += m(torch.argmax(torch.softmax(y_val_pred.float(), dim = 1), dim = 1).cpu(), torch.argmax(torch.softmax(y_val.float(), dim = 1), dim = 1).cpu())

            self.history["val_loss"].append((running_metrics['running_val_loss']/len(self.val_data_loader)))
            for metric in self.metrics.keys():
                self.history[f'val_{metric}'].append((running_metrics[f'running_val_{metric}'] / len(self.train_data_loader)).numpy())

            s = f"EPOCH: {e} -- "
            for metric in self.history.keys():
                s += f"{metric} {self.history[metric][-1]} "

            print(s)

            if self.history[save_on] > best_met:
                self.save_model(e)
            # Measure how long this epoch took.
            print("")
            training_time = str(dt.timedelta(seconds = int(round((time.time() - t0)))))
            print(f"Training epoch took: {training_time}")


    def run_test(self):
        running_metrics = {
            'running_test_loss':0
        }
        for metric in self.metrics.keys():
            running_metrics[f'running_test_{metric}'] = 0

        self.model.eval()
        with torch.no_grad():

            for step, batch in enumerate(self.test_data_loader):
                x_test, y_test = batch[0].to(self.device), batch[1].to(self.device)
                y_test_pred = self.model(x_test.float())
                loss = self.loss(y_test_pred, y_test.float())

                running_metrics['running_test_loss'] += loss.item()
                for metric in self.metrics.keys():
                    m = self.metrics[metric]
                    running_metrics[f'running_test_{metric}'] += m(torch.argmax(torch.softmax(y_test_pred.float(), dim = 1), dim = 1).cpu(), torch.argmax(y_test.float(), dim = 1).cpu())

        s = f"TESTING: "
        for metric in self.metrics.keys():
            s += f"{metric} {round(running_metrics[f'running_test_{metric}']/len(self.test_data_loader), 4)} "

        for metric in self.metrics.keys():
            self.history[f'test_{metric}'].append((running_metrics[f'running_test_{metric}'] / len(self.test_data_loader)).numpy())

        print(s)


    def predict(self, img):
        x_test = img.to(self.device)
        y_pred = self.model(x_test.float())
        return torch.argmax(torch.softmax(y_pred.float(), dim = 1), dim = 1).cpu()


    def plot_train(self, save_loc):
        metrics = self.metrics.keys()
        metrics = np.unique([m[m.find('_')+1:] for m in metrics])
        fig, axes = plt.subplots(nrows = 1, ncols = len(metrics), figsize = (8,8))

        axes[0].plot(self.history['train_loss'], color = "slategrey", label = "Training Loss")
        axes[0].plot(self.history['val_loss'], color = "seagreen", label = "Training Loss")
        axes[0].legend()
        axes[0].set_title(f'MODEL: {self.name} LOSS')

        for i, ax in enumerate(axes[1:]):
            ax.plot(self.history[f'train_{metrics[i]}'], color = "slategrey", label = f"Training {metrics[i]}")
            ax.plot(self.history[f'val_{metrics[i]}'], color = "seagreen", label = f"Training {metrics[i]}")
            ax.legend()
            ax.set_title(f'MODEL: {self.name} {metrics[i]}')
        sns.despine()
        fig.savefig(os.path.join(save_loc, f'{self.name}_training_curves'))

    def save_model(self, epoch):
        save_loc = os.path.join(self.base_loc, 'Models')
        if not os.path.exists(save_loc):
            print('Making Model Dir')
            os.mkdir(save_loc)
        torch.save(self.model.state_dict(), os.path.join(save_loc, f"model_{self.name}_EP{epoch}.pt"))
        print('saving model ...')

    def load_latest_model(self, device):
        model_loc = os.path.join(self.base_loc, 'Models')
        if not os.path.exists(model_loc):
            print('Model folder doesnt exist, skipping loading...')
            return 0
        models = [x for x in os.listdir(model_loc) if '.pt' in x and self.name+'_EP' in x]
        if len(models) == 0:
            print('No models saved to load')
            return 0
        else:
            saved_iterations = sorted([int(x[x.find('EP')+2:x.find('.pt')]) for x in models])
            latest_model = f'model_{self.name}_EP{saved_iterations[-1]}.pt'
            print(f"Latest Model Saved: {latest_model}")
            model_file = os.path.join(model_loc, latest_model)
            self.model.load_state_dict(torch.load(model_file, map_location = device))
            print("Model Loaded!")
            return saved_iterations[-1]

class Pretrained_Model:

    def __init__(self, backbone, encoder_weights, activation, metrics, LR, loss, device, train_data_loader, val_data_loader, test_data_loader, base_loc, name = None):
        self.backbone = backbone
        self.encoder_weights = encoder_weights
        self.activation = activation
        self.metrics = metrics
        self.LR = LR
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader

        self.loss = loss
        self.device = device
        self.name = name
        self.base_loc = base_loc

        self.model = smp.Unet(
                    encoder_name=self.backbone,
                    encoder_weights=self.encoder_weights,
                    classes=4,
                    activation=self.activation,
                )

        self.optimizer = SGD(params=self.model.parameters(), lr=self.LR)

        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(self.backbone, self.encoder_weights)

        self.train_epoch = smp_utils.train.TrainEpoch(
            self.model,
            loss = self.loss,
            metrics = self.metrics,
            optimizer = self.optimizer,
            device = self.device,
            verbose = True,
        )

        self.valid_epoch = smp_utils.train.ValidEpoch(
            self.model,
            loss = self.loss,
            metrics = self.metrics,
            device = self.device,
            verbose = True,
        )

    def run_training(self, n_epochs):
        print(f"Training: {self.name}")
        best_val_iou = 0.0
        train_logs_list, valid_logs_list = [], []
        self.history = {}

        for i in range(0, n_epochs):
            # Perform training & validation
            print('\nEpoch: {}'.format(i))
            train_logs = self.train_epoch.run(self.train_data_loader)
            print(train_logs)
            val_logs = self.valid_epoch.run(self.val_data_loader)
            for key in train_logs.keys():
                if key in self.history.keys():
                    self.history[f'train_{key}'].append([train_logs[key]])
                else:
                    self.history[f'train_{key}'] = [train_logs[key]]
            for key in val_logs.keys():
                if key in self.history.keys():
                    self.history[f'val_{key}'].append([val_logs[key]])
                else:
                    self.history[f'val_{key}'] = [val_logs[key]]

            if self.history[f'val_iou_score'][-1] > best_val_iou:
                best_val_iou = self.history[f'val_iou_score'][-1]
                self.save_model(i)

    def save_model(self, epoch):
        save_loc = os.path.join(self.base_loc, 'Models')
        if not os.path.exists(save_loc):
            print('Making Model Dir')
            os.mkdir(save_loc)
        torch.save(self.model.state_dict(), os.path.join(save_loc, f"model_{self.name}_EP{epoch}.pt"))
        print('saving model ...')

    def run_testing(self):
        test_epoch = smp.utils.train.ValidEpoch(
            model = self.model,
            loss = self.loss,
            metrics = self.metrics,
            device = self.device,
        )
        logs = test_epoch.run(self.test_data_loader)
        for key in logs.keys():
            if key in self.history.keys():
                self.history[f'test_{key}'].append([logs[key]])
            else:
                self.history[f'test_{key}'] = [logs[key]]

    def load(self):
        last_e = self.load_latest_model(self.device)
        return last_e

    def load_latest_model(self, device):
        model_loc = os.path.join(self.base_loc, 'Models')
        if not os.path.exists(model_loc):
            print('Model folder doesnt exist, skipping loading...')
            return 0
        models = [x for x in os.listdir(model_loc) if '.pt' in x and self.name+'_EP' in x]
        if len(models) == 0:
            print('No models saved to load')
            return 0
        else:
            saved_iterations = sorted([int(x[x.find('EP')+2:x.find('.pt')]) for x in models])
            latest_model = f'model_{self.name}_EP{saved_iterations[-1]}.pt'
            print(f"Latest Model Saved: {latest_model}")
            model_file = os.path.join(model_loc, latest_model)
            self.model.load_state_dict(torch.load(model_file, map_location = device))
            print("Model Loaded!")
            return saved_iterations[-1]