"""
Model.py
Object to handle modeling methods.

author: @saharae, @justjoshtings
created: 11/12/2022

CITATIONS:
Pretrained model backbone based on this example: https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb
Scratch model based on this code: https://amaarora.github.io/2020/09/13/unet.html
Weight initialization based on this code: https://wandb.ai/wandb_fc/tips/reports/How-to-Initialize-Weights-in-PyTorch--VmlldzoxNjcwOTg1
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import torch.nn as nn
import torchvision
import time
import datetime as dt
from transformers import get_scheduler
from tqdm.auto import tqdm
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smp_utils
from torch.optim import SGD

class Down(nn.Module):
    '''
    ENCODER of Custom UNet
    '''
    def __init__(self, verbose = False):
        super().__init__()
        self.verbose = verbose
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, padding = "same")
        self.convnorm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, padding = "same")

        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = "same")
        self.convnorm2 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = "same")

        self.conv5 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = "same")
        self.convnorm3 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = "same")

        self.pool = nn.MaxPool2d(kernel_size = 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = 0.2)

        ## Setting Weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))

    def forward(self, x):
        ## Running forward
        ft_maps = []

        # First Block
        x = self.relu(self.conv2(self.relu(self.convnorm1(self.conv1(x)))))
        ft_maps.append(x)
        if self.verbose:
            print('size of first FTMP: ', x.shape)
        x = self.pool(x)
        if self.verbose:
            print('size after first down block: ', x.shape)
        x = self.dropout(x)

        # Second Block
        x = self.relu(self.conv4(self.relu(self.convnorm2(self.conv3(x)))))
        ft_maps.append(x)
        if self.verbose:
            print('size of second FTMP: ', x.shape)
        x = self.pool(x)
        if self.verbose:
            print('size after second down block: ', x.shape)
        x = self.dropout(x)

        # Third Block
        x = self.relu(self.conv6(self.relu(self.convnorm3(self.conv5(x)))))
        ft_maps.append(x)
        if self.verbose:
            print('size of third FTMP: ', x.shape)
        x = self.pool(x)
        if self.verbose:
            print('size after third block: ', x.shape)
        x = self.dropout(x)

        return ft_maps


class Up(nn.Module):
    '''
    DECODER block of U-Net
    '''
    def __init__(self, verbose = False):
        super().__init__()
        self.verbose = verbose

        self.conv_trans1 = nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = 2, stride = 2)
        self.conv_trans2 = nn.ConvTranspose2d(in_channels = 32, out_channels = 16, kernel_size = 2, stride = 2)

        self.conv1 = nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, padding = "same")
        self.convnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = "same")

        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size = 3, padding = "same")
        self.convnorm2 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, padding = "same")

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = 0.2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))

    def forward(self, x, encoder_features):
        '''
        Forward loop
        :param x: input
        :param encoder_features: features from same level block of encoder
        :return:
        '''

        # first block
        x = self.conv_trans1(x)
        ft = self.crop(encoder_features[0], x)
        x = torch.cat([x, ft], dim = 1)
        if self.verbose:
            print('size after first concat: ', x.shape)
        x = self.dropout(x)
        x = self.relu(self.conv2(self.relu(self.convnorm1(self.conv1(x))))) # decoder block 1

        # second block
        x = self.conv_trans2(x)
        ft = self.crop(encoder_features[1], x)
        x = torch.cat([x, ft], dim = 1)
        if self.verbose:
            print('size after second concat: ', x.shape)
        x = self.dropout(x)
        x = self.relu(self.conv4(self.relu(self.convnorm2(self.conv3(x))))) # decoder block 2
        if self.verbose:
            print('final decoder size: ', x.shape)
        return x

    def crop(self, enc_ftrs, x):
        '''
        Crop the features to match the proper size of the inputs to the decoder blocks
        :param enc_ftrs: features to crop
        :param x: input to size match to
        :return:
        '''
        if self.verbose:
            print(f'crop enc_ftrs shape: {enc_ftrs.shape}')
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs

class UNet_scratch(nn.Module):
    '''
    COMBINED U-NET MODEL
    '''
    def __init__(self, num_class = 4, retain_dim = True, out_sz = (256, 256), verbose = False):
        super().__init__()
        self.encoder = Down(verbose = verbose)
        self.decoder = Up(verbose = verbose)

        self.head = nn.Conv2d(in_channels = 16, out_channels = num_class, kernel_size = 1)
        self.retain_dim = retain_dim
        self.out_sz = out_sz
        self.act = nn.Softmax(dim = 1)

    def forward(self, x):
        '''
        Forward loop that connects the encoder and decoder
        :param x: input
        :return:
        '''
        ft_maps = self.encoder(x)
        out = self.decoder(ft_maps[::-1][0], ft_maps[::-1][1:])
        out = self.act(self.head(out))

        if self.retain_dim:
            out = torch.nn.functional.interpolate(out, self.out_sz)
        return out

class Model:
    '''
    Object to handle model and related methods.
    '''

    ## NEED TO ADD THIS
    def __init__(self, model, loss, opt, scheduler, metrics, random_seed, train_data_loader, val_data_loader, test_data_loader, real_test_data_loader, device, base_loc = None, name = None, log_file=None):
        '''
        Scratch Model Wrapper
        :param model: model to train
        :param loss: loss function to use
        :param opt: optimizer
        :param scheduler: LR scheduler
        :param metrics: dictionary of metrics to calculate
        :param random_seed: random seed to set
        :param train_data_loader: dataloader for train data
        :param val_data_loader: dataloader for validation data
        :param test_data_loader: dataloader for testing data
        :param real_test_data_loader: dataloader for real testing data
        :param device: device for pytorch
        :param base_loc: base location to save to
        :param name: model name, used for saving and plottng
        :param log_file: logfile to output to
        '''
        self.log_file = log_file
        self.model = model.to(device)
        self.random_seed = random_seed
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.real_test_data_loader = real_test_data_loader
        self.name = name
        self.loss = loss
        self.opt = opt
        self.scheduler = scheduler
        self.history = {
            "train_loss":[],
            "val_loss":[],
            "test_loss":[]
        }
        for metric in metrics.keys():
            self.history[f'train_{metric}'] = []
            self.history[f'val_{metric}'] = []
            self.history[f'test_{metric}'] = []

        self.metrics = metrics
        self.base_loc = base_loc
        self.device = device

    def load(self):
        '''
        Loads saved model state if it exists
        :return:
        '''
        last_e = self.load_latest_model(self.device)
        return last_e

    def run_training(self, n_epochs, save_on = 'val_IOU', load = False):
        '''
        Runs the training loop for scratch model
        :param n_epochs: number of epochs to train
        :param save_on: what metric to keep track of to save on
        :param load: whether to load a saved model version. If True it will load and begin from last saved epoch, if False it will start from scratch
        :return:
        '''


        ### SET UP
        print(f'Training: {self.name}')
        num_training_steps = n_epochs * len(self.train_data_loader)
        progress_bar = tqdm(range(num_training_steps))
        lr_scheduler = get_scheduler(name = "linear", optimizer = self.opt, num_warmup_steps = 0, num_training_steps = num_training_steps)

        if load:
            last_e = self.load_latest_model(self.device)
        else:
            last_e = 0

        if save_on not in self.history.keys():
            print('that save metric doesnt exist, make sure the metric is passed into the function')
            return


        best_met = 0
        for e in range(last_e, n_epochs):
            ## Start epoch
            # reset metrics each epoch
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
                x_train, y_train = batch[0].to(self.device), batch[1].to(self.device)

                x_train.requires_grad = True

                self.model.zero_grad()
                self.opt.zero_grad()

                pred = self.model.forward(x_train.float())
                loss = self.loss(pred, y_train.float())
                loss.backward()
                self.opt.step()
                lr_scheduler.step()

                pred_argmax = torch.argmax(pred, dim = 1)

                # saving loss and metrics
                running_metrics['running_train_loss'] += loss.item()
                for metric in self.metrics.keys():
                    m = self.metrics[metric]
                    running_metrics[f'running_train_{metric}'] += m(pred_argmax.cpu(), torch.argmax(y_train.float(), dim = 1).cpu())

                progress_bar.update(1)

            # calculating average loss and metrics
            self.history["train_loss"].append((e, (running_metrics['running_train_loss']/len(self.train_data_loader))))
            for metric in self.metrics.keys():
                self.history[f'train_{metric}'].append((e, (running_metrics[f'running_train_{metric}'] / len(self.train_data_loader)).numpy()+0))

            ## VALIDATION LOOP
            self.model.eval()
            with torch.no_grad():
                for step, batch in enumerate(self.val_data_loader):
                    x_val, y_val = batch[0].to(self.device), batch[1].to(self.device)


                    y_val_pred = self.model(x_val.float())
                    loss = self.loss(y_val_pred, y_val.float())


                    running_metrics['running_val_loss'] += loss.item()
                    for metric in self.metrics.keys():
                        m = self.metrics[metric]
                        running_metrics[f'running_val_{metric}'] += m(torch.argmax(y_val_pred.float(), dim = 1).cpu(), torch.argmax(y_val.float(), dim = 1).cpu())

            # Updating validation metrics
            self.history["val_loss"].append((e, (running_metrics['running_val_loss']/len(self.val_data_loader))))
            for metric in self.metrics.keys():
                self.history[f'val_{metric}'].append((e, (running_metrics[f'running_val_{metric}'] / len(self.val_data_loader)).numpy()+0))

            s = f"EPOCH: {e} -- "
            for metric in self.history.keys():
                if len(self.history[metric])>0:
                    s += f"{metric} {self.history[metric][-1][1]} "

            print(s)

            if self.history[save_on][-1][1] > best_met:
                self.save_model(e)
                best_met = self.history[save_on][-1][1]

            # Measure how long this epoch took.
            print("")
            training_time = str(dt.timedelta(seconds = int(round((time.time() - t0)))))
            print(f"Training epoch took: {training_time}")


    def run_test(self):
        '''
        Run model on testing data
        :return:
        '''
        running_metrics = {
            'running_test_loss':0
        }

        for metric in self.metrics.keys():
            running_metrics[f'running_test_{metric}'] = 0
            #running_metrics[f'running_test_{metric}_real'] = 0

        num_training_steps = len(self.test_data_loader)
        #num_training_steps2 = len(self.real_test_data_loader)

        progress_bar = tqdm(range(num_training_steps), desc = 'TESTING: ')
        #progress_bar2 = tqdm(range(num_training_steps2), desc = 'TESTING REAL: ')


        self.model.eval()
        with torch.no_grad():
            for step, batch in enumerate(self.test_data_loader):

                x_test, y_test = batch[0].to(self.device), batch[1].to(self.device)


                y_test_pred = self.model(x_test.float())

                loss = self.loss(y_test_pred, y_test.float())

                running_metrics['running_test_loss'] += loss.item()
                for metric in self.metrics.keys():
                    m = self.metrics[metric]
                    running_metrics[f'running_test_{metric}'] += m(torch.argmax(y_test_pred.float(), dim = 1).cpu(), torch.argmax(y_test.float(), dim = 1).cpu())
                progress_bar.update(1)

        s = f"TESTING: "
        for metric in self.metrics.keys():
            s += f"{metric} {running_metrics[f'running_test_{metric}']/len(self.test_data_loader)} "

        self.history[f'test_loss'].append((-1, (running_metrics[f'running_test_loss'] / len(self.test_data_loader))))

        for metric in self.metrics.keys():
            self.history[f'test_{metric}'].append((-1, (running_metrics[f'running_test_{metric}'] / len(self.test_data_loader)).numpy()+0))
        print(s)


    def predict(self, img):
        '''
        Predicting function
        :param img: img to predict
        :return:
        '''
        x_test = img.to(self.device)
        y_pred = self.model(x_test.float())
        return torch.argmax(y_pred.float(), dim = 1).cpu()


    def plot_train(self, save_loc):
        '''
        plot training curves
        :param save_loc:
        :return:
        '''
        metrics = self.metrics.keys()
        metrics = np.unique([m[m.find('_')+1:] for m in metrics])
        fig, axes = plt.subplots(nrows = 1, ncols = len(metrics), figsize = (8,8))

        axes[0].plot([x[1] for x in self.history['train_loss']], color = "slategrey", label = "Training Loss")
        axes[0].plot([x[1] for x in self.history['val_loss']], color = "seagreen", label = "Training Loss")
        axes[0].legend()
        axes[0].set_title(f'MODEL: {self.name} LOSS')

        for i, ax in enumerate(axes[1:]):
            ax.plot([x[1] for x in self.history[f'train_{metrics[i]}']], color = "slategrey", label = f"Training {metrics[i]}")
            ax.plot([x[1] for x in self.history[f'val_{metrics[i]}']], color = "seagreen", label = f"Training {metrics[i]}")
            ax.legend()
            ax.set_title(f'MODEL: {self.name} {metrics[i]}')
        sns.despine()
        fig.savefig(os.path.join(save_loc, f'{self.name}_training_curves'))

    def save_model(self, epoch):
        '''
        save model to model folder
        :param epoch: what epoch it's saving on
        :return:
        '''
        save_loc = os.path.join(self.base_loc, 'Models', 'lunar_surface_segmentation_models')
        if not os.path.exists(save_loc):
            print('Making Model Dir')
            os.mkdir(save_loc)
        torch.save(self.model.state_dict(), os.path.join(save_loc, f"model_{self.name}_EP{epoch}.pt"))
        print('saving model ...')

    def load_latest_model(self, device):
        '''
        load the latest saved model
        :param device: pytorch device
        :return: the latest epoch the model was trained
        '''
        model_loc = os.path.join(self.base_loc, 'Models', 'lunar_surface_segmentation_models')
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
    '''
    Model wrapper for U-Net with pretrained model backbone from segmentation-models-pytorch
    '''

    def __init__(self, backbone, encoder_weights, activation, metrics, LR, loss, device, train_data_loader, val_data_loader, test_data_loader, real_test_data_loader, base_loc, name = None):
        '''
        init for pretrained model wrapper
        :param backbone: backbone to use ex: 'resnet18'
        :param encoder_weights: weights to use ex: 'imagenet'
        :param activation: activation function to use, None for lienar
        :param metrics: list of metrics to calculate
        :param LR: Learning rate
        :param loss: loss function
        :param device: pytorch device
        :param train_data_loader: dataloader for train
        :param val_data_loader: dataloader for validation
        :param test_data_loader: dataloader for testing
        :param real_test_data_loader: dataloader for real testing images
        :param base_loc: base location of code
        :param name: model name for saving
        '''
        self.backbone = backbone
        self.encoder_weights = encoder_weights
        self.activation = activation

        self.LR = LR
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.real_test_data_loader = real_test_data_loader

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

        self.optimizer = SGD(params=self.model.parameters(), lr=self.LR, momentum = 0.9)

        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(self.backbone, self.encoder_weights)

        self.metrics = metrics
        self.history = {}

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

    def run_training(self, n_epochs, load = False):
        '''
        Training loop
        :param n_epochs: number of epochs to train for
        :param load: whether to load a previous model state
        :return:
        '''
        print(f"Training: {self.name}")
        best_val_iou = 0.0
        train_logs_list, valid_logs_list = [], []

        if load:
            last_e = self.load_latest_model(self.device)
            print(f'Picking up from epoch: {last_e}')
        else:
            last_e = 0

        for i in range(last_e, n_epochs):
            # Perform training & validation
            print('\nEpoch: {}'.format(i))
            train_logs = self.train_epoch.run(self.train_data_loader)
            print(train_logs)
            val_logs = self.valid_epoch.run(self.val_data_loader)
            print(val_logs)
            for key in train_logs.keys():
                if f'train_{key}' in self.history.keys():
                    self.history[f'train_{key}'].append((i, train_logs[key]))
                else:
                    self.history[f'train_{key}'] = [(i, train_logs[key])]
            for key in val_logs.keys():
                if f'val_{key}' in self.history.keys():
                    self.history[f'val_{key}'].append((i, val_logs[key]))
                else:
                    self.history[f'val_{key}'] = [(i, val_logs[key])]

            if self.history[f'val_iou_score'][-1][1] > best_val_iou:
                best_val_iou = self.history[f'val_iou_score'][-1][1]
                self.save_model(i)

    def save_model(self, epoch):
        '''
        save model to model folder
        :param epoch: epoch number of best model
        :return:
        '''
        save_loc = os.path.join(self.base_loc, 'Models', 'lunar_surface_segmentation_models')
        if not os.path.exists(save_loc):
            print('Making Model Dir')
            os.mkdir(save_loc)
        torch.save(self.model.state_dict(), os.path.join(save_loc, f"model_{self.name}_EP{epoch}.pt"))
        print('saving model ...')

    def run_testing(self):
        '''
        test model on test dataset
        :return:
        '''
        test_epoch = smp.utils.train.ValidEpoch(
            model = self.model,
            loss = self.loss,
            metrics = self.metrics,
            device = self.device,
        )
        logs = test_epoch.run(self.test_data_loader)
        s = f"TESTING: "
        for key in logs.keys():
            if key in self.history.keys():
                self.history[f'test_{key}'].append((-1, logs[key]))
            else:
                self.history[f'test_{key}'] = [(-1, logs[key])]
            s += f' {key}: {self.history[f"test_{key}"][-1][1]}'

        print(s)



    def load(self):
        '''
        load latest model
        :return: latest epoch of training
        '''
        last_e = self.load_latest_model(self.device)
        return last_e

    def load_latest_model(self, device):
        '''
        load latest model
        :param device: pytorch device
        :return: last epoch of training
        '''
        model_loc = os.path.join(self.base_loc, 'Models', 'lunar_surface_segmentation_models')
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