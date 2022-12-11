"""
results_viz.py
Script to plot results

author: @saharae, @justjoshtings
created: 12/09/2022
"""
import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from LunarModules.Model import *

CODE_PATH = os.getcwd()
os.chdir('..')
BASE_PATH = os.getcwd()
os.chdir(CODE_PATH)
plots_path = os.path.join(BASE_PATH, 'Results')
results_path = os.path.join(BASE_PATH, 'Results/RESULTS.csv')

res = pd.read_csv(results_path)

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (8,6))
loss = res[res.metric.isin(['train_loss', 'val_loss'])]
pal = {'train_loss': 'cornflowerblue', 'val_loss': 'salmon'}
sns.lineplot(data = loss, x = 'epoch', y = 'value', hue = 'metric', palette = pal)
ax.set_title('Custom U-Net Loss')
ax.set_ylabel('Loss')
sns.despine()
plt.savefig(plots_path + '/custom_loss')
plt.show()

fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (8,5))
lossvgg = res[res.metric.isin(['train_cross_entropy_loss', 'val_cross_entropy_loss']) & (res.model_name == 'VGG11_BN_ground')]
pal = {'train_cross_entropy_loss': 'cornflowerblue', 'val_cross_entropy_loss': 'salmon'}
sns.lineplot(data = lossvgg, x = 'epoch', y = 'value', hue = 'metric', ax = axes[0], palette = pal)
pal = {'train_iou_score': 'cornflowerblue', 'val_iou_score': 'salmon'}
iouvgg = res[res.metric.isin(['train_iou_score', 'val_iou_score']) & (res.model_name == 'VGG11_BN_ground')]
sns.lineplot(data = iouvgg, x = 'epoch', y = 'value', hue = 'metric', ax = axes[1], palette = pal)
axes[0].set_title('VGG11 Loss')
axes[1].set_title('VGG11 IoU')
axes[0].set_ylabel('Loss')
axes[1].set_ylabel('IoU Score')
sns.despine()
plt.tight_layout()
plt.savefig(plots_path + '/vgg_stats')
plt.show()

fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (8,5))
lossvgg = res[res.metric.isin(['train_cross_entropy_loss', 'val_cross_entropy_loss']) & (res.model_name == 'RESNET18_ground')]
pal = {'train_cross_entropy_loss': 'cornflowerblue', 'val_cross_entropy_loss': 'salmon'}
sns.lineplot(data = lossvgg, x = 'epoch', y = 'value', hue = 'metric', ax = axes[0], palette = pal)
pal = {'train_iou_score': 'cornflowerblue', 'val_iou_score': 'salmon'}
iouvgg = res[res.metric.isin(['train_iou_score', 'val_iou_score']) & (res.model_name == 'RESNET18_ground')]
sns.lineplot(data = iouvgg, x = 'epoch', y = 'value', hue = 'metric', ax = axes[1], palette = pal)
axes[0].set_title('ResNet Loss')
axes[1].set_title('ResNet IoU')
axes[0].set_ylabel('Loss')
axes[1].set_ylabel('IoU Score')
sns.despine()
plt.tight_layout()
plt.savefig(plots_path + '/resnet_stats')
plt.show()

fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (8,5))
lossvgg = res[res.metric.isin(['train_cross_entropy_loss', 'val_cross_entropy_loss']) & (res.model_name == 'mobilenetv3_large_100_ground')]
pal = {'train_cross_entropy_loss': 'cornflowerblue', 'val_cross_entropy_loss': 'salmon'}
sns.lineplot(data = lossvgg, x = 'epoch', y = 'value', hue = 'metric', ax = axes[0], palette = pal)
pal = {'train_iou_score': 'cornflowerblue', 'val_iou_score': 'salmon'}
iouvgg = res[res.metric.isin(['train_iou_score', 'val_iou_score']) & (res.model_name == 'mobilenetv3_large_100_ground')]
sns.lineplot(data = iouvgg, x = 'epoch', y = 'value', hue = 'metric', ax = axes[1], palette = pal)
axes[0].set_title('MobileNet Loss')
axes[1].set_title('MobileNet IoU')
axes[0].set_ylabel('Loss')
axes[1].set_ylabel('IoU Score')
sns.despine()
plt.tight_layout()
plt.savefig(plots_path + '/mobilenet_stats')
plt.show()

iou = res[res.metric.isin(['train_IOU', 'val_IOU'])]
pal = {'train_IOU': 'cornflowerblue', 'val_IOU': 'salmon'}
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (8,6))
sns.lineplot(data = iou, x = 'epoch', y = 'value', hue = 'metric', palette = pal)
ax.set_title('Custom U-Net IoU')
ax.set_ylabel('IoU Score')
ax.set_ylim(0.1, 0.4)
sns.despine()
plt.savefig(plots_path + '/custom_iou')
plt.show()


res2 = res.replace(to_replace = 'train_cross_entropy_loss', value = 'train_loss')
res2 = res2.replace(to_replace = 'val_cross_entropy_loss', value = 'val_loss')
res2 = res2.replace(to_replace = 'train_iou_score', value = 'train_IOU')
res2 = res2.replace(to_replace = 'val_iou_score', value = 'val_IOU')

loss_t = res2[res2.metric.isin(['train_loss', 'val_loss'])]
h = sns.relplot(data = loss_t, x = 'epoch', y = 'value', hue = 'model_name', col = 'metric', kind = 'line')
h.fig.savefig(plots_path + '/training_compare')
plt.show()

iou_t = res2[res2.metric.isin(['train_IOU', 'val_IOU'])]
h = sns.relplot(data = iou_t, x = 'epoch', y = 'value', hue = 'model_name', col = 'metric', kind = 'line')
h.fig.savefig(plots_path + '/iou_compare')
plt.show()


## TEST
res2 = res2.replace('test_iou_score', 'test_IOU')
test = res2[res2.metric.isin(['test_IOU'])]
test = pd.concat([test, pd.DataFrame([['RANDOM', -1, 'test_IOU', 0.09]], columns = ['model_name', 'epoch', 'metric', 'value'])])
test.drop_duplicates(subset = 'model_name', keep = 'last', inplace = True)

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (6,4))
colors = sns.color_palette('Greens', n_colors = 4)
colors.insert(0, 'slategrey')
sns.barplot(data = test, x = 'model_name', y = 'value', order = test.sort_values(by = 'value').model_name,ax = ax, palette = colors)
ax.set_ylim(0,0.9)
ax.set_title("Test IoU By Model")
ax.set_xlabel('Model')
ax.set_ylabel("IoU Score")
ax.set_xticklabels(['Random', 'Custom U-Net', 'MobileNet', 'ResNet', 'VGG'])
sns.despine()
fig.savefig(plots_path + '/test_compare')
plt.show()

## REAL IMAGES DATA
real_data_results = pd.read_csv(os.path.join(BASE_PATH, 'Results/real_data_results.csv'), index_col = 0)
real_data_results = pd.concat([real_data_results, pd.DataFrame([['RANDOM', -1, 'test_IOU', 0.09]], columns = ['model_name', 'epoch', 'metric', 'value'])])

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (6,4))
colors = sns.color_palette('Reds', n_colors = 4)
colors.insert(0, 'slategrey')
sns.barplot(data = real_data_results, x = 'model_name', errorbar = None, y = 'value',ax = ax,palette = colors, order = ['RANDOM', 'mobilenetv3_large_100_ground', 'RESNET18_ground', 'VGG11_BN_ground', 'Unet_scratch_ground'])
ax.set_ylim(0,0.6)
ax.set_title("Real Test IoU By Model")
ax.set_xlabel('Model')
ax.set_ylabel("IoU Score")
ax.set_xticklabels(['Random', 'MobileNet','ResNet', 'VGG', 'Custom U-Net'])
sns.despine()
fig.savefig(plots_path + '/real_test_compare')
plt.show()
