import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

os.environ['KAGGLE_USERNAME'] = "saharaensley"
os.environ['KAGGLE_KEY'] = "14b76f6d948ee90f7856dec349c37c6f"
#getting paths
CODE_PATH = os.getcwd()
os.chdir('..')
BASE_PATH = os.getcwd()
os.chdir(CODE_PATH)
DATA_PATH = os.path.join(BASE_PATH, 'Data')
#DATA_PATH = '/Users/sahara/Documents/GW/ML2/Final-Project-GroupX/sahara-ensley-individual-project/Data' # for testing

#downloading data
if not os.path.exists(DATA_PATH):
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files('romainpessia/artificial-lunar-rocky-landscape-dataset', path=DATA_PATH, unzip=True)
    print('done')
else:
    print("data already exists")

bounding = pd.read_csv(os.path.join(DATA_PATH, 'bounding_boxes.csv'))
imgs = os.listdir(os.path.join(DATA_PATH, 'images', 'clean'))

dat = []
#[R, G, B], black = [0,0,0]
if os.path.exists('images_summary.csv'):
    print('alraedy collected data')
else:
    print('doing image stuff')
    for img_path in imgs:
        img = plt.imread(os.path.join(DATA_PATH, 'images', 'clean', img_path))
        sums = np.sum(img, axis = 2)
        idx = np.argmax(img, axis = 2)
        blues = len(np.where(idx == 2)[0])
        greens = len(np.where(idx == 1)[0])
        reds = len(np.where(np.logical_and(idx==0, sums>0))[0])
        blacks = len(np.where(np.logical_and(idx==0, sums==0))[0])
        dat.append([img_path, reds, greens, blues, blacks])

    df = pd.DataFrame(dat, columns = ['image', 'reds', 'greens', 'blues', 'blacks'])
    df.to_csv('images_summary.csv')

df = pd.read_csv("images_summary.csv")