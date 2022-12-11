import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def RUN_EDA():
    #getting paths
    CODE_PATH = os.getcwd()
    os.chdir('..')
    BASE_PATH = os.getcwd()
    os.chdir(CODE_PATH)
    DATA_PATH = os.path.join(BASE_PATH, 'Data')

    imgs = os.listdir(os.path.join(DATA_PATH, 'images', 'clean'))

    dat = []

    #[R, G, B], black = [0,0,0]
    if os.path.exists(os.path.join(DATA_PATH,'images_summary.csv')):
        print('already image collected data')
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
        df.to_csv(os.path.join(DATA_PATH,'images_summary.csv'))


    print('making heatmaps')
    red_mat = np.zeros((480,720))
    green_mat = np.zeros((480,720))
    blue_mat = np.zeros((480,720))
    black_mat = np.zeros((480,720))

    for img_path in imgs:
        img = plt.imread(os.path.join(DATA_PATH, 'images', 'clean', img_path))
        sums = np.sum(img, axis = 2)
        idx = np.argmax(img, axis = 2)
        blues = np.where(idx == 2)
        greens = np.where(idx == 1)
        reds = np.where(np.logical_and(idx==0, sums>0))
        blacks = np.where(np.logical_and(idx==0, sums==0))
        red_mat[reds]+=1
        green_mat[greens]+=1
        blue_mat[blues]+=1
        black_mat[blacks]+=1

    np.save(os.path.join(DATA_PATH,'blue_dat.npy'), blue_mat)
    np.save(os.path.join(DATA_PATH,'red_dat.npy'), red_mat)
    np.save(os.path.join(DATA_PATH,'green_dat.npy'), green_mat)
    np.save(os.path.join(DATA_PATH,'black_dat.npy'), black_mat)

if __name__ == '__main__':
    print('running EDA')
    RUN_EDA()