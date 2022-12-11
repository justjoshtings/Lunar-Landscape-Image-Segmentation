"""
TrainTestSplit.py
Functions to correctly organize dataset
author: @saharae, @justjoshtings
created: 11/18/2022
"""

import numpy as np
import os
from sklearn.model_selection import train_test_split
import pandas as pd
import shutil
import argparse

def get_data(DATA_PATH, SOURCE):
    '''
    Matches input with target by their id and returns dataframe to be split
    :param DATA_PATH: location of data
    :param SOURCE: source of train data ('clean' or 'ground')
    :return: pandas DF with input and target image names and id
    '''
    full_mask_path = os.path.join(DATA_PATH, 'images', SOURCE)
    full_img_path = os.path.join(DATA_PATH, 'images/render')

    all_imgs = pd.DataFrame(os.listdir(full_img_path), columns = ['img'])
    all_masks = pd.DataFrame(os.listdir(full_mask_path), columns = ['mask'])

    all_imgs['id'] = all_imgs.img.apply(lambda x: int(x[-8:-4]))
    all_masks['id'] = all_masks['mask'].apply(lambda x: int(x[-8:-4]))

    data = all_imgs.merge(all_masks, on = 'id')
    return data

def move_data(data, split, source, DATA_PATH):
    '''
    COPIES images in data to desired location
    :param data: dataframe containing image names
    :param split: 'train'/'test'/'val'
    :param source: 'clean'/'ground'
    :param DATA_PATH: location of data
    :return: none
    '''
    final_path = os.path.join(DATA_PATH, 'images', split)
    if not os.path.exists(os.path.join(final_path, 'render')):
        os.makedirs(os.path.join(final_path, 'render'))
    if not os.path.exists(os.path.join(final_path, 'mask')):
        os.makedirs(os.path.join(final_path, 'mask'))

    for img in data.img.to_numpy():
        src = os.path.join(DATA_PATH, 'images', 'render', img)
        dst = os.path.join(final_path, 'render', img)
        shutil.copy2(src, dst)
    for msk in data['mask'].to_numpy():
        src = os.path.join(DATA_PATH, 'images', source, msk)
        dst = os.path.join(final_path, 'mask', msk)
        shutil.copy2(src, dst)

def move_real_test_images(DATA_PATH):
    source_path = os.path.join(DATA_PATH, 'real_moon_images')
    final_path = os.path.join(DATA_PATH, 'images', 'real')
    if not os.path.exists(os.path.join(final_path, 'real_img')):
        os.makedirs(os.path.join(final_path, 'real_img'))
    if not os.path.exists(os.path.join(final_path, 'real_mask')):
        os.makedirs(os.path.join(final_path, 'real_mask'))

    list_real_images = os.listdir(source_path)
    list_real_images = [item for item in list_real_images if '.png' in item]

    list_real_masks = [item for item in list_real_images if item.startswith('g_')]
    list_real_images = [item for item in list_real_images if not item.startswith('g_')]

    # print(list_real_images, len(list_real_images))
    # print(list_real_masks, len(list_real_masks))

    for img in list_real_images:
        src = os.path.join(source_path, img)
        dst = os.path.join(final_path, 'real_img', img)
        shutil.copy2(src, dst)
    for msk in list_real_masks:
        src = os.path.join(source_path, msk)
        dst = os.path.join(final_path, 'real_mask', msk)
        shutil.copy2(src, dst)
    

def run_datasplit(SOURCE = 'clean', RESPLIT = False):
    '''
    main function that splits and copies data into correct folders
    :param SOURCE: source of training data 'clean' or 'ground'
    :param RESPLIT: bool value, if True the existing Train/Val/Test folders will be removed and recreated
                    ** should be used if switching from clean->ground (or vice versa), or random state is changed, etc
    :return: none
    '''
    BASE_PATH = os.getcwd()
    os.chdir('../../Data')
    DATA_PATH = os.getcwd()

    if RESPLIT:
        files = os.listdir(os.path.join(DATA_PATH, 'images', 'render'))
        if len(files) == 0:
            print('------- THERE ARE NO FILES IN RENDER\n------- Resplitting will delete the data. Move everything back to their original directories and run again.')
        print('removing directories to RESPLIT ...')
        try:
            shutil.rmtree(os.path.join(DATA_PATH, 'images', 'train'))
        except FileNotFoundError:
            pass
        try:
            shutil.rmtree(os.path.join(DATA_PATH, 'images', 'test'))
        except FileNotFoundError:
            pass
        try:
            shutil.rmtree(os.path.join(DATA_PATH, 'images', 'val'))
        except FileNotFoundError:
            pass
        try:
            shutil.rmtree(os.path.join(DATA_PATH, 'images', 'real'))
        except FileNotFoundError:
            pass

    if os.path.exists(os.path.join(DATA_PATH, 'images', 'train')):
        print('Data already split ... skipping')
    else:
        print('getting data ...')
        data = get_data(DATA_PATH, SOURCE)
        train, test = train_test_split(data, test_size = 0.3, random_state = 42)
        train, val = train_test_split(train, test_size = 0.3, random_state = 42)

        print('moving train ...')
        move_data(data = train, split = 'train', source = SOURCE, DATA_PATH = DATA_PATH)
        print('moving val ...')
        move_data(data = val, split = 'val', source = SOURCE, DATA_PATH = DATA_PATH)
        print('moving test ...')
        move_data(data = test, split = 'test', source = SOURCE, DATA_PATH = DATA_PATH)
        print('moving real moon images ...')
        move_real_test_images(DATA_PATH)
    
    print('done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default = 'clean', type=str, required = False)
    parser.add_argument('--resplit', default = False, type=bool, required = False)
    args = parser.parse_args()
    SOURCE = args.source
    RESPLIT = args.resplit
    print(f'SPLITTING DATA WITH source={SOURCE}, resplit={RESPLIT}')
    run_datasplit(SOURCE=SOURCE, RESPLIT=RESPLIT)
