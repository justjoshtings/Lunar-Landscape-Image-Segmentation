from LunarModules.TrainTestSplit import *
from google_drive_data_download import *
from trained_model_dl import *
from modeling import *
from EDA import *
import os
import time
import argparse

if __name__ == '__main__':
    CODE_PATH = os.getcwd()
    os.chdir('..')
    BASE_PATH = os.getcwd()
    os.chdir(CODE_PATH)
    TRAINED_MODELS_PATH = os.path.join(BASE_PATH, 'Models')
    DATA_PATH = os.path.join(BASE_PATH, 'Data')
    SPLIT_DATA_PATH = os.path.join(DATA_PATH, 'images', 'train')

    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default = 'test', type = str, required = False)
    parser.add_argument('--EDA', default = False, type = bool, required = False)
    args = parser.parse_args()
    print('RUNNING WITH METHOD: ', args.method, ' EDA: ', args.EDA)

    TRAIN = False
    debug = False
    plot = False

    if args.method == 'test':
        TRAIN = False
        debug = False
        plot = True
    if args.method == 'train':
        TRAIN = True
        debug = False
        plot = True
    if args.method == 'debug':
        TRAIN = True
        debug = True
        plot = True

    # download data from google drive
    if not os.path.exists(DATA_PATH) or len([x for x in os.listdir(DATA_PATH) if x not in ['.DS_Store']]) == 0:
        data_t1 = time.time()
        print('DOWNLOADING DATA ....')
        download_data_gdrive()
        data_t2 = time.time()
        print('DOWNLOAD COMPLETE -- ', ((data_t2 - data_t1)/60), ' minutes')

    # download trained models from google drive
    if not os.path.exists(TRAINED_MODELS_PATH) or len(os.listdir(TRAINED_MODELS_PATH)) == 0:
        models_t1 = time.time()
        print('DOWNLOADING MODELS ....')
        download_trained_models()
        models_t2 = time.time()
        print('DOWNLOAD COMPLETE -- ', ((models_t2 - models_t1)/60), ' minutes')

    # do EDA
    os.chdir(CODE_PATH)
    if args.EDA:
        print('Running EDA script ....')
        eda_t1 = time.time()
        RUN_EDA()
        eda_t2 = time.time()
        print('EDA complete -- ', (eda_t2 - eda_t1)/60, ' minutes -- You can now run the EDA notebook if desired')

    # do traintestsplit
    if not os.path.exists(SPLIT_DATA_PATH):
        print('SPLITTING DATA ....')
        run_datasplit(SOURCE = 'ground' )
    # Run Modeling and Evaluation
    RUN_MODEL_LOOP(TRAIN = TRAIN, debug = debug, plot = plot)
    print("EXITING")


