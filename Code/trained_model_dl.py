"""
trained_model_dl.py
Script to download trained models from Google Drive link

author: @saharae, @justjoshtings
created: 12/09/2022
"""
import gdown
import os

def download_trained_models(trained_models_url="https://drive.google.com/drive/folders/1x8qjoZVuTyvvi7FqkTLpNOimdBWLgapF?usp=sharing"):
    '''
    Function to download trained models from Google Drive link
    '''

    '''
    Set paths
    '''
    CODE_PATH = os.getcwd()
    os.chdir('..')
    BASE_PATH = os.getcwd()
    os.chdir(CODE_PATH)
    TRAINED_MODELS_PATH = os.path.join(BASE_PATH, 'Models')

    if not os.path.exists(TRAINED_MODELS_PATH):
            os.makedirs(TRAINED_MODELS_PATH)

    for root, dirs, files in os.walk(os.path.join(TRAINED_MODELS_PATH, 'lunar_surface_segmentation_models')):
        for file in files:
            os.remove(os.path.join(root, file))

    '''
    Download 
    '''
    os.chdir(TRAINED_MODELS_PATH)
    gdown.download_folder(trained_models_url, quiet=True, use_cookies=False)


if __name__ == '__main__':
    print('Running trained_model_dl.py')
    download_trained_models()