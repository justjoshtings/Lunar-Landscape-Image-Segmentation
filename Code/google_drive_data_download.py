"""
google_drive_data_download.py
Script to download data from Google Drive link

author: @saharae, @justjoshtings
created: 12/09/2022
"""
import gdown
import os
import shutil
from zipfile import ZipFile

def download_data_gdrive(data_url="https://drive.google.com/drive/folders/1UrBuQcoW8i6hzPzpzVWnRyfqVtR2VfoD?usp=sharing"):
    '''
    Function to download data from Google Drive link
    '''

    '''
    Set paths
    '''
    CODE_PATH = os.getcwd()
    os.chdir('..')
    BASE_PATH = os.getcwd()
    os.chdir(CODE_PATH)
    DATA_PATH = os.path.join(BASE_PATH, 'Data')
    ZIP_PATH = os.path.join(BASE_PATH,'lunar_surface_image_data')

    if os.path.exists(DATA_PATH):
        shutil.rmtree(DATA_PATH)
    if os.path.exists(ZIP_PATH):
        shutil.rmtree(ZIP_PATH)

    '''
    Download 
    '''
    os.chdir(BASE_PATH)
    print('Downloading data...')
    gdown.download_folder(data_url, quiet=True, use_cookies=False)

    '''
    Unzip: https://www.geeksforgeeks.org/unzipping-files-in-python/
    '''
    print('Unzipping data...')
    with ZipFile(BASE_PATH+'/lunar_surfce_image_data/Data.zip', 'r') as zObject:
    
        # Extracting all of the zip into a specific location.
        zObject.extractall(path=BASE_PATH)


if __name__ == '__main__':
    print('Running google_drive_data_download.py')
    download_data_gdrive()