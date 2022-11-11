"""
kaggle_download.py
Script to download Kaggle datasets

author: @saharae, @justjoshtings
created: 11/11/2022
"""

from LunarModules.KaggleAPI import KaggleAPI
import os

CODE_PATH = os.getcwd()
os.chdir('..')
BASE_PATH = os.getcwd()
os.chdir(CODE_PATH)
DATA_PATH = os.path.join(BASE_PATH, 'Data')

def main():
    kaggle_dataset_owner = 'romainpessia'
    path_to_data = DATA_PATH
    data_url_end_point = 'artificial-lunar-rocky-landscape-dataset'

    kaggle = KaggleAPI(kaggle_dataset_owner)

    # Download dataset
    kaggle.download_dataset(owner=kaggle_dataset_owner, data_url_end_point=data_url_end_point, path_to_data=path_to_data)

if __name__ == "__main__":
    print("Executing kaggle_download.py")
    main()