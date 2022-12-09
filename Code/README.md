# Code

## Description
This directory holds all relevant code for data acquisition, preprocessing, model building/training, and evaluation.

Below is the description of each script:
1. main.py - Executes main script.
2. env_setup.sh - runs environment setup and installs needed software
3. EDA.py - Runs the preprocessing and collection of meta data of the images in order to plot EDA images.
4. EDA_figures.ipynb - Jupyter notebook with EDA figures.
5. kaggle_download.py - Downloads data from Kaggle.
6. google_drive_data_download.py - Downloadas data from our own distribution on Google Drive.
7. modeling.py - Executes modeling.
8. results_viz.py - Script to plot results.
9. trained_model_dl.py - Script to download trained models from Google Drive.
10. LunarModules/CustomDataLoader.py - A custom built data loader to handle data generator.
11. LunarModules/ImageProcessor.py - Object to handle all processing of images/data.
12. LunarModules/KaggleAPI.py - Object to handle connection to Kaggle API to upload and download files.
13. LunarModules/Logger.py - Object to handle logging.
14. LunarModules/Model.py - Object to handle modeling methods.
15. LunarModules/Plotter.py - Object to handle all plotting of images/ground truth masks.
16. LunarModules/TrainTestSplit.py - Functions to correctly organize dataset.
17. LunarModules/utils.py - Utility functions to help with programs.


# <a name="app-execution"></a>
## App Execution

## Main Script

The main script carries out the following subroutines. Execute Main Script with options...
Test
```
python3 main --method test
```

Train
```
python3 main --method train
```

Debug
```
python3 main --method debug
```

## Subroutines
The following subroutines are executed within the main script but can also be executed individually using the following execution commands.

### Data Download 
Download from Kaggle the dataset.
```
cd Final-Project-Group5/Code/
python3 kaggle_download.py
```

or from Google Drive
```
cd Final-Project-Group5/Code/
python3 google_drive_data_download.py
```

### Trained Models Download 
Download from trained models from Google Drive.
```
cd Final-Project-Group5/Code/
python3 trained_model_dl.py
```

### EDA 

Make sure all previous steps are completed first (data download/env setup)
1. Run the EDA python script first (this only needs to be run once). It will take a few minutes.
```
cd Final-Project-Group5/Code/
python3 EDA.py
```

2. You can now run and edit the Jupyter notebook as desired. Make sure you're in the directory with the notebook 
   then run
```
cd Final-Project-Group5/Code/
jupyter notebook
```

### Split Data
```
cd Final-Project-Group5/Code/LunarModules
python3 TrainTestSplit.py
```

### Modeling and Evaluation 
Modeling and evaluation can be executed using:
```
cd Final-Project-Group5/Code/
python3 modeling.py
```

# <a name="data-download"></a>
## Data Distribution and Download - Old/Initial Method
After cloning the repo, navigate to the Code folder and set permissions for the following bash script.
```
cd Final-Project-Group5/Code/
chmod u+x env_setup.sh
```

Next, you can either download data from Kaggle manually or setup Kaggle API credentials to download through a prepared script. See [data download](https://github.com/justjoshtings/Final-Project-Group5/blob/main/Code/README.md#data-download) section for more details on both options.

Next, run the env_setup.sh script.
```
cd Final-Project-Group5/Code/
./env_setup.sh
```

Data can be accessed publicly on [Kaggle](https://www.kaggle.com/datasets/romainpessia/artificial-lunar-rocky-landscape-dataset). 

**Two options to download data:**

**Option 1:** Manual Download
1. Manually download, unzip, and move all contents within **archive** folder to **FINAL-PROJECT-GROUP5/Data/**. You will need to create the data directory. You can use scp to move files from local machine to a remote machine if needed.

**Option 2:** Use Kaggle API to download data
1. Make .kaggle directory
```
mkdir ~/.kaggle/
```
2. Create a Kaggle account API. See [here](https://github.com/Kaggle/kaggle-api#api-credentials) or [here](https://adityashrm21.github.io/Setting-Up-Kaggle/).
3. Download the kaggle.json file of your API credentials and save to **~/.kaggle/kaggle.json**
```
mv [downloaded kaggle.json path] ~/.kaggle/kaggle.json
```
ie: 
```
mv /home/ubuntu/Final-Project-Group5/Code/kaggle.json ~/.kaggle/kaggle.json
```
4. Set permissions.
```
chmod 600 ~/.kaggle/kaggle.json
```

