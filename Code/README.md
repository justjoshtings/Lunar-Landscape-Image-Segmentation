# Code

## Description
This directory holds all relevant code for data acquisition, preprocessing, model building/training, and evaluation.

# <a name="app-execution"></a>
## App Execution

After cloning the repo, navigate to the Code folder and set permissions for 4 bash scripts.
```
cd Final-Project-Group5/Code/
chmod u+x env_setup.sh
```
Below is the description of each script:
1. env_setup.sh - runs environment setup and installs needed software
2. ...

Next, you can either download data from Kaggle manually or setup Kaggle API credentials to download through a prepared script. See [data download](https://github.com/justjoshtings/Final-Project-Group5/blob/main/Code/README.md#data-download) section for more details on both options.

Next, run the env_setup.sh script.
```
cd Final-Project-Group5/Code/
./env_setup.sh
```

# <a name="data-download"></a>
## Data Distribution and Download

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
