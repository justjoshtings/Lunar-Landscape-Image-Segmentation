#!/bin/bash

echo "Project Setup for Lunar Landscape Segmentation"

# Ensure python3 and pip are installed
# sudo apt update
# sudo apt install -y python3-pip

# Clone project
cd ~/
git clone https://github.com/justjoshtings/Final-Project-Group5.git

cd ./Final-Project-Group5

# sudo apt -y install python3.8-venv
# python3 -m venv ./myenv/
# source myenv/bin/activate

# Install python requirements
# pip3 install -r requirements.txt
pip3 install kaggle

# Make Log File
mkdir ./Lunar_Log/
cd ./Lunar_Log/
touch lunar.log
cd ../Code/

# Set up Kaggle API
echo "Setup Kaggle API and download kaggle.json"

FILE=~/.kaggle/kaggle.json
echo "Checking if kaggle.json exists in: $FILE"

if test -f "$FILE"; then
    echo "$FILE exists."
    chmod 600 ~/.kaggle/kaggle.json

    echo "Testing kaggle API, running 'kaggle competitions list'"
    kaggle competitions list
else 
    echo "Set up Kaggle API with the following resources and download kaggle.json to ~/.kaggle/kaggle.json"
    echo "https://adityashrm21.github.io/Setting-Up-Kaggle/"
    echo "https://github.com/Kaggle/kaggle-api#api-credentials"
    echo "Or ignore and download manually instead. Check README data-download section for more."
fi