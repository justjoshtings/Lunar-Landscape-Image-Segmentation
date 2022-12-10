# LunarModules

## Description
All util code

Below is the description of each utility object:
1. CustomDataLoader.py - A custom built data loader to handle data generator.
2. ImageProcessor.py - Object to handle all processing of images/data.
3. KaggleAPI.py - Object to handle connection to Kaggle API to upload and download files.
4. Logger.py - Object to handle logging.
5. Model.py - Object to handle modeling methods.
6. Plotter.py - Object to handle all plotting of images/ground truth masks.
7. TrainTestSplit.py - Functions to correctly organize dataset.
8. utils.py - Utility functions to help with programs.

### TrainTestSplit.py

This script splits the data and copies all the images into
appropriate train/test/val folders for data training.

There are 2 ways to run this code.\
**Command Line**:
```bash
python3 TrainTestSplit.py
```
There are 2 arguments passed to the main function:\
*source*: This determines the source of the mask data. The two 
options are 'clean' and 'ground'. DEAFAULT = clean.\
*resplit*: If True then the existing train/test/val folders
will be deleted and the data re-split and copied. This should
be used when switching from clean->ground or vice versa.\
```bash
python3 TrainTestSplit.py --source 'ground' --resplit True
```
would delete the original split folders and resplit them using
the 'ground' images as the mask data.\
**In Code**:
```python3
from LunarModules.TrainTestSplit import *
run(SOURCE='clean', RESPLIT=True)
```
would perform the same action as running from the command line.