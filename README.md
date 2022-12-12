# Final-Project-Group5
## George Washington University, Machine Learning II - DATS203_10, Fall 2022

## Project
Lunar Landscape Imagery Segmentation 

![sample_diagram](https://github.com/justjoshtings/Final-Project-Group5/blob/main/Code/plots/render7871.png)

## Table of Contents
1. [Team Members](#team_members)
2. [How to Run](#instructions)
3. [Folder Structure](#structure)
2. [Timeline](#timeline)
3. [Topic Proposal](#topic_proposal)
4. [Datasets](#datasets)
5. [Presentation](#presentation)
6. [Report](#report)
7. [References](#references)
8. [Licensing](#license)

# <a name="team_members"></a>
## Team Members
* [Sahara Ensley](https://github.com/Saharae)
* [Joshua Ting](https://github.com/justjoshtings)

# <a name="instructions"></a>
## How to Run
1. Clone this repo
2. Install python packages. After cloning the repo and download python packages.
    ```
    cd Final-Project-Group5/Code/
    pip install -r requirements.txt
    ```
4. Execute Main Script with options...

   **Test (~10 minutes):** This will skip any of the training and run the testing loops with the models downloaded from Google 
   Drive
   ```
   python3 main.py --method 'test'
   ```
   
   **Train (~10 hours):** This will run the full training loops, overwriting the downloaded Models (if any) and then test the 
   results.
   ```
   python3 main.py --method 'train'
   ```
   
   **Debug (~10 hours):** This will run the training loops along with any debugging code, this includes checks for the data 
   loaders and 
   plotting outputs between models in addition to at the end of the loops.
   ```
   python3 main.py --method 'debug'
   ```
   
   **EDA (additional 10+ minutes):** Running with EDA set to True will run the EDA python script before any modeling code, 
   this will allow the EDA notebook to be executed without errors. If you don't want to execute the EDA notebook then 
   this argument should be left out as the default is False.
   ```
   python3 main.py --method 'test' --EDA True
   ```

# <a name="structure"></a>
## Folder Structure
```
.
├── Code                                # Final code for the project, navigate here to run.
│   ├── LunarModules                    # Modules to support codebase
│   ├── plots                           # Plots folder to save plots
├── Final-Group-Presentation            # Presentation Slides PDF
├── Final-Group-Presentation            # Final Report
├── Group-Proposal                      # Group Proposal Report
├── joshua-ting-individual-project      # Individual report - Josh
├── sahara-ensley-individual-project    # Individual report - Sahara
├── Results                             # This folder contains results from the models we tuned. The GUI pulls from this folder.
│ 
└── requirements.txt        # Python package requirements
```

# <a name="timeline"></a>
## Timeline
- [X] Proposal - 11/8/2022
- [X] Environment Setup - 11/8/2022
- [X] EDA - 11/11/2022
- [X] Start Model Training - 11/18/2022
- [X] Final Model and Results - 12/02/2022
- [X] Google Drive Models Download
- [X] Google Drive Data Download
- [X] Main Script with option to run saved model or train from scratch
- [X] Freeze requirements.txt
- [X] Finalize README
- [X] Test On Clean EC2
- [X] Final Report - 12/12/2022
- [X] Individual Reports/Code
- [X] Final Presentation - 12/12/2022

# <a name="topic_proposal"></a>
## Topic Proposal
* [Topic Proposal Google Doc](https://docs.google.com/document/d/1gTb3xTB7aXJ7cCjL_SwqE0ElDZd4y_njZNcG0bAr5q8/edit?usp=sharing)

# <a name="datasets"></a>
## Datasets
* [Kaggle Artificial Lunar Landscape Dataset](https://www.kaggle.com/datasets/romainpessia/artificial-lunar-rocky-landscape-dataset)

# <a name="presentation"></a>
## Presentation
* [Google Slides Presentation](https://docs.google.com/presentation/d/1N0azL_rzTkx4bbQPJFXbIkvIjRVbqXGzX1lXuviBQzU/edit?usp=sharing)

# <a name="report"></a>
## Report
* [Final Report Google Doc](https://docs.google.com/document/d/1w5YAu1uEHxkzkeqVPvH7H5MKZHtm0U8PbnMYADZHXp4/edit?usp=sharing)

# <a name="references"></a>
## References
* [Jonathan Long et. al (2014) - Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)
* [Ronneberger et. al (2015) - UNet: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597v1)
*[Artificial Lunar Landscape Dataset on Kaggle](https://www.kaggle.com/datasets/romainpessia/artificial-lunar-rocky-landscape-dataset)
* [Lunar Surface Image - thespaceacademy.org](http://www.thespaceacademy.org/2017/10/here-is-your-best-chance-to-explore.html)
* [An Overview of Semantic Segmentation](https://www.jeremyjordan.me/semantic-segmentation/)
* [Stanford CS231: Detection and Segmentation](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture11.pdf)
* [Kaggle - Artificial Lunar Landscape Dataset](https://www.kaggle.com/datasets/romainpessia/artificial-lunar-rocky-landscape-dataset)
* [Kaggle - Artificial Lunar Landscape Dataset - Silver Notebook](https://www.kaggle.com/code/basu369victor/transferlearning-and-unet-to-segment-rocks-on-moon)
* [Jaccard Index](https://deepai.org/machine-learning-glossary-and-terms/jaccard-index)
* [Understanding and Visualizing ResNets](https://towardsdatascience.com/understanding-and-visualizing-resnets-442284831be8)
* [Architecture and Implementation of VGG16](https://towardsai.net/p/machine-learning/the-architecture-and-implementation-of-vgg-16)
* [MobileNet v3](https://towardsdatascience.com/everything-you-need-to-know-about-mobilenetv3-and-its-comparison-with-previous-versions-a5d5e5a6eeaa)
* [Metrics to Evaluate Semantic Segmentation](https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2)
* [Cross Entropy Loss](https://medium.com/unpackai/cross-entropy-loss-in-ml-d9f22fc11fe0)

# <a name="license"></a>
## Licensing
* MIT License
* Dataset under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
