# Comparison of KNN to ResNet50 for pneumonia detection in X-rays:floppy_disk:

This project looks at comparing ResNet50 and KNN for classifying X-rays of pneumonia. 

## Projectstructure :clipboard:

![alt text](https://github.com/anderf2706/ML-/blob/main/Images/structure.png)

**Structure**

## Getting Started :checkered_flag:

To get started, download this repo either by cloning it from your IDE, or download it as a zip. 
The majority of methods can be found in ML.py and KNN_classifier.py.

The images used for training, should be put in the respective folders described in the readme-files found in the data folders.

## Running the system :rocket:

Both of the main py-files contain a main method at the bottom of the files, and can be run as is. 

## Methods :star:

methods from ML.py and KNN_classifier.py

ML.py:

### make_pred(Learner learn, string file):
Make predictions on a single image.

### make_and_train():
Creates the data used for the model. Then creates the model for the ResNet50, and trains it for 2-4-20-10 epochs. To save the model, uncomment the learn.save() line.

### interp():
creates a confusion matrix for the learner.

### load_model():
takes in the saved model from the folder specified in the train method, and loads it back, to be used without re-training. 

KNN_classifier.py:

### dataset():
Takes the images from the x-ray folder, and creates test, train and validation sets to be used for the model. 

### save_model(Trainer classifier):
saves the model to be used later.

### load_model():
loads the model saved by the save_model method to be used for classifyinfg 

### plot_best_K():
plots a graph showing what the best K-value is for the model. 



## Prerequisites :white_check_mark:
-------------------

from fastai.vision import *
import numpy as np 
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import torchvision
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
import matplotlib.pyplot as plt

## Authors :pencil2:

Anders Fredriksen
Marianne Pettersen
Elise Bø
