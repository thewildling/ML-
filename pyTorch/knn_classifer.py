######### Imports ############
import pickle

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import *
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import torchvision
import numpy as np
import torchvision.models as models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
import matplotlib.pyplot as plt
from pyTorch.utils import ROOT_DIR
####################################


dataset_root = ROOT_DIR + '\\chest_xray\\' # folder with images
new_size = (224,224) # new size of images.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # string so thet pyTorch uses cuda for data preparation.

labels = ['NORMAL', 'PNEUMONIA'] # labels for the images

def dataset(Normalized): # Take in the datasets from folders, and make them into tensors, so they can be used for training.
    transform = transforms.Compose([
        transforms.Resize(new_size), # resizing the images
        transforms.Grayscale(), # transforing them to grey-scale as they are black/white
        transforms.ToTensor(), # toTensor so it can be used as arrays for
        nn.Flatten() # flatten into 1 dim
    ])
    if(Normalized):
        train = torchvision.datasets.ImageFolder(root=dataset_root + 'train_normalized',
                                                 transform=transform)  # Make an image folder for train, using the transforms from above
    else:
        train = torchvision.datasets.ImageFolder(root=dataset_root + 'train',
                                                 transform=transform)  # Make an image folder for train, using the transforms from above

    test = torchvision.datasets.ImageFolder(root=dataset_root + 'test', transform=transform) # make an image folder for test

    val = torchvision.datasets.ImageFolder(root=dataset_root + 'val', transform=transform) # image folder for validation

    train_loader = torch.utils.data.DataLoader(train, batch_size=len(train), shuffle=True) # Make a dataloader object for train, where we set the batch to the whole set. Then
    #use shuffle to shuffle the images.

    val_loader = torch.utils.data.DataLoader(val, batch_size=len(val), shuffle=True)# Make a dataloader object for validation, where we set the batch to the whole set. Then
    #use shuffle to shuffle the images.

    test_loader = torch.utils.data.DataLoader(test, batch_size=len(test), shuffle=True)# Make a dataloader object for test, where we set the batch to the whole set. Then
    #use shuffle to shuffle the images.


    #convert the data in the sets to tensors, so we can map them in 2d to calculate nearest neighbour
    x_val = []
    y_val = []
    for idx, (data_in_test, target) in enumerate(val_loader):
        x_val = data_in_test.squeeze()
        y_val = target.squeeze()

    x_test = []
    y_test =[]
    for idx, (data_in_test, target) in enumerate(test_loader):
        x_test = data_in_test.squeeze()
        y_test = target.squeeze()

    x_train = []
    y_train = []
    for idx, (data_in_train, target) in enumerate(train_loader):
        x_train = data_in_train.squeeze()
        y_train = target.squeeze()

    # Transform to tensors, and then to numpy arrays. Use cpu() to move the data from the gpu to cpu, as scikit learn does not have cuda support.
    x_val = torch.tensor(x_val, device=device)
    y_val = torch.tensor(y_val, device=device)
    x_test = torch.tensor(x_test, device=device)
    y_test = torch.tensor(y_test, device=device)
    x_train = torch.tensor(x_train, device=device)
    y_train = torch.tensor(y_train, device=device)
    x_val = x_val.cpu().data.numpy()
    y_val = y_val.cpu().data.numpy()
    x_test = x_test.cpu().data.numpy()
    y_test = y_test.cpu().data.numpy()
    x_train = x_train.cpu().data.numpy()
    y_train = y_train.cpu().data.numpy()

    return x_train, y_train, x_test, y_test, x_val, y_val # return 3 featurelists, and 3 labellists.


def KNN_model(x_train, y_train, x_test, neighbours, device, log_boolean, log = 100):
    amount_of_img = x_test.shape[0]
    print(amount_of_img)
    amount_of_train = y_train.shape[0]
    img_size = x_test.shape[1]

    y_test = torch.zeros((amount_of_img), device=device, dtype=torch.float)

    for test_idx in range(0, amount_of_img):
        #calculate distance to all datapoints in the training set
        test_img = x_test[test_idx]
        distances_for_test = torch.norm(x_train - test_img, dim=1)

        indexes = torch.topk(distances_for_test, neighbours, largest=False)[1]
        classes = torch.gather(y_train, 0, indexes)
        modus = int(torch.mode(classes)[0])

        y_test[test_idx] = modus

        if log_boolean:
            if test_idx % log == 0:
                print("predicting on index = %d" % test_idx)

    return y_test


def train():

    x_train, y_train, x_test, y_test, x_val, y_val = dataset(False)

    print("train and test sizes are: %s, %s" % (str(x_train.shape), str(x_test.shape)))

    #pred = KNN_model(x_train, y_train, x_test, neighbours=1, device = device)
    #correct = pred.eq(y_test.to(device).view_as(pred)).sum()
    #print("Correct pred %d/%d, Accuracy %f" % (correct, y_test.shape[0], 100. * correct/y_test.shape[0]))


    k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 25, 37, 49, int(np.floor(np.sqrt(x_train.shape[0]))), 200]

    correct_vals = []

    best_k = -1
    best_correct = 0

    for k in k_values:
        pred = KNN_model(x_train, y_train, x_test, neighbours=k, device=device, log_boolean=False)
        correct = pred.eq(y_test.view_as(pred)).sum()
        print("K = %d, Correct: %d, Accuracy: %.2f" % (k, correct, 100. * correct / y_test.shape[0]))


x_train, y_train, x_test, y_test, x_val, y_val = dataset(False)




def save_model(classifer): # method for saving the model, so it doesnt need to be fitted anymore.
    filename = 'finalized_model.sav'
    pickle.dump(classifer, open(filename, 'wb'))

def load_model(): # Method for loading a saved model back for use.
    return pickle.load(open('model.sav', 'rb'))


def plot_best_K(): # A method used for showcasing the best K-value for the KNN-model.
    error = []

    # Calculating error for K values between 1 and 20
    for i in range(1, 20):
        print(i)
        knn = KNeighborsClassifier(n_neighbors=i) #Makes the classifier, with K=i
        knn.fit(x_val, y_val) # fits the validation data
        pred_i = knn.predict(x_test) # predicts on the test data
        error.append(np.mean(pred_i != y_test)) # finds the mean error for the predictions.

    #Plots a graph, showing the mean error for each K-value. Low errors, mean a good K-value.
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 20), error, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')

    plt.show()


if __name__ == '__main__': # main func

    classifier = KNeighborsClassifier(n_neighbors=12) #Creating the KNN-model with K=12
    classifier.fit(x_train, y_train) # training the model with the train data, features and labels
    y_pred = classifier.predict(x_test) # predicting with the model on the test features.

    #printing accuracy, and a report showing different metrics, as well as the confusion matrix.
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))










