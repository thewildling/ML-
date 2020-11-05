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
from pyTorch.utils import ROOT_DIR
dataset_root = ROOT_DIR + '\\chest_xray\\'
batch_size = 128
new_size = (224,224)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dataset(): # Take in the datasets from folders, and make them into tensors, so they can be used for training.
    transform = transforms.Compose([
        transforms.Grayscale(), # Since the images are blanc/white
        transforms.Resize(new_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        nn.Flatten()
    ])

    train = torchvision.datasets.ImageFolder(root=dataset_root + 'train', transform=transform)

    test = torchvision.datasets.ImageFolder(root=dataset_root + 'test', transform=transform)

    val = torchvision.datasets.ImageFolder(root=dataset_root + 'val', transform=transform)

    train_loader = torch.utils.data.DataLoader(train, batch_size=len(train), shuffle=True)

    val_loader = torch.utils.data.DataLoader(val, batch_size=len(val), shuffle=True)

    test_loader = torch.utils.data.DataLoader(test, batch_size=len(test), shuffle=True)


    #convert the data in the sets to tensors, so we can map them in 2d to calculate nearest neighbour
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

    x_test = torch.tensor(x_test, device=device)
    y_test = torch.tensor(y_test, device=device)
    x_train = torch.tensor(x_train, device=device)
    y_train = torch.tensor(y_train, device=device)
    return x_train, y_train, x_test, y_test


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

x_train, y_train, x_test, y_test = dataset()

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




