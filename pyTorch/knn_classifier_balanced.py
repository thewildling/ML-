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
import scikitplot as skplt
from joblib import dump
from sklearn.metrics import plot_confusion_matrix

import wandb
from pyTorch.utils import ROOT_DIR
dataset_root = ROOT_DIR + '\\chest_xray\\'
batch_size = 128
new_size = (224,224)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

labels = ['NORMAL', 'PNEUMONIA']

def dataset(): # Take in the datasets from folders, and make them into tensors, so they can be used for training.
    transform = transforms.Compose([ # Since the images are blanc/white
        transforms.Resize(new_size),
        transforms.Grayscale(),
        transforms.ToTensor(),
        nn.Flatten()
    ])

    train = torchvision.datasets.ImageFolder(root=dataset_root + 'train_balanced', transform=transform)

    test = torchvision.datasets.ImageFolder(root=dataset_root + 'test', transform=transform)

    val = torchvision.datasets.ImageFolder(root=dataset_root + 'val', transform=transform)

    train_loader = torch.utils.data.DataLoader(train, batch_size=len(train), shuffle=True)

    val_loader = torch.utils.data.DataLoader(val, batch_size=len(val), shuffle=True)

    test_loader = torch.utils.data.DataLoader(test, batch_size=len(test), shuffle=True)


    #convert the data in the sets to tensors, so we can map them in 2d to calculate nearest neighbour
    x_val = []
    y_val = []
    for idx, (data_in_test, target) in enumerate(val_loader):
        x_val = data_in_test.squeeze()
        y_val = target.squeeze()

    x_test = []
    y_test = []
    for idx, (data_in_test, target) in enumerate(test_loader):
        x_test = data_in_test.squeeze()
        y_test = target.squeeze()

    x_train = []
    y_train = []
    for idx, (data_in_train, target) in enumerate(train_loader):
        x_train = data_in_train.squeeze()
        y_train = target.squeeze()

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
    print(x_train.shape)

    return x_train, y_train, x_test, y_test, x_val, y_val


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

    x_train, y_train, x_test, y_test, x_val, y_val = dataset()

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


x_train, y_train, x_test, y_test, x_val, y_val = dataset()




def save_model(classifer):
    filename = 'finalized_model.sav'
    pickle.dump(classifer, open(filename, 'wb'))

"""
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifer.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
"""
def load_model():
    return pickle.load(open('model.sav', 'rb'))


def plot_best_K():
    error = []

    # Calculating error for K values between 1 and 40
    for i in range(1, 20):
        print(i)
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(x_train, y_train)
        pred_i = knn.predict(x_test)
        error.append(np.mean(pred_i != y_test))

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 20), error, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')

    plt.show()


classifier = KNeighborsClassifier(n_neighbors=12)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


"""
from PIL import Image
img = Image.open(ROOT_DIR + '\\chest_xray\\test\\NORMAL\\IM-0083-0001.jpeg')

img = np.array(img).reshape(1, -1)
img = torch.tensor(img, device='cpu')
img = img.cpu().data.numpy()
model = load_model()
model.predict(img)

y_true = []
for num in y_test:
    y_true.append(labels[num])

classifier = KNeighborsClassifier(n_neighbors=2)
classifier.fit(x_train, y_train)
pred = classifier.predict(x_test)

plot_confusion_matrix(classifier, x_test, y_test, )
plt.show()

from PIL import Image
img = Image.open(ROOT_DIR + '\\chest_xray\\test\\NORMAL\\IM-0083-0001.jpeg')
img = img.resize((224,224), Image.ANTIALIAS)
img = np.array(img)
img = img.reshape(1, 50176)

print(img)
print(classifier.predict(img))
print(labels[classifier.predict(img)[0]])
"""
#print(f1_score(y_test, pred))








