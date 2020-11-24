import os
from pyTorch.utils import ROOT_DIR


def copy_files(PATH, NEWPATH, AMOUNT): # Method made to move images from test and train to validation, to make it usable.
    filelist = os.listdir(PATH) # finds file-folder.
    for i in range(0, AMOUNT):
        os.rename(PATH + filelist[i], NEWPATH + filelist[i]) # Renameing all files to a new folder.


target_path = ROOT_DIR + '\\chest_xray\\val\\'
train_path = ROOT_DIR + '\\chest_xray\\train\\'
test_path = ROOT_DIR + '\\chest_xray\\test\\'

# Taking 200 from train and 100 from test.
copy_files(train_path + "NORMAL/", target_path + "NORMAL/", 100)
copy_files(train_path + "PNEUMONIA/", target_path + "PNEUMONIA/", 100)
copy_files(test_path + "NORMAL/", target_path + "NORMAL/", 50)
copy_files(test_path + "PNEUMONIA/", target_path + "PNEUMONIA/", 50)


def balance(PATH, PATH_NORMAL): # Method to balance the train dataset, by deleting pneumonia images, so they end up having a 50/50 distribution.
    file_length = len(os.listdir(PATH_NORMAL)) # length of normal images
    filelist = os.listdir(PATH) # pneumonia folder
    for i in range(0, len(filelist) - file_length): # for loop so it deletes the surplus images
        os.remove(PATH + filelist[i]) # removing file
    return len(filelist) # return length of new folder

"""
balance(ROOT_DIR + '\\chest_xray\\train_balanced\\PNEUMONIA\\', ROOT_DIR + '\\chest_xray\\train_balanced\\NORMAL\\')
balance(ROOT_DIR + '\\chest_xray\\test_balanced\\PNEUMONIA\\', ROOT_DIR + '\\chest_xray\\test_balanced\\NORMAL\\')
"""

