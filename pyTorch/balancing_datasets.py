import os
from pyTorch.utils import ROOT_DIR


def copy_files(PATH, NEWPATH, AMOUNT):
    filelist = os.listdir(PATH)
    for i in range(0, AMOUNT):
        os.rename(PATH + filelist[i], NEWPATH + filelist[i])

train_balance = 200

# This is to balance out the 8+8 validation set by taking from test and train.
target_path = ROOT_DIR + '\\chest_xray\\val\\'
train_path = ROOT_DIR + '\\chest_xray\\train\\'
test_path = ROOT_DIR + '\\chest_xray\\test\\'

copy_files(train_path + "NORMAL/", target_path + "NORMAL/", 100)
copy_files(train_path + "PNEUMONIA/", target_path + "PNEUMONIA/", 100)
copy_files(test_path + "NORMAL/", target_path + "NORMAL/", 50)
copy_files(test_path + "PNEUMONIA/", target_path + "PNEUMONIA/", 50)


def balance(PATH, PATH_NORMAL):
    file_length = len(os.listdir(PATH_NORMAL))
    filelist = os.listdir(PATH)
    for i in range(0, len(filelist) - file_length):
        os.remove(PATH + filelist[i])
    return len(filelist)

"""
balance(ROOT_DIR + '\\chest_xray\\train_balanced\\PNEUMONIA\\', ROOT_DIR + '\\chest_xray\\train_balanced\\NORMAL\\')
balance(ROOT_DIR + '\\chest_xray\\test_balanced\\PNEUMONIA\\', ROOT_DIR + '\\chest_xray\\test_balanced\\NORMAL\\')
"""

