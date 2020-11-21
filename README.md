# ML-

## 02.11.20
Used ResNet50 to predict the pictures. Gets an accuray of around 95 % after 6 epochs. 

## 03.11.20
A lack of validation data could make it difficult to validate the result of the accuracy of the model.

### This could be a way to balance the dataset for more validation 
def copy_files(PATH, NEWPATH, AMOUNT):
    filelist = os.listdir(PATH)
    for i in range(0, AMOUNT):
        os.rename(PATH + filelist[i], NEWPATH + filelist[i])

test_balance = 50
train_balance = 200

target_path = "/content/chest_xray/val/"
train_path = "/content/chest_xray/train/"
test_path = "/content/chest_xray/test/"

copy_files(train_path + "NORMAL/", target_path + "NORMAL/", 100)
copy_files(train_path + "PNEUMONIA/", target_path + "PNEUMONIA/", 100)
copy_files(test_path + "NORMAL/", target_path + "NORMAL/", 50)
copy_files(test_path + "PNEUMONIA/", target_path + "PNEUMONIA/", 50)

#### 04.11.20
The test, train and validation sets have now been balanced. These new sets will be used for the KNN model, by using the code-snippet provided above. The reason is to get a bigger validation set, as the one provided from Kaggle.com only had 8 images in it. It now has over 150 images, making validation and accuracy more complete. 

### KNN

Looking into K-Nearest Neighbours. Probably going with PyTorch without FastAI, as FastAI does not have a KNN model. 
#### 05.11.20
A working KNN-model have been made, which reaches an accuracy of 79 % with k = 2. The code for this is under the pyTorch-folder.
