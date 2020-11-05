# ML-

## 02.11.20
Used resnet18 to predict the pictures. Gets an accuray of around 95 % after 6 epochs. 

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
I have now balanced the test, train and validation sets, that will be used for the KNN model, using the code-snippet provided above. The reason is to get a bigger validation set, as the one provided from kaggle only has 8 images in it. It now has over 150 images, making validation and accuracy more complete. 

### KNN

Looking into k-nearest neighbours. Probably going with pyTorch wothout fastai, as fastai does not have a knn model. 
#### 05.11.20
I have made a working KNN-model, that gets an accuracy of 79 % with k = 2. Its under pyTorch folder 
