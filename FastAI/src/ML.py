#Imports for the fastAI model
from fastai.vision import *
import numpy as np # linear algebra
import pandas as pd
from FastAI.utils import ROOT_DIR
from matplotlib import pyplot as pl
import matplotlib.image as mpimg
from PIL import Image
from torchvision import transforms
###############################


def make_pred(learn, file):#MEthod for predicting on a single image, using the learner-object from fastAI's predict function.
    file = file
    img = open_image(file)  # open the image using open_image func from fast.ai
    print(learn.predict(img)[0])  # printing the predictions


def make_and_train(): # The main function for the ResNet50 model. It makes the databunch, makes the classifier, and traind the classifier. It also saves the model to pkl.
    path = Path(ROOT_DIR + '\\consolidated\\') # path to data folder conataining images.
    # The databunch object that contains all the images, the validation set and training set.
    # Params: Path = path to image folder. train= is same folder, consolidated. valid_pct = the precentage of the images that is going to be the validation set.
    # ds_tfms=is the different transformations that can be done to the images. size= is the new size of the images, 224*224. num_workers= sub-processes for data_loading.
    # .nomralize(imagenet_stats) is used to normalize the images based on the imagenet_stats, since it is pretrained.
    data = ImageDataBunch.from_folder(path, train='.', valid_pct=0.20, ds_tfms=get_transforms(), size=224,
                                      num_workers=4).normalize(imagenet_stats)

    fb = FBeta() # the F1 score
    fb.average = 'macro' # setting the F1 score to macro.
    print(data) # printing the IDB

    # Setting up the model.
    # Params: data= the IDB that the model will be using. model= models.resnet50, is where we select the model that will be used. metrics= list of metrics to use.
    # model_dir= where the model will be saved if the save func is used. path= where the pkl.export model is saved.
    learn = cnn_learner(data, models.resnet50, metrics=[accuracy, fb], model_dir=ROOT_DIR + '\\models\\', path=Path('.'))
    learn.lr_find() # Finding the LR using 2 epochs.
    learn.recorder.plot(suggestion=True) # plotting it with suggestion of best.
    plt.show()

    lr1 = 1e-3 #LR after 2 epochs
    lr2 = 1e-1 #Lr after 2 epochs
    learn.fit_one_cycle(4, slice(lr1, lr2))
    learn.lr_find()
    learn.recorder.plot(suggestion=True)
    plt.show()
    # lr1 = 1e-3

    learn.fit_one_cycle(20, 1e-3)

    learn.unfreeze()
    learn.lr_find()
    learn.recorder.plot(suggestion=True)
    plt.show()

    learn.fit_one_cycle(10, 1e-4)

    learn.recorder.plot_losses()

    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_top_losses(9, figsize=(20, 8))
    interp.most_confused(min_val=3)
    interp.plot_confusion_matrix()
    plt.show()


    #learn.export() # Use to save a pkl.file used in the website for predictions
    #learn.save() " Save the file as a pth file, which can be loaded, so no more epochs need to be run.



def test_net(learn): # Method used for testing the output from the pkl file to the website.
    img = ROOT_DIR + '\\consolidated\\NORMAL\\IM-0001-0001.jpeg'
    print(learn.predict(open_image(img)))
    pred_class, pred_idx, outputs = learn.predict(open_image(img))  # [0]
    pred_probs = outputs / sum(outputs)
    pred_probs = pred_probs.tolist()
    predictions = []
    for image_class, output, prob in zip(learn.data.classes, outputs.tolist(), pred_probs):
        output = round(output, 1)
        prob = round(prob, 2)
        predictions.append(
            {"class": image_class.replace("_", " "), "output": output, "prob": prob}
        )

    predictions = sorted(predictions, key=lambda x: x["output"], reverse=True)
    predictions = predictions[0:2]
    print({"class": str(pred_class), "predictions": predictions})

def interp(): # Method used for checking the results of the model, by displaying the 9 top losses images for the model.
    learn = load_learner(ROOT_DIR + '\\consolidated\\')

    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_top_losses(9, figsize=(20, 8))
    interp.most_confused(min_val=3)

    plt.show()


def load_model(): # Method to load the model from the pth-file.
    path = Path(ROOT_DIR + '\\consolidated\\') # path to the model

    # Remaking the dataset, as it has to be created first, then loaded
    data = ImageDataBunch.from_folder(path, train='.', valid_pct=0.20, ds_tfms=get_transforms(), size=224,
                                      num_workers=4).normalize(imagenet_stats)

    # Remaking the model, as it has to be loaded
    learn = cnn_learner(data, models.resnet50, metrics=[accuracy], model_dir=ROOT_DIR + '\\models\\', path=Path('.'))
    learn.load(ROOT_DIR + '\\models\\tmp') # loading the model
    return learn # returning the model


if __name__ == '__main__': # main func

    make_and_train() # make and train the model




