from fastai.vision import *
import numpy as np # linear algebra
import pandas as pd
from FastAI.utils import ROOT_DIR
from matplotlib import pyplot as pl
import matplotlib.image as mpimg
from PIL import Image
from torchvision import transforms

def make_pred(learn, file):
    file = file
    img = open_image(file)  # open the image using open_image func from fast.ai
    print(learn.predict(img)[0])  # lets make some prediction
    #plt.imshow(img)
    #plt.show()

def make_and_train():
    path = Path(ROOT_DIR + '\\consolidated\\')


    data = ImageDataBunch.from_folder(path, train='.', valid_pct=0.20, ds_tfms=get_transforms(), size=224,
                                      num_workers=4).normalize(imagenet_stats)
    # valid_pct: chosing to create the validation folder of 20% of the images
    # train = train folder
    # setting size of image to 224*224 pixels
    # Normalizing the data set
    fb = FBeta()
    fb.average = 'macro'
    print(data)
    learn = cnn_learner(data, models.resnet50, metrics=[accuracy, fb], model_dir=ROOT_DIR + '\\models\\', path=Path('.'))
    learn.lr_find()
    learn.recorder.plot(suggestion=True)
    plt.show()

    lr1 = 1e-3 #initial guess
    lr2 = 1e-1 #intial guess
    learn.fit_one_cycle(4, slice(lr1, lr2))
    lr = learn.lr_find()
    learn.recorder.plot(suggestion=True)
    plt.show()
    # lr1 = 1e-3

    learn.fit_one_cycle(20, slice(lr))

    learn.unfreeze()
    lr = learn.lr_find()
    learn.recorder.plot(suggestion=True)
    plt.show()

    learn.fit_one_cycle(10, slice(lr))

    learn.recorder.plot_losses()

    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_top_losses(9, figsize=(20, 8))
    interp.most_confused(min_val=3)
    interp.plot_confusion_matrix()
    plt.show()


    #learn.export()
    #learn.save()

    """
    data.show_batch(rows=3)
    plt.show()
    fb = FBeta()
    fb.average = 'macro'
    # We are using fbeta macro average in case some class of birds have less train images

    learn = cnn_learner(data, models.resnet18, metrics=[error_rate, fb], model_dir=ROOT_DIR + '\\models\\')
    learn.lr_find()
    learn.recorder.plot()

    lr = 1e-2  # learning rate

    learn.fit_one_cycle(6, lr, moms=(0.8, 0.7))  # moms
    learn.load(ROOT_DIR + "\\models\\tmp")
    """


def test_net(learn):
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


def test():
    path = Path(ROOT_DIR + '\\consolidated\\')

    tfms = get_transforms(do_flip=True, max_lighting=0.1, max_rotate=0.1)
    data = ImageDataBunch.from_folder(path, train='.', valid_pct=0.15, ds_tfms=tfms, size=196,
                                     num_workers=4).normalize(imagenet_stats)
    fb = FBeta()
    fb.average = 'macro'
    learn = cnn_learner(data, models.resnet18, metrics=[error_rate, fb])
    learn.load(ROOT_DIR + "\\models\\tmp")

def interp():
    learn = load_learner(ROOT_DIR + '\\consolidated\\')

    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_top_losses(9, figsize=(20, 8))
    interp.most_confused(min_val=3)

    plt.show()


def load_model():
    path = Path(ROOT_DIR + '\\consolidated\\')

    np.random.seed(40)

    data = ImageDataBunch.from_folder(path, train='.', valid_pct=0.20, ds_tfms=get_transforms(), size=224,
                                      num_workers=4).normalize(imagenet_stats)

    learn = cnn_learner(data, models.resnet50, metrics=[accuracy], model_dir=ROOT_DIR + '\\models\\', path=Path('.'))
    learn.load(ROOT_DIR + '\\models\\tmp')
    return learn


if __name__ == '__main__':
    # valid size here its 15% of total images,
    # train = train folder here we use all the folder
    make_and_train()


    # print(data)
    # data.show_batch(rows=3)

