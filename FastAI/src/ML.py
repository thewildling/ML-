from fastai.vision import *
import numpy as np # linear algebra
import pandas as pd
from utils import ROOT_DIR
from matplotlib import pyplot as pl
import matplotlib.image as mpimg
from PIL import Image

def make_pred(learn, file):
    file = file
    img = open_image(file)  # open the image using open_image func from fast.ai
    print(learn.predict(img)[0])  # lets make some prediction
    #plt.imshow(img)
    #plt.show()

def make_and_train():
    path = Path(ROOT_DIR + '\\consolidated\\')

    tfms = get_transforms(do_flip=True, max_lighting=0.1, max_rotate=0.1)

    data = ImageDataBunch.from_folder(path, train='.', valid_pct=0.15, ds_tfms=tfms, size=196,
                                      num_workers=4).normalize(imagenet_stats)
    # valid size here its 15% of total images,
    # train = train folder here we use all the folder

    print(data)
    data.show_batch(rows=3)
    plt.show()
    fb = FBeta()
    fb.average = 'macro'
    # We are using fbeta macro average in case some class of birds have less train images

    learn = cnn_learner(data, models.resnet18, metrics=[error_rate, fb], model_dir=ROOT_DIR + '\\models\\')
    learn.lr_find()
    learn.recorder.plot()

    lr = 1e-2  # learning rate

    #learn.fit_one_cycle(6, lr, moms=(0.8, 0.7))  # moms
    learn.load(ROOT_DIR + "\\models\\tmp")
    """
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_top_losses(9, figsize=(20, 8))
    interp.most_confused(min_val=3)
    interp.plot_confusion_matrix()
    #interp.losses.view(5)
    """
    plt.show()
    test_net(learn)
    #make_pred(learn, ROOT_DIR + '\\own_images\\robin_6.jpg')

    # make_pred(learn, ROOT_DIR + '\\valid\\ANTBIRD\\2.jpg')
    #learn.export()
    #learn.save()

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
    make_pred(learn, ROOT_DIR + '\\own_images\\robin_6.jpg')

    make_pred(learn, ROOT_DIR + '\\valid\\ANTBIRD\\2.jpg')


if __name__ == '__main__':
    # valid size here its 15% of total images,
    make_and_train()
    # train = train folder here we use all the folder

    # print(data)
    # data.show_batch(rows=3)

