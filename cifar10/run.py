import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from pathlib import Path
import argparse
from models import models

def load_dataset():
    (train_x, train_y), (test_x, test_y) = cifar10.load_data()
    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)
    return (train_x, train_y), (test_x, test_y)

def prepare_images(images):
    images = images.astype("float32")
    mean = [125.307, 122.95, 113.865]
    std  = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        images[:,:,:,i] = (images[:,:,:,i] - mean[i]) / std[i]
    return images

def scheduler(epoch):
    if epoch < 100:
        return 0.01
    if epoch < 200:
        return 0.001
    return 0.0001

def main(model, log_dir):
    INIT_LR = 0.001
    NUM_EPOCHS = 200
    BATCH_SIZE = 64

    train_dataset, test_dataset = load_dataset()
    train_x, train_y = train_dataset
    train_x = prepare_images(train_x)
    train_datagen = ImageDataGenerator(
            width_shift_range=0.125,
            height_shift_range=0.125,
            horizontal_flip=True,
            fill_mode="constant",
            cval=0,
            dtype="float32")

    train_datagen.fit(train_x)

    test_x, test_y = test_dataset
    test_x = prepare_images(test_x)
    model.summary()

    opt = SGD(lr=INIT_LR, momentum=0.9)
    # compile the model
    model.compile(loss=categorical_crossentropy, optimizer=opt, metrics=["accuracy"])

    # set callback
    tensorboard_cb = TensorBoard(log_dir="/tmp/tensorboard/"+log_dir, histogram_freq=0)
    learning_rate_cb = LearningRateScheduler(scheduler)
    callbacks = [tensorboard_cb, learning_rate_cb]
    history  = model.fit(train_datagen.flow(train_x, train_y, batch_size=BATCH_SIZE),
           steps_per_epoch=len(train_x) // BATCH_SIZE, epochs=NUM_EPOCHS,
           validation_data=(test_x, test_y), callbacks=callbacks)

    model.save(str(Path(log_dir).joinpath("model", "model.h5")))

if __name__ == "__main__":
    print("Exploring architectures on cifar10")
    model_name = "MobileNetV2_naive"
    main(models[model_name].get(), model_name)
