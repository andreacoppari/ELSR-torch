import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from model import ELSR
import numpy as np
import os

train_X = np.load(os.path.join(os.getcwd(), "datasets/npy/train_aug_X.npy"))
train_Y = np.load(os.path.join(os.getcwd(), "datasets/npy/train_aug_Y.npy"))

val_X = np.load(os.path.join(os.getcwd(), "datasets/npy/val_X.npy"))
val_Y = np.load(os.path.join(os.getcwd(), "datasets/npy/val_Y.npy"))

model = ELSR(upscale_factor=4)
model.compile(optimizer=Adam(learning_rate=5e-4), loss='mse')

model.fit(train_X, train_Y, batch_size=8, epochs=500, validation_data=(val_X, val_Y), 
            callbacks=[LearningRateScheduler(lambda epoch: 5e-4*0.5**(epoch//200))])