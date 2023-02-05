from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from model import ELSR
import numpy as np
import os

train_X = np.array(np.load(os.path.join(os.getcwd(), "datasets/npy/train_aug_X.npy")), dtype="float32")
train_Y = np.array(np.load(os.path.join(os.getcwd(), "datasets/npy/train_aug_Y.npy")), dtype="float32")
val_X = np.array(np.load(os.path.join(os.getcwd(), "datasets/npy/val_X.npy")), dtype="float32")
val_Y = np.array(np.load(os.path.join(os.getcwd(), "datasets/npy/val_Y.npy")), dtype="float32")

model = ELSR(upscale_factor=4)
model.compile(optimizer=Adam(learning_rate=5e-4), loss='mse')

model.fit(train_X, train_Y, batch_size=32, epochs=500, validation_data=(val_X, val_Y), 
            callbacks=[LearningRateScheduler(lambda epoch: 5e-4*0.5**(epoch//200))])