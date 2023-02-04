from tensorflow.keras.optimizers import Adam
from model import ELSR
from dataset import get_training_data
from preprocessing import augment_data

train_X, train_Y, val_X, val_Y = get_training_data()

print(train_Y)