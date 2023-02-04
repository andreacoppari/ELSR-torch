from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from model import ELSR
from dataset import get_training_data
from preprocessing import augment_data

train_X, train_Y, val_X, val_Y = get_training_data()

train_aug_X = []
train_aug_Y = []

for lr, hr in train_X, train_Y:
    l,h = augment_data(lr, hr)
    train_aug_X.append(l)
    train_aug_Y.append(h)

print(f"train_X samples: {len(train_X)}, {len(val_X)}")
print(f"train_X sample shape: {train_X[0].shape}")

model = ELSR(upscale_factor=4)
model.compile(optimizer=Adam(learning_rate=5e-4), loss='mae')

model.fit(train_X, train_Y, batch_size=32, epochs=500, validation_data=(val_X, val_Y), 
            callbacks=[LearningRateScheduler(lambda epoch: 5e-4*0.5**(epoch//200))])