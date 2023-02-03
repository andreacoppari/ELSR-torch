import tensorflow as tf
from tensorflow.keras import layers, Input

def ELSR():
    input = Input(shape=(256, 256, 3))
    x = layers.Conv2D(6, kernel_size=(3, 3), padding="same", kernel_initializer="he_normal", activation="PReLU")(input)
    x = layers.Conv2D(6, kernel_size=(3, 3), padding="same", kernel_initializer="he_normal", activation="PReLU")(x)
    x = layers.Conv2D(6, kernel_size=(3, 3), padding="same", kernel_initializer="he_normal", activation="PReLU")(x)
    x = layers.Conv2D(6, kernel_size=(3, 3), padding="same", kernel_initializer="he_normal", activation="PReLU")(x)
    x = layers.Conv2D(48, kernel_size=(3, 3), padding="same", kernel_initializer="he_normal", activation="PReLU")(x)
    x = layers.UpSampling2D(size=(4, 4), interpolation="nearest")(x)
    x = layers.PReLU()
    model = tf.keras.Model(inputs=input, outputs=x)
    return model