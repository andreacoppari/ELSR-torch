import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, PReLU

class PixelShuffle(tf.keras.layers.Layer):
    def __init__(self, upscale_factor, **kwargs):
        super(PixelShuffle, self).__init__(**kwargs)
        self.upscale_factor = upscale_factor

    def call(self, inputs):
        x = inputs
        batch_size, h, w, c = x.shape
        x = tf.reshape(x, (batch_size, h // self.upscale_factor, self.upscale_factor, 
                         w // self.upscale_factor, self.upscale_factor, c))
        x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
        x = tf.reshape(x, (batch_size, h // self.upscale_factor, w // self.upscale_factor, 
                         c * (self.upscale_factor ** 2)))
        return x

def ELSR(upscale_factor=4) -> tf.keras.models.Model:
    input = Input(shape=(320, 180, 3))
    x = Conv2D(6, kernel_size=(3, 3), padding="same", kernel_initializer="he_normal", activation="relu")(input)
    x = Conv2D(6, kernel_size=(3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = PReLU()(x)
    x = Conv2D(6, kernel_size=(3, 3), padding="same", kernel_initializer="he_normal", activation="relu")(x)
    x = Conv2D(48, kernel_size=(3, 3), padding="same", kernel_initializer="he_normal", activation="relu")(x)
    x = PixelShuffle(upscale_factor=upscale_factor)(x)
    model = tf.keras.Model(inputs=input, outputs=x)
    return model
