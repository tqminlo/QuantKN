import keras
from keras.layers import *
from keras.models import Model
from PIL import Image


class MobileNet:
    def __init__(self, size):
        self.size = size

    def __call__(self):
        inp = Input(shape=(self.size, self.size, 3))

        x = Conv2D(filters=16, kernel_size=3, strides=2, padding='same', activation='relu')(inp)
        x1 = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', activation='relu')(x)
        x1 = Conv2D(filters=16, kernel_size=1, strides=1, padding='same')(x1)
        x = Add()((x1, x))

        x = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu')(x)
        x = DepthwiseConv2D(kernel_size=3, strides=2, padding='same', activation='relu')(x)
        x = Conv2D(filters=24, kernel_size=1, strides=1, padding='same')(x)
        x1 = Conv2D(filters=72, kernel_size=1, strides=1, padding='same', activation='relu')(x)
        x1 = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', activation='relu')(x1)
        x1 = Conv2D(filters=24, kernel_size=1, strides=1, padding='same')(x1)
        x = Add()((x1, x))

        x = Conv2D(filters=72, kernel_size=1, strides=1, padding='same', activation='relu')(x)
        x = DepthwiseConv2D(kernel_size=3, strides=2, padding='same', activation='relu')(x)
        x = Conv2D(filters=40, kernel_size=1, strides=1, padding='same')(x)
        x1 = Conv2D(filters=120, kernel_size=1, strides=1, padding='same', activation='relu')(x)
        x1 = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', activation='relu')(x1)
        x1 = Conv2D(filters=40, kernel_size=1, strides=1, padding='same')(x1)
        x = Add()((x1, x))

        x1 = Conv2D(filters=120, kernel_size=1, strides=1, padding='same', activation='relu')(x)
        x1 = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', activation='relu')(x1)
        x1 = Conv2D(filters=40, kernel_size=1, strides=1, padding='same')(x1)
        x = Add()((x1, x))

        x = Conv2D(filters=240, kernel_size=1, strides=1, padding='same', activation='relu')(x)
        x = DepthwiseConv2D(kernel_size=3, strides=2, padding='same', activation='relu')(x)
        x = Conv2D(filters=80, kernel_size=1, strides=1, padding='same')(x)
        x1 = Conv2D(filters=200, kernel_size=1, strides=1, padding='same', activation='relu')(x)
        x1 = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', activation='relu')(x1)
        x1 = Conv2D(filters=80, kernel_size=1, strides=1, padding='same')(x1)
        x = Add()((x1, x))

        x1 = Conv2D(filters=184, kernel_size=1, strides=1, padding='same', activation='relu')(x)
        x1 = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', activation='relu')(x1)
        x1 = Conv2D(filters=80, kernel_size=1, strides=1, padding='same')(x1)
        x = Add()((x1, x))

        x1 = Conv2D(filters=184, kernel_size=1, strides=1, padding='same', activation='relu')(x)
        x1 = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', activation='relu')(x1)
        x1 = Conv2D(filters=80, kernel_size=1, strides=1, padding='same')(x1)
        x = Add()((x1, x))

        x = Conv2D(filters=480, kernel_size=1, strides=1, padding='same', activation='relu')(x)
        x = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', activation='relu')(x)
        x = Conv2D(filters=112, kernel_size=1, strides=1, padding='same')(x)
        x1 = Conv2D(filters=672, kernel_size=1, strides=1, padding='same', activation='relu')(x)
        x1 = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', activation='relu')(x1)
        x1 = Conv2D(filters=112, kernel_size=1, strides=1, padding='same')(x1)
        x = Add()((x1, x))

        x = Conv2D(filters=672, kernel_size=1, strides=1, padding='same', activation='relu')(x)
        x = DepthwiseConv2D(kernel_size=3, strides=2, padding='same', activation='relu')(x)
        x = Conv2D(filters=160, kernel_size=1, strides=1, padding='same')(x)
        x1 = Conv2D(filters=960, kernel_size=1, strides=1, padding='same', activation='relu')(x)
        x1 = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', activation='relu')(x1)
        x1 = Conv2D(filters=160, kernel_size=1, strides=1, padding='same')(x1)
        x = Add()((x1, x))

        x1 = Conv2D(filters=960, kernel_size=1, strides=1, padding='same', activation='relu')(x)
        x1 = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', activation='relu')(x1)
        x1 = Conv2D(filters=160, kernel_size=1, strides=1, padding='same')(x1)
        x = Add()((x1, x))

        x = Conv2D(filters=960, kernel_size=1, strides=1, padding='same', activation='relu')(x)
        x = AveragePooling2D(pool_size=(7,7))(x)
        x = Conv2D(filters=1280, kernel_size=1, strides=1, padding='same', activation='relu')(x)
        x = Conv2D(filters=100, kernel_size=1, strides=1, padding='same')(x)
        x = Reshape(target_shape=(100,))(x)
        x = Softmax()(x)

        model = Model(inp, x)
        return model


if __name__ == "__main__":
    model = MobileNet(224)()
    model.summary()