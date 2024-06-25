"""
Build Yolov8n-light backbone + Neck + Head
"""

import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import *


cfg = {"m": [0.67, 0.75, 768],
       "n": [0.33, 0.25, 1024],
       "s": [0.33, 0.50, 1024],
       }


class Conv:
    def __init__(self, filters, kernel, strides=(1, 1), padding="same", data_format=None,
                 dilation_rate=(1, 1), groups=1, activation=None, bias=True, fuse=False, name=""):
        self.conv = Conv2D(filters, kernel, strides, padding, data_format, dilation_rate, groups,
                           activation=activation if fuse else None, use_bias=bias, name=name)
        self.bn = BatchNormalization()
        self.act = keras.activations.get(activation) if activation else None
        self.fuse = fuse
        self.name = name

    def __call__(self, x):
        x = self.conv(x)
        if not self.fuse:
            x = self.bn(x)
            x = self.act(x)
        return x


class DWConv:
    def __init__(self, kernel, strides=(1, 1), padding="same", depth_multiplier=1, data_format=None,
                 dilation_rate=(1, 1), activation=None, bias=True, fuse=False, name="name"):
        self.dw = DepthwiseConv2D(kernel, strides, padding, depth_multiplier, data_format, dilation_rate,
                                  activation=activation if fuse else None, use_bias=bias, name=name)
        self.bn = BatchNormalization()
        self.act = keras.activations.get(activation)
        self.fuse = fuse

    def __call__(self, x):
        x = self.dw(x)
        if not self.fuse:
            x = self.bn(x)
            x = self.act(x)
        return x


class LightConv:
    def __init__(self, filters, kernel, strides=(1, 1), padding="same", depth_multiplier=1, data_format=None,
                 dilation_rate=(1, 1), groups=1, activation=None, bias=True, fuse=False, name=""):
        self.dw = DWConv(kernel, strides, padding, depth_multiplier, data_format, dilation_rate,
                         activation=None, bias=bias, fuse=fuse, name=name+"_dw")
        self.pw = Conv(filters, 1, 1, padding, data_format, dilation_rate, groups,
                       activation=activation, bias=True, fuse=fuse, name=name + "_pw")

    def __call__(self, x):
        return self.pw(self.dw(x))


class Bottleneck:
    def __init__(self, c1, c2, shortcut=True, k=(3, 3), padding="same", e=0.5, fuse=False, name=""):
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(filters=c_, kernel=k[0], strides=1, padding=padding, activation="relu", fuse=fuse, name=name+"_conv0")
        self.cv2 = Conv(filters=c2, kernel=k[1], strides=1, padding=padding, activation="relu", fuse=fuse, name=name+"_conv1")
        self.add = shortcut and c1 == c2

    def __call__(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return Add()([x, self.cv2(self.cv1(x))]) if self.add else self.cv2(self.cv1(x))


class LightBottleneck:
    def __init__(self, c1, c2, shortcut=True, k=(3, 3), padding="same", e=0.5, fuse=False, name=""):
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = LightConv(filters=c_, kernel=k[0], strides=1, padding=padding, activation="relu", fuse=fuse, name=name+"_lconv0")
        self.cv2 = LightConv(filters=c2, kernel=k[1], strides=1, padding=padding, activation="relu", fuse=fuse, name=name+"_lconv1")
        self.add = shortcut and c1 == c2

    def __call__(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return Add()([x, self.cv2(self.cv1(x))]) if self.add else self.cv2(self.cv1(x))


class C2f:
    def __init__(self, c2, n=1, shortcut=False, e=0.5, fuse=False, name=""):
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(filters=2*self.c, kernel=1, strides=1, activation="relu", fuse=fuse, name=name+"_pw0")
        self.cv2 = Conv(filters=c2, kernel=1, activation="relu", fuse=fuse, name=name+"_pw1")
        self.btn = [Bottleneck(self.c, self.c, shortcut, k=(3, 3), e=1.0, fuse=fuse, name=name+f"_btn{i}")
                    for i in range(n)]
        self.n = n

    def __call__(self, x):
        """Forward pass through C2f layer."""
        x = self.cv1(x)
        c05 = x.shape[-1] // 2
        x = [Lambda(lambda x: x[:,:,:,:c05])(x), Lambda(lambda x: x[:,:,:,c05:])(x)]
        x.extend(btn(x[-1]) for btn in self.btn)
        x = Concatenate(axis=-1)(x)
        return self.cv2(x)


class LightC2f:
    def __init__(self, c2, n=1, shortcut=False, e=0.5, fuse=False, name=""):
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(filters=2*self.c, kernel=1, strides=1, activation="relu", fuse=fuse, name=name+"_pw0")
        self.cv2 = Conv(filters=c2, kernel=1, activation="relu", fuse=fuse, name=name+"_pw1")
        self.btn = [LightBottleneck(self.c, self.c, shortcut, k=(3, 3), e=1.0, fuse=fuse, name=name+f"_lbtn{i}")
                    for i in range(n)]
        self.n = n

    def __call__(self, x):
        """Forward pass through C2f layer."""
        x = self.cv1(x)
        c05 = x.shape[-1] // 2
        x = [Lambda(lambda x: x[:,:,:,:c05])(x), Lambda(lambda x: x[:,:,:,c05:])(x)]
        x.extend(btn(x[-1]) for btn in self.btn)
        x = Concatenate(axis=-1)(x)
        return self.cv2(x)


class SPPF:
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5, fuse=False, name=""):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c_, 1, 1, activation="relu", fuse=fuse, name=name+"_conv0")
        self.cv2 = Conv(c2, 1, 1, activation="relu", fuse=fuse, name=name+"_conv1")
        # self.m = MaxPooling2D(pool_size=k, strides=1, padding="same")
        self.k = k

    def __call__(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = MaxPooling2D(pool_size=self.k, strides=1, padding="same")(x)
        y2 = MaxPooling2D(pool_size=self.k, strides=1, padding="same")(y1)
        y3 = MaxPooling2D(pool_size=self.k, strides=1, padding="same")(y2)
        y = Concatenate(axis=-1)([x,y1,y2,y3])
        return self.cv2(y)


class LightDetect:
    def __init__(self, c2, c3, nc=1, fuse=False, name=""):
        self.nc = nc  # number of classes
        self.c2 = c2
        self.c3 = c3
        self.fuse = fuse
        self.name = name

    def __call__(self, x):
        x = LightConv(self.c2, 3, 1, activation="relu", fuse=self.fuse, name=self.name+f"_lconv0")(x)
        x = LightConv(self.c2, 3, 1, activation="relu", fuse=self.fuse, name=self.name+f"_lconv1")(x)
        x = Conv2D(3*(5+self.nc), kernel_size=1, name=self.name+f"_loc_pw0")(x)
        # x = Reshape(target_shape=(x_box.shape[1] * x_box.shape[2], 4))(x_box)

        return x


def YOLOv8Light(type="n", fuse=False):
    d, w, mc = cfg[type]
    inp = Input(shape=(320, 320, 3))

    x = Conv(filters=round(64*w), kernel=3, strides=2, activation="relu", fuse=fuse, name="conv_cv")(inp)
    x = LightConv(filters=round(128*w), kernel=3, strides=2, activation="relu", name="lconv0", fuse=fuse)(x)
    x = LightC2f(c2=round(128*w), n=round(3*d), shortcut=True, name="lc2f", fuse=fuse)(x)
    x = LightConv(filters=round(256*w), kernel=3, strides=2, activation="relu", name="lconv1", fuse=fuse)(x)
    x1 = LightC2f(c2=round(256*w), n=round(6*d), shortcut=True, name="lc2f1", fuse=fuse)(x)
    x = LightConv(filters=round(512*w), kernel=3, strides=2, activation="relu", name="lconv2", fuse=fuse)(x1)
    x2 = LightC2f(c2=round(512 * w), n=round(6 * d), shortcut=True, name="lc2f2", fuse=fuse)(x)
    x = LightConv(filters=round(min(1024, mc) * w), kernel=3, strides=2, activation="relu", name="lconv3", fuse=fuse)(x2)
    x = LightC2f(c2=round(min(1024, mc) * w), n=round(3 * d), shortcut=True, name="lc2f3", fuse=fuse)(x)
    x3 = SPPF(c1=round(min(1024, mc) * w), c2=round(min(1024, mc) * w), k=5, name="sppf", fuse=fuse)(x)

    x = UpSampling2D(size=2, interpolation="nearest")(x3)
    x = Concatenate()([x2, x])
    x2 =LightC2f(c2=round(512 * w), n=round(3 * d), shortcut=False, name="lc2f4", fuse=fuse)(x)
    x = UpSampling2D(size=2, interpolation="nearest")(x2)
    x = Concatenate()([x1, x])
    x1 = LightC2f(c2=round(256 * w), n=round(3 * d), shortcut=False, name="lc2f5", fuse=fuse)(x)
    x = LightConv(filters=round(256 * w), kernel=3, strides=2, padding="same", activation="relu", name="lconv4", fuse=fuse)(x1)
    x = Concatenate()([x2, x])
    x2 = LightC2f(c2=round(512 * w), n=round(3 * d), shortcut=False, name="lc2f6", fuse=fuse)(x)
    x = LightConv(filters=round(512 * w), kernel=3, strides=2, padding="same", activation="relu", name="lconv5", fuse=fuse)(x2)
    x = Concatenate()([x3, x])
    x3 = LightC2f(c2=round(min(1024, mc) * w), n=round(3 * d), shortcut=False, name="lc2f7", fuse=fuse)(x)

    y1 = LightDetect(c2=64, c3=64, nc=1, fuse=fuse, name="ldetect")(x3)
    y2 = LightDetect(c2=64, c3=64, nc=1, fuse=fuse, name="ldetect1")(x2)
    y3 = LightDetect(c2=64, c3=64, nc=1, fuse=fuse, name="ldetect2")(x1)

    # y3 = Reshape(target_shape=(y3.shape[1] * y3.shape[2], y3.shape[3]))(y3)
    # y2 = Reshape(target_shape=(y2.shape[1] * y2.shape[2], y2.shape[3]))(y2)
    # y1 = Reshape(target_shape=(y1.shape[1] * y1.shape[2], y1.shape[3]))(y1)
    # y = Concatenate(axis=1)([y3, y2, y1])

    model = Model(inp, [y1, y2, y3])
    return model


if __name__ == "__main__":
    model = YOLOv8Light(type="n", fuse=True)
    model.summary()

    # model.save("../saved_models/light8n_fuse_detail_demo.h5")

    if isinstance(model.output, list):
        last_layers = [out.name.split("/")[0] for out in model.output]
    else:
        last_layers = model.output.name.split("/")[0]
    print(last_layers)