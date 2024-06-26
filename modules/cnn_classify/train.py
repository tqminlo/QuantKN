import os
import keras
import numpy as np
from models.mobilenet_minimalistic import MobileNet
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import sys
sys.path.append("../../")
import argparse


SIZE = 224
DATA_DIR = "../../datasets/imagenet100"
train_dir = f"{DATA_DIR}/train"
val_dir = f"{DATA_DIR}/val"
classify_zoo = {"mobilenet": {"train": MobileNet(SIZE)(),
                              "fuse": MobileNet(SIZE)()}
                }


class TrainFP:
    def __init__(self, model, batch_size, pretrained=None):
        self.batch_size = batch_size
        self.model = model
        if pretrained:
            self.model.load_weights(pretrained)
        process_data = ImageDataGenerator(rescale=1./255)
        self.train_data = process_data.flow_from_directory(directory=train_dir, target_size=(SIZE, SIZE),
                                                           class_mode='categorical', batch_size=batch_size)
        self.val_data = process_data.flow_from_directory(directory=val_dir, target_size=(SIZE, SIZE),
                                                         class_mode='categorical', batch_size=batch_size)

    def train(self, epochs, save_path, lr=0.0001):
        self.model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=['acc'])
        self.model.summary()

        callbacks = [keras.callbacks.EarlyStopping(patience=50),
                     keras.callbacks.ReduceLROnPlateau(factor=0.3, patience=15),
                     keras.callbacks.ModelCheckpoint(filepath=save_path, save_weights_only=True, monitor='val_acc',
                                                     mode='max', save_freq="epoch", save_best_only=True, verbose=1)]

        self.model.fit(self.train_data, batch_size=self.batch_size, epochs=epochs,
                       verbose=1, callbacks=callbacks, validation_data=self.val_data)

    def eval(self, weight_path):
        self.model.load_weights(weight_path)
        self.model.evaluate(self.val_data, batch_size=self.batch_size)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-a', '--architecture', help='architecture', default="mobilenet")
    ap.add_argument('-p', '--pretrained', help='pretrained', default=None)
    ap.add_argument('-b', '--batch_size', help='batch_size', default=32)
    ap.add_argument('-e', '--epochs', help='epochs', default=100)
    ap.add_argument('-s', '--save_path', help='save_path', default="saved/mobilenet-w-1st.h5")
    args = ap.parse_args()

    architecture = classify_zoo[args.architecture]["train"]
    train_fp = TrainFP(model=architecture, batch_size=args.batch_size, pretrained=args.pretrained)
    train_fp.train(epochs=args.epochs, save_path=args.save_path)
    train_fp.eval(weight_path=args.save_path)

    