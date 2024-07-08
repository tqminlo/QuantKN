import os

import cv2
import keras
import sys
import numpy as np
sys.path.append("../../")
from infer_int_base import InferIntBase
from models.mobilenet_minimalistic import MobileNet
from keras.models import Model
import argparse
from train import classify_zoo, SIZE, val_dir
from ptq import remove_softmax_layer


class InferInt(InferIntBase):
    def __init__(self, model, json_path, size):
        super().__init__(model, json_path, size)

    def infer_img(self, img_path):
        all_output = self.infer(img_path)
        last_layer = [self.model.output.name.split("/")[0]]
        output = all_output[last_layer]
        output = output[0]
        idc = np.argmax(output)
        print("id:", idc)
        print("value:", output[idc])

    def eval(self, data_dir=val_dir, batch_size=32):
        last_layer = [self.model.output.name.split("/")[0]]
        inputs = []
        labels = []
        classes = os.listdir(data_dir)
        label = 0

        for class_name in classes:
            class_dir = os.path.join(val_dir, class_name)
            images = os.listdir(class_dir)
            image_paths = [os.path.join(class_dir, img) for img in images]
            sub_labels = [label for img in images]
            inputs += image_paths
            labels += sub_labels
            label += 1

        num_steps = len(inputs) // batch_size
        score = 0
        num_val = len(inputs)
        for i in range(num_steps):
            inps_batch = inputs[batch_size*i: batch_size*(i+1)] if i < num_steps-1 else inputs[batch_size*i:]
            labels_batch = labels[batch_size*i: batch_size*(i+1)] if i < num_steps-1 else labels[batch_size*i:]
            inps_batch = [cv2.resize(cv2.imread(path), (SIZE, SIZE)) for path in inps_batch]
            inps_batch = np.array(inps_batch)
            labels_batch = np.array(labels_batch)

            all_output = self.infer(inps_batch)
            outs_batch = all_output[last_layer]
            idc_batch = np.argmax(outs_batch, axis=1)

            for j in range(len(idc_batch)):
                if idc_batch[j] == labels_batch[j]:
                    score += 1

            print("----", i, "----")

        acc = score / num_val
        print("acc: ", acc)

        return acc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-t', '--type', help='infer image or evaluate, "infer", "eval"', default="infer")
    ap.add_argument('-i', '--inp_path', help='image path with "infer" or directory with "eval"')
    ap.add_argument('-a', '--architecture', help='architecture', default="mobilenet")
    ap.add_argument('-j', '--json_path', help='json_path', default="saved/mobilenet-1st.json")
    args = ap.parse_args()

    model = classify_zoo[args.architecture]["fuse"]
    model = remove_softmax_layer(model)

    infer = InferInt(model, args.json_path, size=SIZE)
    if args.type == "infer":
        infer.infer_img(args.inp_path)
    else:
        infer.eval(args.inp_path)






