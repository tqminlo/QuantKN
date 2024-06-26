import os
import json
import numpy as np
from sklearn.utils import shuffle
import cv2
from keras.models import Model
import sys
import argparse
sys.path.append("../../")
from ptq_base import BasePTQ

from train import SIZE, DATA_DIR, classify_zoo


def remove_softmax_layer(m):
    """"Remove last layer in model if it is softmax layer (no need quantize that layer)"""
    if "softmax" in m.layers[-1].name.split("/")[0]:
        m = Model(m.input, m.layers[-2].output)
    return m


def get_data_calib(dataset_dir=f"{DATA_DIR}/val", num_calib=200, size=SIZE):
    """Get random minidata for calib all ranges"""
    data = []
    all_classes = os.listdir(dataset_dir)
    num_per_class = num_calib // len(all_classes) + 1
    for label in all_classes:
        img_dir = os.path.join(dataset_dir, label)
        all_img_inside = os.listdir(img_dir)
        assert len(all_img_inside) >= num_per_class, "num data in this class not valid"
        for i in range(len(all_img_inside)):
            img_name = all_img_inside[i]
            img_path = os.path.join(img_dir, img_name)
            data.append(img_path)

    data = shuffle(data)
    data = data[:num_calib]
    data = [cv2.resize(cv2.imread(path), (size, size)) / 255. for path in data]
    data = np.array(data)

    return data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-a', '--architecture', help='architecture', default="mobilenet")
    ap.add_argument('-w', '--weights', help='weights', default="saved/mobilenet-w-1st.h5")
    ap.add_argument('-j', '--save_json', help='save_json', default="saved/mobilenet-1st.json")
    args = ap.parse_args()

    model = classify_zoo[args.architecture]["fuse"]

    model.load_weights(args.weights)
    model = remove_softmax_layer(model)
    data_calib = get_data_calib()

    ptq = BasePTQ(model, data_calib)
    ptq(args.json_path)

