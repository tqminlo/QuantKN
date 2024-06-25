import os
import keras
import sys
import numpy as np
sys.path.append("../../")
from infer_int_base import InferIntBase
from models.mobilenet_minimalistic import MobileNet
from keras.models import Model
import argparse


# model = MobileNet(224)()
model = keras.models.load_model("saved/float-1st.h5")
model = Model(model.input, model.layers[-2].output)
json_path = "saved/1st.json"
infer_int = InferIntBase(model, json_path, size=224)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--inp_path', help='image path')
    args = ap.parse_args()

    img_path = args.inp_path
    all_output = infer_int.infer(img_path)
    last_layer = [model.output.name.split("/")[0]]
    output = all_output[last_layer]
    output = output[0]
    idc = np.argmax(output)
    print("id:", idc)
    print("value:", output[idc])



