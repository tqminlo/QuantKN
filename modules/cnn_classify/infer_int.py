import os
import keras
import sys
import numpy as np
sys.path.append("../../")
from infer_int_base import InferIntBase
from models.mobilenet_minimalistic import MobileNet
from keras.models import Model
import argparse
from train import classify_zoo, SIZE
from ptq import remove_softmax_layer


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--inp_path', help='image path')
    ap.add_argument('-a', '--architecture', help='architecture', default="mobilenet")
    ap.add_argument('-j', '--json_path', help='json_path', default="saved/mobilenet-1st.json")
    args = ap.parse_args()

    model = classify_zoo[args.architecture]
    model = remove_softmax_layer(model)
    infer_int = InferIntBase(model, args.json_path, size=SIZE)
    img_path = args.inp_path

    all_output = infer_int.infer(img_path)
    last_layer = [model.output.name.split("/")[0]]
    output = all_output[last_layer]
    output = output[0]
    idc = np.argmax(output)
    print("id:", idc)
    print("value:", output[idc])



