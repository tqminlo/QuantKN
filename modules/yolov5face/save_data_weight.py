import os
import json
import numpy as np
import cv2
import sys
sys.path.append("../../")
import argparse
from infer_ptq import save_txt


def save_all_cfgs_array(cfg_json_path):
    os.makedirs("saved/save_all_weights_array", exist_ok=True)
    with open(cfg_json_path) as f:
        cfgs = json.load(f)

    all_saved = {}
    all_keys = list(cfgs.keys())
    for i in range(len(all_keys)):
        key = all_keys[i]
        print("--- ", key)
        all_saved[key] = cfgs[key]

        if key in ["conv2d", "conv2d_2", "conv2d_4", "conv2d_7"]:
            kernel = np.array(all_saved[key]["kernel"])
            bias = np.array(all_saved[key]["bias"])
            all_saved[key]["kernel"] = save_txt(kernel)
            all_saved[key]["bias"] = save_txt(bias)

        elif key in ["conv2d_1", "conv2d_3", "conv2d_5", "conv2d_6", "conv2d_8"]:
            kernel = np.array(all_saved[key]["kernel"])
            bias = np.array(all_saved[key]["bias"])
            kernel = kernel.reshape((kernel.shape[0], kernel.shape[1], 16, kernel.shape[2] // 16, kernel.shape[3]))  #
            kernel = kernel.transpose(0, 1, 2, 4, 3)
            all_saved[key]["kernel"] = save_txt(kernel)
            all_saved[key]["bias"] = save_txt(bias)

        txt_path = f"saved/save_all_weights_array/layer{i:03}_{key}.txt"
        txt = ""
        for sub_key in all_saved[key].keys():
            txt += f"{sub_key}      : {all_saved[key][sub_key]}\n"
        with open(txt_path, "w") as f:
            f.write(txt)


def save_inp(inp_path):
    inp = cv2.imread(inp_path)
    inp = cv2.resize(inp, (640, 640))
    print(inp)
    save_txt(inp, "saved/save_all_weights_array/layer000_input_1.txt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--img_path', help='image path', default="saved/test.jpg")
    ap.add_argument('-f', '--file_model', help='file_model', default="saved/sub_model.h5")
    ap.add_argument('-j', '--json_path', help='json_path', default="saved/sub_model.json")
    args = ap.parse_args()

    save_inp(args.img_path)
    save_all_cfgs_array(args.json_path)