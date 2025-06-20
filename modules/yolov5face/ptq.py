import os
import json
import keras
import numpy as np
from sklearn.utils import shuffle
import cv2
import scipy
from keras.models import Model
import sys
import argparse

sys.path.append("../../")
from ptq_base import BasePTQ

from onnx_to_h5 import SplitLayer


class PTQ(BasePTQ):
    def __init__(self, model, data_quantize, batch_size):
        super().__init__(model, data_quantize, batch_size)

    def get_all_range_quantize(self, batch_size):
        print("*** Init all-range dictionary : ")
        for i in range(len(self.model.layers)):
            name = self.model.layers[i].name
            self.range_dict[name] = np.array([10e6, -10e6])

        print("*** Infer batches and update all-range : ")
        len_data = len(self.data_quantize)
        if not batch_size:
            batch_size = len_data
        num_steps = len_data // batch_size

        for s in range(num_steps):
            print("--- ", s)
            for i in range(len(self.model.layers)):
                name = self.model.layers[i].name

                if "input" in name:
                    output = self.data_quantize[batch_size*s: batch_size*(s+1)]

                else:
                    try:
                        input_name = self.model.layers[i].input.name.split("/")[0]
                        input = self.output_dict[input_name]
                    except:
                        input_names = [inp.name.split("/")[0] for inp in self.model.layers[i].input]
                        input = [self.output_dict[inp_name] for inp_name in input_names]
                    output = self.model.layers[i](input).numpy()

                self.output_dict[name] = output

                min_update = min(np.min(output), self.range_dict[name][0])
                max_update = max(np.max(output), self.range_dict[name][1])
                self.range_dict[name] = np.array([min_update, max_update])

                if "activation" in name:        # sigmoid
                    self.range_dict[name] = np.array([0, 1])

    def quant_conv_silu(self, layer, name):
        layer_info = {}
        w = layer.get_weights()
        kernel = w[0]
        input_name = layer.input.name.split("/")[0]
        Zi, Si = self.scale_dict[input_name]
        Zk, Sk, kernel_quant = self.quant_uint8(kernel)

        # quantize output in range (-4, max_output)
        range_o = self.range_dict[name]
        range_o[0] = -4
        Zo1, So1, _ = self.quant_uint8(range_o)
        if len(w) == 1:
            bias_quant = np.zeros(shape=w[0].shape[-1])
        else:
            bias = w[1]
            S_bias = Sk * Si
            bias_quant = self.quant_bias(bias, S_bias)
        M1 = Sk * Si / So1
        M01, n1 = self.get_multiplier(M1)

        # quantize clip(output) in range (-4, 4)
        _, So2, _ = self.quant_uint8(np.array([-4, 4]))
        M2 = Sk * Si / So2
        M02, n2 = self.get_multiplier(M2)

        layer_info["input_shape"] = self.output_dict[input_name].shape[1:]
        layer_info["output_shape"] = self.output_dict[name].shape[1:]
        layer_info["filter_shape"] = kernel.shape
        layer_info["stride"] = layer.get_config()['strides']
        layer_info["padding"] = layer.get_config()['padding']
        layer_info["flt_off"] = int(Zk)
        layer_info["inp_off"] = int(Zi)
        layer_info["out1_off"] = int(Zo1)
        layer_info["multiplier1"] = int(M01)
        layer_info["shift1"] = int(n1)
        layer_info["multiplier2"] = int(M02)
        layer_info["shift2"] = int(n2)
        layer_info["kernel"] = kernel_quant.tolist()
        layer_info["bias"] = bias_quant.tolist()

        self.scale_dict[name] = [Zo1, So1, 0, So2]                          # Zo2 = 127,5 but not use so put Zo2 = 0

        return layer_info

    def quant_silu(self, layer, name_sig, name_mul):
        layer_info = {}
        input_name = layer.input.name.split("/")[0]
        Zi1, Si1, _, Si2 = self.scale_dict[input_name]

        a = -1/32
        S_sig = a * (Si2 ** 2)
        qc = round((1/2) / S_sig)  # = qd = -17987
        qb = 255
        range_sig = np.array([0, 1])
        _, S_, _ = self.quant_uint8(range_sig / S_sig)
        print("--- S_ ---: ", S_)
        M_ = 1 / S_                                                         # M_ * q_erf_int32 = q_erf_int8
        M_, n_ = np.frexp(M_)
        M0_, n_ = round(M_ * (2**31)), int(n_)                              # q_erf_int8 = (q_erf_int32*M0_) >> (31-n_)

        range_o = self.range_dict[name_mul]
        Zo, So, _ = self.quant_uint8(range_o)
        S_final = (Si1 * S_sig * S_) / So                             # cuz So*qo = 1/2 * Si1*qi1 * S_erf8*q_erf8
        M, n = np.frexp(S_final)
        M0, n = round(M * (2 ** 31)), int(n)

        layer_info["input_shape"] = self.output_dict[input_name].shape[1:]
        layer_info["output_shape"] = self.output_dict[input_name].shape[1:]
        layer_info["qb"] = int(qb)
        layer_info["qc"] = int(qc)
        layer_info["inp1_off"] = int(Zi1)
        layer_info["sig_off"] = int(255)
        layer_info["sig_multiplier"] = int(M0_)
        layer_info["sig_shift"] = int(n_)
        layer_info["out_off"] = int(Zo)
        layer_info["out_multiplier"] = int(M0)
        layer_info["out_shift"] = int(n)

        self.scale_dict[name_mul] = [Zo, So]

        return layer_info

    def quantize_model(self):
        conv_silu_names = []
        for layer in self.model.layers:
            name = layer.name
            if "activation" in name:
                input_name = layer.input.name.split("/")[0]
                conv_silu_names.append(input_name)

        all_layer_info = {}
        '''quantize backbone'''
        for layer in self.model.layers:
            name = layer.name
            print(name)

            if "input" in name:
                range_o = self.output_dict[name]
                Z, S, q = self.quant_uint8(range_o)  # Z, S, _ = self.quant_uint8(output)
                self.scale_dict[name] = [Z, S]
                layer_info = {}

            elif "conv" in name and name not in conv_silu_names:
                layer_info = self.quant_conv(layer, name)

            elif "add" in name and "padding" not in name:
                layer_info = self.quant_add(layer, name)

            elif "concatenate" in name:
                layer_info = self.quant_concat(layer, name)

            elif name in conv_silu_names:
                layer_info = self.quant_conv_silu(layer, name)

            elif "activation" in name:
                name_mul = name.replace("activation", "multiply")
                layer_info = self.quant_silu(layer, name, name_mul)

            elif "multiply" in name:
                layer_info = {}

            else:       # Padding, Pooling, Reshape, Transpose, ...
                input_name = layer.input.name.split("/")[0]
                self.scale_dict[name] = self.scale_dict[input_name]
                layer_info = {}

            all_layer_info[name] = layer_info
            print("-----------------")

        for layer in self.model.layers:
            name = layer.name
            if "add_" in name:
                print(f"--- check scale {name}:", self.scale_dict[name])

        return all_layer_info

    def __call__(self, json_path):
        all_layer_info = self.quantize_model()
        with open(json_path, "w") as f:
            json.dump(all_layer_info, f)


def get_data_calib(dataset_dir, num_calib=200, size=640):
    """Get random minidata for calib all ranges"""
    names = os.listdir(dataset_dir)
    inp_img = [os.path.join(dataset_dir, name) for name in names]
    # inp_img = shuffle(inp_img)
    inp_img = inp_img[:num_calib]
    inp_img = [cv2.resize(cv2.imread(path), (size, size)) / 255. for path in inp_img]
    inp_img = np.array(inp_img)

    return inp_img


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--file_model', help='file_model', default="saved/sub_model.h5")
    ap.add_argument('-j', '--save_json', help='save_json', default="saved/sub_model.json")
    args = ap.parse_args()

    model = keras.models.load_model(args.file_model, custom_objects={"SplitLayer": SplitLayer})
    data_calib = get_data_calib("datasets/calib_b", num_calib=200, size=640)

    ptq = PTQ(model, data_calib, batch_size=20)
    ptq(args.save_json)

