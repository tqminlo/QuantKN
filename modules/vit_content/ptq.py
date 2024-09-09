import os
import json
import numpy as np
from sklearn.utils import shuffle
import cv2
import scipy
from keras.models import Model
import sys
import argparse
sys.path.append("../../")
from ptq_base import BasePTQ
from models.dat_model import DAT


vit_zoo = {"dat_v1": {"fuse": DAT()()}
           }


class PTQ(BasePTQ):
    def __init__(self, model, data_quantize, batch_size):
        super().__init__(model, data_quantize, batch_size)

    def get_all_range_quantize(self, batch_size):
        print("*** Init all-range dictionary : ")
        for i in range(len(self.model.layers)):
            name = self.model.layers[i].name
            self.range_dict[name] = np.array([10e6, -10e6])
            if "gelu" in name:
                self.range_dict[name+"_erf"] = np.array([10e6, -10e6])

        print("*** Infer batches and update all-range : ")
        len_data = len(self.data_quantize[0])
        if not batch_size:
            batch_size = len_data
        num_steps = len_data // batch_size

        for s in range(num_steps):
            print("--- ", s)
            for i in range(len(self.model.layers)):
                name = self.model.layers[i].name

                if name == "input_1":
                    output = self.data_quantize[0][batch_size*s: batch_size*(s+1)]

                elif name == "input_2":
                    output = self.data_quantize[1][batch_size*s: batch_size*(s+1)]

                elif name == "input_3":
                    output = self.data_quantize[2][batch_size*s: batch_size*(s+1)]

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
                if "gelu" in name:
                    input_name = self.model.layers[i].input.name.split("/")[0]
                    input = self.output_dict[input_name]
                    erf = 1 + scipy.special.erf(input / np.sqrt(2))
                    min_update = min(np.min(erf), self.range_dict[name+"_erf"][0])
                    max_update = max(np.max(erf), self.range_dict[name+"_erf"][1])
                    self.range_dict[name+"_erf"] = np.array([min_update, max_update])

    def quant_matmul(self, layer, name):
        layer_info = {}
        inputs = layer.input
        input1_name = inputs[0].name.split("/")[0]
        input2_name = inputs[1].name.split("/")[0]
        # print("inp name of matmul layer: ", input1_name, input2_name)
        Z1, S1 = self.scale_dict[input1_name]
        Z2, S2 = self.scale_dict[input2_name]
        range_o = self.range_dict[name]
        Zo, So, _ = self.quant_uint8(range_o)
        if name == "block0_matmul_qkv":
            print("--- check: Z1, S1: ", Z1, S1)
            print("--- check: Z2, S2: ", Z2, S2)
            print("--- check: Zo, So: ", Zo, So)
        # Zo, So = self.scale_dict(name)
        M = S1 * S2 / So
        M0, n = self.get_multiplier(M)

        layer_info["input1_shape"] = self.output_dict[input1_name].shape[1:]
        layer_info["input2_shape"] = self.output_dict[input2_name].shape[1:]
        layer_info["output_shape"] = self.output_dict[name].shape[1:]
        layer_info["inp1_off"] = int(Z1)
        layer_info["inp2_off"] = int(Z2)
        layer_info["out_off"] = int(Zo)
        layer_info["multiplier"] = int(M0)
        layer_info["shift"] = int(n)

        self.scale_dict[name] = [Zo, So]

        return layer_info

    def quant_matmul_qk(self, layer, name):
        """
        Target output fp32: clip(-5, 0)((q@k_t / 8) - max(q@k_t / 8)) ---> range output: (-8, 0)
        """
        layer_info = {}
        inputs = layer.input
        input1_name = inputs[0].name.split("/")[0]
        input2_name = inputs[1].name.split("/")[0]
        Z1, S1 = self.scale_dict[input1_name]
        Z2, S2 = self.scale_dict[input2_name]
        range_o = self.range_dict[name]
        Zo2, So2, _ = self.quant_uint8(range_o)
        Zo1, So1 = 255, 8/255.                                          # quantize for only range (-8, 0)
        M2 = S1 * S2 / (So2 * 8)                                        # for (q@k_t)/8
        M02, n2 = self.get_multiplier(M2)
        M1 = S1 * S2 / (So1 * 8)
        M01, n1 = self.get_multiplier(M1)
        M_qmax, n_qmax = np.frexp(So2/So1)
        M0_qmax = min(round(M_qmax * 256), 255)                           # M0_qmax is represented by 8 bits
        n_qmax = int(n_qmax - 8)

        layer_info["input1_shape"] = self.output_dict[input1_name].shape[1:]
        layer_info["input2_shape"] = self.output_dict[input2_name].shape[1:]
        layer_info["output_shape"] = self.output_dict[name].shape[1:]
        layer_info["inp1_off"] = int(Z1)
        layer_info["inp2_off"] = int(Z2)
        layer_info["out1_off"] = int(Zo1)
        layer_info["multiplier1"] = int(M01)
        layer_info["shift1"] = int(n1)
        layer_info["out2_off"] = int(Zo2)
        layer_info["multiplier2"] = int(M02)
        layer_info["shift2"] = int(n2)
        layer_info["scale_max"] = int(M0_qmax)
        layer_info["shift_max"] = int(n_qmax)

        self.scale_dict[name] = [Zo1, So1]                              # Get only scale in range (-8,0)

        return layer_info

    def quant_softmax(self, layer, name):
        layer_info = {}
        input_name = layer.input.name.split("/")[0]
        layer_info["input_shape"] = self.output_dict[input_name].shape[1:]
        layer_info["output_shape"] = self.output_dict[input_name].shape[1:]
        layer_info["inp_off"] = 255
        layer_info["out_off"] = 0

        # approximate e^x
        layer_info["qb1"] = 255                                         # for 0 <= x < 128
        layer_info["qc1"] = 0
        layer_info["multiplier1"] = 1880595030
        layer_info["shift1"] = -11
        layer_info["qb2"] = 129                                         # for 128 <= x < 192
        layer_info["qc2"] = 747
        layer_info["multiplier2"] = 1884245926
        layer_info["shift2"] = -7
        layer_info["qb3"] = 80                                          # for 192 <= x < 224
        layer_info["qc3"] = 931
        layer_info["multiplier3"] = 1981416020
        layer_info["shift3"] = -5
        layer_info["qb4"] = 48                                          # for 224 <= x <= 255
        layer_info["qc4"] = 931
        layer_info["multiplier4"] = 1346511827
        layer_info["shift4"] = -3

        self.scale_dict[name] = [0, 1/65535]                              # range fp32: (0, 1)

        return layer_info

    def quant_dense_gelu(self, layer, name):
        layer_info = {}
        w = layer.get_weights()
        kernel = w[0]
        input_name = layer.input.name.split("/")[0]
        Zi, Si = self.scale_dict[input_name]
        Zk, Sk, kernel_quant = self.quant_uint8(kernel)

        # quantize org output
        range_o = self.range_dict[name]
        Zo1, So1, _ = self.quant_uint8(range_o)
        if len(w) == 1:
            bias_quant = np.zeros(shape=w[0].shape[-1])
        else:
            bias = w[1]
            S_bias = Sk * Si
            bias_quant = self.quant_bias(bias, S_bias)
        M1 = Sk * Si / So1
        M01, n1 = self.get_multiplier(M1)

        # quantize clip(output/sqrt(2)) in range (-1.769, 1.769)
        _, So2, _ = self.quant_uint8(np.array([-1.769, 1.769]))
        M2 = Sk * Si / (np.sqrt(2) * So2)
        M02, n2 = self.get_multiplier(M2)

        layer_info["input_shape"] = self.output_dict[input_name].shape[1:]
        layer_info["output_shape"] = self.output_dict[name].shape[1:]
        layer_info["filter_shape"] = kernel.shape
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

    def quant_gelu(self, layer, name):
        layer_info = {}
        input_name = layer.input.name.split("/")[0]

        Zi1, Si1, _, Si2 = self.scale_dict[input_name]
        range_o = self.range_dict[name]
        Zo, So, _ = self.quant_uint8(range_o)
        a = -0.2888
        S_erf = a * (Si2 ** 2)
        qc = round(1 / S_erf)  # = qd = -17987
        qb = 255
        range_erf = self.range_dict[name+"_erf"]
        _, S_, _ = self.quant_uint8(range_erf / S_erf)
        print("--- S_ ---: ", S_)

        M_ = 1 / S_                                                         # M_ * q_erf_int32 = q_erf_int8
        M_, n_ = np.frexp(M_)
        M0_, n_ = round(M_ * (2**31)), int(n_)                              # q_erf_int8 = (q_erf_int32*M0_) >> (31-n_)
        S_final = (Si1 * S_erf * S_) / (2 * So)                             # cuz So*qo = 1/2 * Si1*qi1 * S_erf8*q_erf8
        M, n = np.frexp(S_final)
        M0, n = round(M * (2 ** 31)), int(n)

        layer_info["input_shape"] = self.output_dict[input_name].shape[1:]
        layer_info["output_shape"] = self.output_dict[input_name].shape[1:]
        layer_info["qb"] = int(qb)
        layer_info["qc"] = int(qc)
        layer_info["inp1_off"] = int(Zi1)
        layer_info["erf_off"] = int(255)
        layer_info["erf_multiplier"] = int(M0_)
        layer_info["erf_shift"] = int(n_)
        layer_info["out_off"] = int(Zo)
        layer_info["out_multiplier"] = int(M0)
        layer_info["out_shift"] = int(n)

        self.scale_dict[name] = [Zo, So]

        return layer_info

    def quant_ln(self, layer, name):
        layer_info = {}
        input_name = layer.input.name.split("/")[0]
        Zi, Si = self.scale_dict[input_name]
        Mi, ni = np.frexp(Si)
        M0i, ni = round(Mi * (2**31)), int(ni)

        range_o = self.range_dict[name]
        Zo, So, _ = self.quant_uint8(range_o)
        mul_extra = round(1 / So)
        M_extra, n_extra = np.frexp(mul_extra)
        M0_extra, n_extra = round(M_extra * (2 ** 23)), int(n_extra - 23)
        print("--- check mul extra: ", M0_extra, n_extra)

        layer_info["inp_multiplier"] = int(M0i)
        layer_info["inp_shift"] = int(ni)
        layer_info["inp_off"] = int(Zi)
        layer_info["extra_multiplier"] = int(M0_extra)
        layer_info["extra_shift"] = int(n_extra)
        layer_info["out_off"] = int(Zo)

        self.scale_dict[name] = [Zo, So]

        return layer_info

    def quantize_model(self):
        """like basic, but range softmax is (-8, 0) and output u16"""
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

            elif any(check in name for check in ["conv", "Conv"]):
                layer_info = self.quant_conv(layer, name)

            elif "add" in name and "padding" not in name:
                layer_info = self.quant_add(layer, name)

            elif "concatenate" in name:
                layer_info = self.quant_concat(layer, name)

            elif "matmul_qk" in name and "qkv" not in name:
                layer_info = self.quant_matmul_qk(layer, name)

            elif "matmul" in name:
                layer_info = self.quant_matmul(layer, name)

            elif "softmax" in name:
                layer_info = self.quant_softmax(layer, name)

            elif "fc1._MatMulAdd" in name:                                          # quantize for Dense-before-Gelu
                layer_info = self.quant_dense_gelu(layer, name)

            elif any(check in name for check in ["Dense", "MatMulAdd"]):            # quantize for normal Dense
                layer_info = self.quant_dense(layer, name)

            elif "gelu" in name:
                layer_info = self.quant_gelu(layer, name)

            elif "LayerNormalization" in name:
                layer_info = self.quant_ln(layer, name)

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


def get_data_calib(dataset_dir, num_calib=200, size=518):
    """Get random minidata for calib all ranges"""
    names = os.listdir(dataset_dir)
    inp_img = [os.path.join(dataset_dir, name) for name in names]
    inp_img = shuffle(inp_img)
    inp_img = inp_img[:num_calib]
    inp_img = [cv2.resize(cv2.imread(path), (size, size)) / 255. for path in inp_img]

    inp_img = np.array(inp_img)
    inp_expand = np.tile(np.load("saved/inp1.npy"), reps=(num_calib, 1, 1))
    inp_pos = np.tile(np.load("saved/inp2.npy"), reps=(num_calib, 1, 1))
    print(inp_img.shape)
    print(inp_expand.shape)
    print(inp_pos.shape)

    return [inp_img, inp_expand, inp_pos]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-a', '--architecture', help='architecture', default="dat_v1")
    ap.add_argument('-w', '--weights', help='weights', default="saved/dat_v1.3_w.h5")
    ap.add_argument('-j', '--save_json', help='save_json', default="saved/dat-5th.2.json")
    args = ap.parse_args()

    model = vit_zoo[args.architecture]["fuse"]
    model.load_weights(args.weights)
    data_calib = get_data_calib("datasets/calib", num_calib=180, size=518)

    ptq = PTQ(model, data_calib, batch_size=3)
    ptq(args.save_json)

