import os
import json
import numpy as np
from keras.layers import *
import cv2
import sys
sys.path.append("../../")
from infer_ptq_base import InferPTQBase
from ptq_old import vit_zoo
import argparse


def save_txt(tensor, txt_path=None):
    tensor = tensor.flatten()
    # text = str(list(tensor)).replace("[", "").replace("]", "")
    text = ""
    num_lines = len(tensor) // 1370
    for i in range(num_lines):
        text += (str(list(tensor[1370 * i: 1370 * (i + 1)])).replace("[", "").replace("]", "") + ", \n")
    if len(tensor) > num_lines * 1370:
        text += (str(list(tensor[1370 * num_lines:])).replace("[", "").replace("]", ""))
    if txt_path:
        with open(txt_path, 'w') as f:
            f.write(text)

    return text


def save_all_outputs(all_outputs):
    all_names = list(all_outputs.keys())
    for i in range(len(all_names)):
        layer_name = all_names[i]
        # if layer_name not in list_save:          # chi save some layers
        #     continue
        output = all_outputs[layer_name]

        if layer_name == "input_1":
            output = output.transpose((0, 3, 1, 2))         # batch, C, H, W

        elif layer_name == "input_3":
            shape = output.shape
            div = shape[2] / 8
            new_dim2 = int(np.ceil((div - 1) / 2) * 2 + 1) * 8
            print("-------- ", shape[2], new_dim2)
            new_shape = (shape[0], shape[1], new_dim2)
            output_pad = np.ones(shape=new_shape, dtype=int) * 144
            output_pad[:, :, :shape[2]] = output
            output = output_pad

        if not isinstance(output, list):
            print(output.shape)
            txt_path = f"saved/save_all_outputs/out{i:03}_{layer_name}.txt"
            save_txt(output, txt_path)

        else:
            for j in range(len(output)):
                print(output[j].shape)
                txt_path = f"saved/save_all_outputs/out{i:03}_{j}_{layer_name}.txt"
                save_txt(output[j], txt_path)
        print("---", layer_name, "Done!")


def save_all_cfgs(cfg_json_path):
    with open(cfg_json_path) as f:
        cfgs = json.load(f)

    all_saved = {}
    all_keys = list(cfgs.keys())
    for i in range(len(all_keys)):
        key = all_keys[i]
        print("--- ", key)
        all_saved[key] = cfgs[key]

        if key == "._patch_embed._proj._Conv":
            kernel = np.array(all_saved[key]["kernel"])
            kernel = kernel.transpose((0, 2, 1, 3))                 # fy, ic, fx, oc
            bias = np.array(all_saved[key]["bias"])
            all_saved[key]["kernel"] = save_txt(kernel)
            all_saved[key]["bias"] = save_txt(bias)

        elif key == "._depth_head._output_conv2._output_conv2.2._Conv":
            kernel = np.array(all_saved[key]["kernel"])
            flt_off = all_saved[key]["flt_off"]
            new_shape = (1, 1, 32, 8)
            new_kernel = np.ones(shape=new_shape, dtype=int) * flt_off
            new_kernel[:, :, :, :1] = kernel
            fh, fw, ic, oc = new_kernel.shape
            kernel = new_kernel.reshape((fh, fw, ic, oc // 8, 8))
            kernel = kernel.transpose(3, 0, 1, 2, 4)
            print("-----check last cv-----:", kernel.shape)
            bias = np.array(all_saved[key]["bias"])
            all_saved[key]["kernel"] = save_txt(kernel)
            all_saved[key]["bias"] = save_txt(bias)

        elif "ConvTranspose" in key:
            kernel = np.array(all_saved[key]["kernel"])
            shape = kernel.shape
            txt_kernel = ""
            for fh in range(shape[0]):
                for fw in range(shape[1]):
                    sub_kernel = kernel[fh][fw]
                    sub_kernel = sub_kernel.transpose()
                    sub_kernel = sub_kernel.reshape((shape[2], shape[3]//8, 8))         # dam bao ix, oc chia het cho 8
                    sub_kernel = sub_kernel.transpose((1, 0, 2))
                    txt_kernel += save_txt(sub_kernel) + "\n\n"
            bias = np.array(all_saved[key]["bias"])
            all_saved[key]["kernel"] = txt_kernel
            all_saved[key]["bias"] = save_txt(bias)

        elif "Conv" in key:
            kernel = np.array(all_saved[key]["kernel"])
            fh, fw, ic, oc = kernel.shape
            kernel = kernel.reshape((fh, fw, ic, oc // 8, 8))
            kernel = kernel.transpose(3, 0, 1, 2, 4)
            bias = np.array(all_saved[key]["bias"])
            all_saved[key]["kernel"] = save_txt(kernel)
            all_saved[key]["bias"] = save_txt(bias)

        elif "MatMulAdd" in key:
            kernel = np.array(all_saved[key]["kernel"])
            flt_off = all_saved[key]["flt_off"]
            shape = kernel.shape
            div = shape[1] / 8
            new_dim1 = int(np.ceil((div - 1) / 2) * 2 + 1) * 8
            print("-------- ", shape[1], new_dim1)
            new_shape = (shape[0], new_dim1)
            new_kernel = np.ones(shape=new_shape, dtype=int) * flt_off
            new_kernel[:, :shape[1]] = kernel
            bias = np.array(all_saved[key]["bias"])
            all_saved[key]["kernel"] = save_txt(new_kernel)
            all_saved[key]["bias"] = save_txt(bias)

        txt_path = f"saved/save_all_weights/layer{i:03}_{key}.txt"
        txt = ""
        for sub_key in all_saved[key].keys():
            txt += f"{sub_key}      : {all_saved[key][sub_key]}\n"
        with open(txt_path, "w") as f:
            f.write(txt)


class InferPTQ(InferPTQBase):
    """
    Infer integer-only from json-file and model graph.
    """
    def __init__(self, model, json_path, size=518):
        super().__init__(model, json_path, size)
        self.inp1 = np.load("saved/q_inp1.npy")
        self.inp2 = np.load("saved/q_inp2.npy")

    def matmul_qk_infer_quant(self, layer, name, all_output):
        data_quant = self.model_quant[name]
        Z1 = data_quant["inp1_off"]
        Z2 = data_quant["inp2_off"]
        Zo1 = data_quant["out1_off"]
        M01 = data_quant["multiplier1"]
        n1 = data_quant["shift1"]
        Zo2 = data_quant["out2_off"]
        M02 = data_quant["multiplier2"]
        n2 = data_quant["shift2"]
        M0_qmax = data_quant["scale_max"]
        n_qmax = data_quant["shift_max"]

        inputs = layer.input
        input_name1 = inputs[0].name.split("/")[0]
        input_name2 = inputs[1].name.split("/")[0]
        print("--", input_name1)
        print("--", input_name2)
        inp1 = all_output[input_name1].astype(float)
        inp2 = all_output[input_name2].astype(float)

        x2 = (inp1 - Z1) @ (inp2 - Z2)
        x2 = np.floor(x2 * M02 / 2147483648 + 0.5)
        x2 = (x2 / pow(2, -n2) + np.sign(x2) * 0.5).astype(int)
        x2 += Zo2
        x2 = np.clip(x2, 0, 255)
        x_max = np.max(x2, axis=-1, keepdims=True)
        x_max_scale = x_max - Zo2
        x_max_scale = (x_max_scale * M0_qmax / (2 ** -n_qmax) + np.sign(x_max_scale) * 0.5).astype(int)         ###
        # x_max_scale = (x_max_scale * M0_qmax / (2 ** -n_qmax) + 0.5).astype(int)  ###

        x1 = (inp1 - Z1) @ (inp2 - Z2)
        x1 = np.floor(x1 * M01 / 2147483648 + 0.5)
        x1 = (x1 / pow(2, -n1) + np.sign(x1) * 0.5).astype(int)
        x1 = x1 - x_max_scale
        # x1 = x1 - np.max(x1, axis=-1, keepdims=True)
        x1 += Zo1                                               # x1 += 255 (Zo1 = 255)
        x1 = np.clip(x1, 0, 255)

        return x1

    def matmul_infer_quant(self, layer, name, all_output):
        data_quant = self.model_quant[name]
        Z1 = data_quant["inp1_off"]
        Z2 = data_quant["inp2_off"]
        Zo = data_quant["out_off"]
        M0 = data_quant["multiplier"]
        n = data_quant["shift"]

        inputs = layer.input
        input_name1 = inputs[0].name.split("/")[0]
        input_name2 = inputs[1].name.split("/")[0]
        print("--", input_name1)
        print("--", input_name2)
        inp1 = all_output[input_name1].astype(float)
        inp2 = all_output[input_name2].astype(float)

        x = (inp1 - Z1) @ (inp2 - Z2)
        # print("--- check min, max qi*qv:", np.min(x), np.max(x))
        # if "matmul_qkv" in name:
        #     M0 = M0 / 257.
        x = np.floor(x * M0 / 2147483648 + 0.5)
        x = (x / pow(2, -n) + np.sign(x) * 0.5).astype(int)
        x = x + Zo
        # print("--- check min, max x:", np.min(x), np.max(x))
        # print("--- check x:", x[0][0][1000:1010, 30:40])
        x = np.clip(x, 0, 255)
        return x

    def softmax_infer_quant(self, layer, name, all_output):
        data_quant = self.model_quant[name]
        Zi = data_quant["inp_off"]                          # 255
        Zo = data_quant["out_off"]                          # 0
        qb1 = data_quant["qb1"]                             #
        qc1 = data_quant["qc1"]
        M01 = data_quant["multiplier1"]
        n1 = data_quant["shift1"]
        qb2 = data_quant["qb2"]                             #
        qc2 = data_quant["qc2"]
        M02 = data_quant["multiplier2"]
        n2 = data_quant["shift2"]
        qb3 = data_quant["qb3"]                             #
        qc3 = data_quant["qc3"]
        M03 = data_quant["multiplier3"]
        n3 = data_quant["shift3"]
        qb4 = data_quant["qb4"]  #
        qc4 = data_quant["qc4"]
        M04 = data_quant["multiplier4"]
        n4 = data_quant["shift4"]

        input_name = layer.input.name.split("/")[0]
        print("--", input_name)
        qi = all_output[input_name].astype(float)

        sign1 = np.clip(np.sign(qi - 128), None, 0) * (-1)
        sign2 = np.clip(np.sign(qi - 192), None, 0) * (-1) - sign1
        sign3 = np.clip(np.sign(qi - 224), None, 0) * (-1) - sign1 - sign2
        sign4 = 1 - sign1 - sign2 - sign3
        qb = sign1 * qb1 + sign2 * qb2 + sign3 * qb3 + qb4 * sign4
        qc = sign1 * qc1 + sign2 * qc2 + sign3 * qc3 + qc4 * sign4
        M0 = sign1 * M01 + sign2 * M02 + sign3 * M03 + M04 * sign4
        n = sign1 * n1 + sign2 * n2 + sign3 * n3 + n4 * sign4
        qex = ((qi - 255 + qb) ** 2 + qc).astype(float)
        qex = np.floor(qex * M0 / 2147483648 + 0.5)
        qex = (qex / pow(2, -n) + np.sign(qex) * 0.5).astype(int)
        qex = np.clip(qex, 0, 255)
        # print("--- check qi: ", qi[0, 0, 0, :])                                   # head 0, row 1
        # print("--- check qex: ", qex[0, 0, 0, :])                                 # head 0, row 1
        qex_sum = np.sum(qex, axis=-1, keepdims=True)

        div = ((2 ** 31) / qex_sum)
        bit_shift_left = np.log2(div)
        bit_shift_left = (np.ceil(bit_shift_left) - 1).astype(int)
        d = qex_sum * pow(2, bit_shift_left)
        d = d.astype(np.int64)
        const_48div17 = 1515870810
        const_n32div17 = -1010580540
        x = const_48div17 + np.floor((const_n32div17 * d) / (1 << 31) + 0.5)
        for i in range(3):
            mul_xd = np.floor((x * d) / (2 ** 31) + 0.5)
            sub_by1 = (1 << 29) - mul_xd
            x = x + (np.floor((x * sub_by1) / (2 ** 31) + 0.5)) * (2 ** 2)          ###
            # x = x + (np.floor((x * sub_by1) / (2 ** 29) + 0.5))                   ###
        # print("--- check x: ", x[:, 0, :5, :])                                    # head 0, first 5 row
        shift_right = 29 - bit_shift_left
        Msum = np.floor((x / (2 ** shift_right) + 0.5))

        qy = np.floor((qex * 65535) * Msum / (2 ** 31) + 0.5)
        # print("--- check qy: ", qy[0, 0, 0, :])                                   # head 0, row 1
        # print("--- check min, max qy:", np.min(qy), np.max(qy))
        qsum = np.sum(qy, axis=-1)
        print("--- check min, max qsum:", np.min(qsum), np.max(qsum))
        qy = np.clip(qy, 0, 65535).astype(int)

        return qy

    def dense_gelu_infer_quant(self, layer, name, all_output):
        data_quant = self.model_quant[name]
        Zi = data_quant["inp_off"]
        Zk = data_quant["flt_off"]
        Zo1 = data_quant["out1_off"]
        M01 = data_quant["multiplier1"]
        n1 = data_quant["shift1"]
        M02 = data_quant["multiplier2"]
        n2 = data_quant["shift2"]                               # Zo2 = 127.5 but not care
        kernel = np.array(data_quant["kernel"])
        bias = np.array(data_quant["bias"])
        # print("--- check update: ", Zi, M01, n1, M02, n2)

        input_name = layer.input.name.split("/")[0]
        print("--", input_name)
        inp = all_output[input_name].astype(float)
        # if len(layer.get_weights()) == 2:
        #     layer.set_weights([kernel - Zk, bias])
        # else:
        #     layer.set_weights([kernel - Zk])

        # x = inp - Zi
        # x = layer(x).numpy()
        x = (inp - Zi) @ (kernel - Zk) + bias
        x1 = np.floor(x * M01 / 2147483648 + 0.5)
        x1 = (x1 / pow(2, -n1) + np.sign(x1) * 0.5).astype(int)
        x1 = x1 + Zo1
        print("--- check Zo1:", Zo1)
        print("--- check x1:", np.min(x1), np.max(x1))
        x1 = np.clip(x1, 0, 255)

        x2 = np.floor(x * M02 / 2147483648 + 0.5)
        x2 = (x2 / pow(2, -n2) + np.sign(x2) * 0.5).astype(int)
        x2 = x2 + 127
        print("--- check x2:", np.min(x2), np.max(x2))
        x2 = np.clip(x2, 0, 255)

        return [x1, x2]

    def gelu_infer_quant(self, layer, name, all_output):
        data_quant = self.model_quant[name]
        qb = data_quant["qb"]                                   # 255
        qc = data_quant["qc"]                                   # -17987
        Z_erf = data_quant["erf_off"]                           # 255
        M0_erf = data_quant["erf_multiplier"]
        n_erf = data_quant["erf_shift"]
        Zi1 = data_quant["inp1_off"]
        Zo = data_quant["out_off"]
        M0 = data_quant["out_multiplier"]
        n = data_quant["out_shift"]

        input_name = layer.input.name.split("/")[0]
        print("--", input_name)
        inp = all_output[input_name]
        q1, q2 = inp[0].astype(float), inp[1].astype(float)

        q_erf = np.sign(q2 - 127.5) * ((q2 - (q2 // 128) * qb) ** 2 + qc)
        q_erf = q_erf + qc
        q_erf = np.floor(q_erf * M0_erf / 2147483648 + 0.5)
        q_erf = (q_erf / (2 ** -n_erf) + np.sign(q_erf) * 0.5).astype(int)
        q_erf = q_erf + Z_erf
        q_erf = np.clip(q_erf, 0, 255)
        qo = (q1 - Zi1) * (q_erf - Z_erf)
        qo = np.floor(qo * M0 / 2147483648 + 0.5)
        qo = (qo / pow(2, -n) + np.sign(qo) * 0.5).astype(int)
        qo = qo + Zo
        qo = np.clip(qo, 0, 255)

        return qo

    def ln_infer_quant(self, layer, name, all_output):
        data_quant = self.model_quant[name]
        M0_extra = data_quant["extra_multiplier"]
        n_extra = data_quant["extra_shift"]
        Zo = data_quant["out_off"]

        input_name = layer.input.name.split("/")[0]
        print("--", input_name)
        inp = all_output[input_name]
        mean = np.average(inp, axis=-1, keepdims=True).astype(int)
        V = np.sum((inp - mean)**2, axis=-1, keepdims=True)             # V * 384
        V = np.floor(V / 384 + 0.5) * (2**16)
        std = (np.sqrt(V)).astype(int)                            # std * (2**8)
        # std = (np.std(inp, axis=-1, keepdims=True))
        # std = ((np.std(inp, axis=-1, keepdims=True) * (2**8)) + 0.5).astype(int)

        bit_shift_left = (30 - (np.log2(std)).astype(int))
        print("--- check bit_shift_left shape: ", bit_shift_left.shape)
        q_sd = (std * (2 ** bit_shift_left)).astype(np.int64)
        const_48div17 = 1515870810
        const_n32div17 = -1010580540
        x = const_48div17 + np.floor((q_sd * const_n32div17) / (2 ** 31) + 0.5)
        for i in range(3):
            mul_xd = np.floor((x * q_sd) / (2 ** 31) + 0.5)
            sub_by1 = (1 << 29) - mul_xd
            x = x + (np.floor((x * sub_by1) / (2 ** 31) + 0.5)) * (2 ** 2)

        n_final = (29 - n_extra - bit_shift_left) - 8
        out = np.int64((inp - mean) * M0_extra)
        out = np.floor(out * x / 2147483648 + 0.5)
        out = (out / pow(2, n_final) + np.sign(out) * 0.5).astype(int)
        # out = np.int64((inp - mean) / std)
        # out = (out * M0_extra / pow(2, -n_extra) + 0.5).astype(int)
        out = out + Zo
        print("--- check min, max out:", np.min(out), np.max(out))
        out = np.clip(out, 0, 255)

        return out

    def relu_infer_quant(self, layer, name, all_output):
        input_name = layer.input.name.split("/")[0]
        print("--", input_name)
        inp = all_output[input_name]
        inp_off = self.model_quant[input_name]["out_off"]
        print("-- check inp_off: ", inp_off)
        x = np.clip(inp, inp_off, 255)

        return x

    def infer(self, inp):
        """inp can be an image-path or a raw 3d-tensor or a 4d-tensor resized"""
        if isinstance(inp, str):
            inp = cv2.resize(cv2.imread(inp), (self.size, self.size))
            inp = np.expand_dims(inp, axis=0)
        elif len(inp.shape) != 4:
            inp = cv2.resize(inp, (self.size, self.size))
            inp = np.expand_dims(inp, axis=0)

        all_output = {}
        for i in range(len(self.model.layers)):
            layer = self.model.layers[i]
            name = layer.name
            print(name)

            if name == "input_1":
                output = inp
                all_output[name] = output
                print(np.min(output), np.max(output))

            elif name == "input_2":
                output = self.inp1
                all_output[name] = output
                print(np.min(output), np.max(output))

            elif name == "input_3":
                output = self.inp2
                all_output[name] = output
                print(np.min(output), np.max(output))

            elif any(check in name for check in ["conv", "Conv"]):
                output = self.conv_infer_quant(layer, name, all_output)
                print(np.min(output), np.max(output))

            elif "add" in name and "padding" not in name:
                output = self.add_infer_quant(layer, name, all_output)
                print(np.min(output), np.max(output))

            elif "concat" in name:
                output = self.concat_infer_quant(layer, name, all_output)
                print(np.min(output), np.max(output))

            elif "fc1._MatMulAdd" in name:
                output = self.dense_gelu_infer_quant(layer, name, all_output)
                print(np.min(output[0]), np.max(output[0]), np.min(output[1]), np.max(output[1]))

            elif any(check in name for check in ["Dense", "MatMulAdd"]):
                output = self.dense_infer_quant(layer, name, all_output)
                print(np.min(output), np.max(output))

            elif "gelu" in name:
                output = self.gelu_infer_quant(layer, name, all_output)
                print(np.min(output), np.max(output))

            elif "matmul_qk" in name and "qkv" not in name:
                output = self.matmul_qk_infer_quant(layer, name, all_output)
                print(np.min(output), np.max(output))

            elif "matmul" in name:
                output = self.matmul_infer_quant(layer, name, all_output)
                print(np.min(output), np.max(output))

            elif "softmax" in name:
                output = self.softmax_infer_quant(layer, name, all_output)
                print(np.min(output), np.max(output))

            elif "LayerNormalization" in name:
                output = self.ln_infer_quant(layer, name, all_output)
                print(np.min(output), np.max(output))

            elif "re_lu" in name:
                output = self.relu_infer_quant(layer, name, all_output)
                print(np.min(output), np.max(output))

            else:
                input_name = layer.input.name.split("/")[0]
                print("--", input_name)
                inp = all_output[input_name].astype(float)
                output = layer(inp).numpy()
                output = (output + np.sign(output) * 0.5).astype(int)  # Average layer can cause numbers to not int
                output = np.clip(output, 0, 255)
                print(np.min(output), np.max(output))

            all_output[name] = output
            print("***********")

        return all_output


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--img_path', help='image path')
    ap.add_argument('-a', '--architecture', help='architecture', default="dat_v1")
    ap.add_argument('-j', '--json_path', help='json_path', default="saved/dat-5th.json")
    args = ap.parse_args()

    # save_all_cfgs(args.json_path)

    model = vit_zoo[args.architecture]["fuse"]
    infer_ptq = InferPTQ(model, args.json_path)
    inp = cv2.imread(args.img_path)
    # inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
    org_h, org_w, _ = inp.shape

    all_output = infer_ptq.infer(inp)
    last_layer = infer_ptq.model.output.name.split("/")[0]
    depth = all_output[last_layer]
    depth = depth[0].astype(np.uint8)

    from infer_dat_fp32 import show_depth
    show_depth(depth, inp, (org_w, org_h))

    # save_all_outputs(all_output)