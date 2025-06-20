import os
import json
import numpy as np
from keras.layers import *
import cv2
import sys
sys.path.append("../../")
from infer_ptq_base import InferPTQBase
import argparse
import keras


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
    os.makedirs("saved/save_all_outputs", exist_ok=True)
    all_names = list(all_outputs.keys())
    for i in range(len(all_names)):
        layer_name = all_names[i]
        output = all_outputs[layer_name]

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


class InferPTQ(InferPTQBase):
    """
    Infer integer-only from json-file and model graph.
    """
    def __init__(self, model, json_path, size=640):
        super().__init__(model, json_path, size)

    def conv_silu_infer_quant(self, layer, name, all_output):
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

        if len(layer.get_weights()) == 2:
            layer.set_weights([kernel - Zk, bias])
        else:
            layer.set_weights([kernel - Zk])

        x = inp - Zi
        x = layer(x).numpy()

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

    def silu_infer_quant(self, layer, name, all_output):
        data_quant = self.model_quant[name]
        qb = data_quant["qb"]                                   # 255
        qc = data_quant["qc"]                                   # -17987
        Z_sig = data_quant["sig_off"]                           # 255
        M0_sig = data_quant["sig_multiplier"]
        n_sig = data_quant["sig_shift"]
        Zi1 = data_quant["inp1_off"]
        Zo = data_quant["out_off"]
        M0 = data_quant["out_multiplier"]
        n = data_quant["out_shift"]

        input_name = layer.input.name.split("/")[0]
        print("--", input_name)
        inp = all_output[input_name]
        q1, q2 = inp[0].astype(float), inp[1].astype(float)

        q_sig = np.sign(q2 - 127.5) * ((q2 - (q2 // 128) * qb) ** 2 + qc)
        q_sig = q_sig + qc
        q_sig = np.floor(q_sig * M0_sig / 2147483648 + 0.5)
        q_sig = (q_sig / (2 ** -n_sig) + np.sign(q_sig) * 0.5).astype(int)
        q_sig = q_sig + Z_sig
        q_sig = np.clip(q_sig, 0, 255)
        qo = (q1 - Zi1) * (q_sig - Z_sig)
        qo = np.floor(qo * M0 / 2147483648 + 0.5)
        qo = (qo / pow(2, -n) + np.sign(qo) * 0.5).astype(int)
        qo = qo + Zo
        qo = np.clip(qo, 0, 255)

        return qo

    def infer(self, inp):
        """inp can be an image-path or a raw 3d-tensor or a 4d-tensor resized"""
        if isinstance(inp, str):
            inp = cv2.resize(cv2.imread(inp), (self.size, self.size))
            inp = np.expand_dims(inp, axis=0)
        elif len(inp.shape) != 4:
            inp = cv2.resize(inp, (self.size, self.size))
            inp = np.expand_dims(inp, axis=0)

        conv_silu_names = []
        for layer in self.model.layers:
            name = layer.name
            if "activation" in name:
                input_name = layer.input.name.split("/")[0]
                conv_silu_names.append(input_name)

        all_output = {}
        for i in range(len(self.model.layers)):
            layer = self.model.layers[i]
            name = layer.name
            print(name)

            if name == "input_1":
                output = inp
                all_output[name] = output
                print(np.min(output), np.max(output))

            elif "conv" in name and name not in conv_silu_names:
                output = self.conv_infer_quant(layer, name, all_output)
                print(np.min(output), np.max(output))

            elif "add" in name and "padding" not in name:
                output = self.add_infer_quant(layer, name, all_output)
                print(np.min(output), np.max(output))

            elif "concat" in name:
                output = self.concat_infer_quant(layer, name, all_output)
                print(np.min(output), np.max(output))

            elif name in conv_silu_names:
                output = self.conv_silu_infer_quant(layer, name, all_output)
                print(np.min(output[0]), np.max(output[0]), np.min(output[1]), np.max(output[1]))

            elif "activation" in name:
                output = self.silu_infer_quant(layer, name, all_output)
                print(np.min(output), np.max(output))

            elif "multiply" in name:
                sig_name = name.replace("multiply", "activation")
                output = all_output[sig_name]
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
    ap.add_argument('-f', '--file_model', help='file_model', default="saved/sub_model.h5")
    ap.add_argument('-j', '--json_path', help='json_path', default="saved/sub_model.json")
    args = ap.parse_args()

    model = keras.models.load_model(args.file_model)
    infer_ptq = InferPTQ(model, args.json_path)

    inp = cv2.imread(args.img_path)
    all_output = infer_ptq.infer(inp)
    save_all_outputs(all_output)