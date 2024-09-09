import os
import json
import numpy as np
from keras.layers import ReLU
import cv2


class InferPTQBase:
    """
    Infer integer-only from json-file and model graph.
    """
    def __init__(self, model, json_path, size=320):
        self.model = model
        self.size = size
        with open(json_path) as f:
            self.model_quant = json.load(f)

    def conv_infer_quant(self, layer, name, all_output):
        data_quant = self.model_quant[name]

        input_name = layer.input.name.split("/")[0]
        print("--", input_name)
        inp = all_output[input_name].astype(float)

        kernel = np.array(data_quant["kernel"])
        # print("kernel:", np.min(kernel), np.max(kernel))
        bias = np.array(data_quant["bias"])
        Z_k = data_quant["flt_off"]
        Z_i = data_quant["inp_off"]
        Z_o = data_quant["out_off"]
        M0 = data_quant["multiplier"]
        n = data_quant["shift"]
        try:
            x = inp - Z_i
            if len(layer.get_weights()) == 2:
                layer.set_weights([kernel - Z_k, bias])
            else:
                layer.set_weights([kernel - Z_k])
            x = layer(x).numpy()
        except:
            print("--- check Z_i shape: ", Z_i.shape)
            print("--- check bias shape: ", bias.shape)
            print("--- check M0 shape: ", M0.shape)
            Z_i = Z_i[:, 1:, :].reshape((1,37,37,1))
            bias = bias[1:, :].reshape((37,37,bias.shape[-1]))
            M0 = M0[1:, :].reshape((37,37,1))
            n = n[1:, :].reshape((37, 37, 1))
            layer.set_weights([kernel - Z_k, np.zeros(shape=(bias.shape[-1],))])
            x = layer(inp - Z_i).numpy() + bias
        x = np.floor(x * M0 / 2147483648 + 0.5).astype(int)   ### -34,6 --> -35, -34,5 --> -34
        x = (x / pow(2, -n) + np.sign(x) * 0.5).astype(int)   ### -34,6 --> -35, -34,5 --> -35
        x = x + Z_o
        print("--- check min, max x:", np.min(x), np.max(x))
        x = np.clip(x, 0, 255)

        return x

    def add_infer_quant(self, layer, name, all_output):
        '''
        for more than 2 input, need modify after
        '''
        data_quant = self.model_quant[name]

        inputs = layer.input
        input_name1 = inputs[0].name.split("/")[0]
        input_name2 = inputs[1].name.split("/")[0]
        print("--", input_name1)
        print("--", input_name2)
        input1 = all_output[input_name1].astype(float)
        input2 = all_output[input_name2].astype(float)

        Z_i1 = data_quant["inp1_off"]
        Z_i2 = data_quant["inp2_off"]
        Z_o = data_quant["out_off"]
        M10 = data_quant["inp1_multiplier"]
        n1 = data_quant["inp1_shift"]
        M20 = data_quant["inp2_multiplier"]
        n2 = data_quant["inp2_shift"]
        Mo0 = data_quant["output_multiplier"]
        no = data_quant["output_shift"]
        nleft = data_quant["left_shift"]  # = 20
        inp1 = input1 - Z_i1
        inp1 = np.floor(inp1 * M10 / pow(2, 31-nleft) + 0.5)               ### -34,6 --> -35, -34,5 --> -34
        inp1 = (inp1 / pow(2, -n1) + np.sign(inp1) * 0.5).astype(int)      ### -34,6 --> -35, -34,5 --> -35
        inp2 = input2 - Z_i2
        inp2 = np.floor(inp2 * M20 / pow(2, 31-nleft) + 0.5)               ### -34,6 --> -35, -34,5 --> -34
        inp2 = (inp2 / pow(2, -n2) + np.sign(inp2) * 0.5).astype(int)      ### -34,6 --> -35, -34,5 --> -35
        x = inp1 + inp2
        x = x.astype(np.int64)
        x = np.floor(x * Mo0 / 2147483648 + 0.5).astype(int)               ### -34,6 --> -35, -34,5 --> -34
        x = (x / pow(2, -no) + np.sign(x) * 0.5).astype(int)               ### -34,6 --> -35, -34,5 --> -35
        x = x + Z_o
        print("--- check min, max x:", np.min(x), np.max(x))
        x = np.clip(x, 0, 255)

        return x

    def concat_infer_quant(self, layer, name, all_output):
        data_quant = self.model_quant[name]

        inputs = layer.input
        inputs_array = []
        for i in range(len(inputs)):
            input_name = inputs[i].name.split("/")[0]
            print("--", input_name)
            inp = all_output[input_name].astype(float)
            inputs_array.append(inp)

        inps_new = []
        for i in range(len(inputs_array)):
            x = inputs_array[i]
            Z_i = data_quant[f"inp{i+1}_off"]
            Z_o = data_quant["out_off"]
            M0 = data_quant[f"multiplier_{i+1}"]
            n = data_quant[f"shift_{i+1}"]
            x = x - Z_i
            x = np.floor(x * M0 / 2147483648 + 0.5).astype(int)            ### -34,6 --> -35, -34,5 --> -34
            x = (x / pow(2, -n) + np.sign(x) * 0.5).astype(int)            ### -34,6 --> -35, -34,5 --> -35
            x = x + Z_o
            inps_new.append(x)
        x = layer(inps_new).numpy()
        print("--- check min, max x:", np.min(x), np.max(x))
        x = np.clip(x, 0, 255).astype(int)

        return x

    def dense_infer_quant(self, layer, name, all_output):
        data_quant = self.model_quant[name]

        input_name = layer.input.name.split("/")[0]
        print("--", input_name)
        inp = all_output[input_name].astype(float)

        kernel = np.array(data_quant["kernel"])
        bias = np.array(data_quant["bias"])
        Z_k = data_quant["flt_off"]
        Z_i = data_quant["inp_off"]
        Z_o = data_quant["out_off"]
        M0 = data_quant["multiplier"]
        n = data_quant["shift"]
        # x = inp - Z_i
        # try:
        #     if len(layer.get_weights()) == 2:
        #         layer.set_weights([kernel - Z_k, bias])
        #     else:
        #         layer.set_weights([kernel - Z_k])
        #     x = layer(x).numpy()
        # except:
        x = (inp - Z_i) @ (kernel - Z_k) + bias
        x = np.floor(x * M0 / 2147483648 + 0.5).astype(int)  ### -34,6 --> -35, -34,5 --> -34
        x = (x / pow(2, -n) + np.sign(x) * 0.5).astype(int)  ### -34,6 --> -35, -34,5 --> -35
        x = x + Z_o
        print("--- check Zo:", Z_o)
        print("--- check min, max x:", np.min(x), np.max(x))
        print("--- check num value in x > 255:", np.sum(np.clip(np.sign(x-255), 0, 1)))
        x = np.clip(x, 0, 255)

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

            if "input" in name:
                output = inp
                all_output[name] = output
                print(np.min(output), np.max(output))

            elif any(check in name for check in ["conv", "cv", "pw", "dw"]):
                output = self.conv_infer_quant(layer, name, all_output)
                print(np.min(output), np.max(output))

            elif "add" in name and "padding" not in name:
                output = self.add_infer_quant(layer, name, all_output)
                print(np.min(output), np.max(output))

            elif "concat" in name:
                output = self.concat_infer_quant(layer, name, all_output)
                print(np.min(output), np.max(output))

            else:
                input_name = layer.input.name.split("/")[0]
                print("--", input_name)
                inp = all_output[input_name].astype(float)
                output = layer(inp).numpy()
                output = (output + np.sign(output) * 0.5).astype(int)    # Average layer can cause numbers to not int
                output = np.clip(output, 0, 255)
                print(np.min(output), np.max(output))

            all_output[name] = output
            print(output.shape)
            print("***********")

        return all_output
