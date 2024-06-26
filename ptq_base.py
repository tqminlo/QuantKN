import os
import json
import numpy as np
from sklearn.utils import shuffle
import cv2


def fused_bn_to_cv(model_base, model_fused, save_path=None):
    """
    Created fuse-model
    """

    model_base.summary()
    model_fused.summary()

    for layer in model_base.layers:
        print(layer.name)
        if "batch_normalization" in layer.name:
            print("bn name: ", layer.name)
            gamma = layer.weights[0].numpy()
            beta = layer.weights[1].numpy()
            moving_mean = layer.weights[2].numpy()
            moving_variance = layer.weights[3].numpy()
            epsilon = layer.epsilon
            momentum = layer.momentum
            num_channel = gamma.shape[0]
            print("num channel: ", num_channel)

            input_name = layer.input.name.split("/")[0]
            print("conv input name: ", input_name)
            conv_layer_base = model_base.get_layer(input_name)
            filters_base = conv_layer_base.weights[0].numpy()
            bias_base = conv_layer_base.weights[1].numpy() if len(conv_layer_base.weights) > 1 else np.zeros(
                shape=(num_channel,))

            conv_layer_fused = model_fused.get_layer(input_name)
            if "dw" in input_name:
                filters_fused = filters_base[:,:,:,0] * (gamma / (np.sqrt(moving_variance + epsilon)))
                filters_fused = np.expand_dims(filters_fused, axis=-1)
            else:
                filters_fused = filters_base * (gamma / (np.sqrt(moving_variance + epsilon)))
            # filters_fused = filters_base * (gamma / (np.sqrt(moving_variance + epsilon)))
            bias_fused = (bias_base - moving_mean) * gamma / (np.sqrt(moving_variance + epsilon)) + beta
            conv_layer_fused.set_weights([filters_fused, bias_fused])

        elif any(check in layer.name for check in ["conv", "cv", "dense", "pw", "dw"]):
            print("layer name: ", layer.name)
            layer_fused = model_fused.get_layer(layer.name)
            if len(layer.weights) == len(layer_fused.weights):
                layer_fused.set_weights(layer.weights)
                print("copy weights ok")

        print("-------------\n")

    if save_path:
        model_fused.save(save_path)

    return model_fused


class BasePTQ:
    """
    Post-training Quantization uint8 and integer-only-inference for model
    One input
    Layer wise
    Activate valid: ReLU
    Input is read with format BGR and norm by 1/255 in float model.
    """
    def __init__(self, model, data_quantize):
        self.model = model
        self.data_quantize = data_quantize
        self.output_dict = {}
        self.scale_dict = {}
        self.model.summary()
        self.get_all_tensor_quantize()

    def quant_uint8(self, tensor, activation=None):
        """Quantize a tensor to uint8"""
        if activation == "relu" and np.min(tensor) < 0:
            a = 0
            b = np.max(tensor)
            S = (b - a) / 255.
            tensor = np.floor(tensor / S + 0.5).astype(int)
            Z = 0
        else:
            a = np.min(tensor)
            b = np.max(tensor)
            S = (b - a) / 255.
            tensor = np.floor(tensor / S + 0.5).astype(int)
            Z = 0 - np.min(tensor)
            tensor = tensor + Z
        return Z, S, tensor

    def quant_bias(self, tensor, S):
        """Quantize bias to int32"""
        tensor = tensor / S + 0.5
        tensor = np.floor(tensor).astype(int)
        return tensor

    def get_multiplier(self, M):
        """Get multiplier and shift"""
        M0, n = np.frexp(M)
        M0 = round(M0 * 2147483648)
        return M0, n

    def get_all_tensor_quantize(self):
        for i in range(len(self.model.layers)):
            name = self.model.layers[i].name
            print(name)

            if "input" in name:
                output = self.data_quantize

            # elif "add" in name and "padding" not in name:
            #     inputs = self.model.layers[i].input
            #     input_name1 = inputs[0].name.split("/")[0]
            #     input_name2 = inputs[1].name.split("/")[0]
            #     input1 = self.output_dict[input_name1]
            #     input2 = self.output_dict[input_name2]
            #     output = input1 + input2

            else:
                try:
                    input_name = self.model.layers[i].input.name.split("/")[0]
                    input = self.output_dict[input_name]
                except Exception as e:
                    input_names = [inp.name.split("/")[0] for inp in self.model.layers[i].input]
                    input = [self.output_dict[inp_name] for inp_name in input_names]
                output = self.model.layers[i](input).numpy()

            self.output_dict[name] = output

    def quantize_model(self):
        all_layer_info = {}
        '''quantize backbone'''
        for layer in self.model.layers:
            layer_info = {}
            name = layer.name
            print(name)

            if "input" in name:
                output = self.data_quantize
                Z, S = 0, 1/255.
                # Z, S, _ = self.quant_uint8(output)   ### 0, 1/255.
                self.scale_dict[name] = [Z, S]

            elif any(check in name for check in ["conv", "cv", "pw", "dw"]):
                # quantize kernel
                w = layer.get_weights()
                kernel = w[0]
                input_name = layer.input.name.split("/")[0]
                input = self.output_dict[input_name]
                output = self.output_dict[name]
                strides = layer.get_config()['strides']
                padding = layer.get_config()['padding']
                Z_i, S_i = self.scale_dict[input_name]
                Z_k, S_k, kernel_quant = self.quant_uint8(kernel)
                Z_o, S_o, _ = self.quant_uint8(output)
                if len(w) == 1:
                    bias_quant = np.zeros(shape=w[0].shape[-1])
                else:
                    bias = w[1]
                    S_bias = S_k * S_i
                    # print("--S_bias:", S_bias, "--S_k:", S_k, "--S_i:", S_i, "--S_o:", S_o)
                    bias_quant = self.quant_bias(bias, S_bias)
                M = S_k * S_i / S_o
                M0, n = self.get_multiplier(M)
                # print(Z_k, S_k, Z_i, S_i, Z_o, S_o, S_bias, M0, n)

                layer_info["input_shape"] = input.shape[1:]
                layer_info["output_shape"] = output.shape[1:]
                layer_info["filter_shape"] = kernel.shape[:2]
                layer_info["stride"] = strides
                layer_info["padding"] = padding
                layer_info["flt_off"] = int(Z_k)
                layer_info["inp_off"] = int(Z_i)
                layer_info["out_off"] = int(Z_o)
                layer_info["multiplier"] = int(M0)
                layer_info["shift"] = int(n)
                layer_info["kernel"] = kernel_quant
                layer_info["bias"] = bias_quant
                self.scale_dict[name] = [Z_o, S_o]

            elif "add" in name and "padding" not in name:
                output = self.output_dict[name]
                Z_o, S_o, _ = self.quant_uint8(output)  # quantize for yolov8, need modify after
                inputs = layer.input
                if len(inputs) == 2:
                    input1_name = inputs[0].name.split("/")[0]
                    input2_name = inputs[1].name.split("/")[0]
                    Z_i1, S_i1 = self.scale_dict[input1_name]
                    Z_i2, S_i2 = self.scale_dict[input2_name]
                    if S_i2 < 2 * S_i1:
                        M1 = 1 / 2
                        M2 = S_i2 / (2 * S_i1)
                        Mo = 2 * S_i1 / S_o
                    else:
                        M1 = S_i1 / (2 * S_i2)
                        M2 = 1 / 2
                        Mo = 2 * S_i2 / S_o
                    M10, n1 = self.get_multiplier(M1)
                    M20, n2 = self.get_multiplier(M2)
                    Mo0, no = self.get_multiplier(Mo)

                    input1 = self.output_dict[input1_name]
                    input2 = self.output_dict[input2_name]
                    layer_info["input1_shape"] = input1.shape[1:]
                    layer_info["input2_shape"] = input2.shape[1:]
                    layer_info["output_shape"] = output.shape[1:]
                    layer_info["inp1_off"] = int(Z_i1)
                    layer_info["inp2_off"] = int(Z_i2)
                    layer_info["out_off"] = int(Z_o)
                    layer_info["inp1_multiplier"] = int(M10)
                    layer_info["inp1_shift"] = int(n1)
                    layer_info["inp2_multiplier"] = int(M20)
                    layer_info["inp2_shift"] = int(n2)
                    layer_info["output_multiplier"] = int(Mo0)
                    layer_info["output_shift"] = int(no - 20)
                    layer_info["left_shift"] = 20
                self.scale_dict[name] = [Z_o, S_o]

            elif "concatenate" in name:
                inputs = layer.input
                inp_names = [inp.name.split("/")[0] for inp in inputs]
                scales = [self.scale_dict[inp_name] for inp_name in inp_names]
                ranges = [[(0-scale[0])*scale[1], (255-scale[0])*scale[1]] for scale in scales]
                out_range = [min([r[0] for r in ranges]), max([r[1] for r in ranges])]
                S = (out_range[1] - out_range[0]) / 255.
                print("--", [scale[1] for scale in scales], S)
                Z = 0 - round(out_range[0] / S)
                self.scale_dict[name] = [Z, S]

                output = self.output_dict[name]
                for i in range(len(inp_names)):
                    M = scales[i][1] / S
                    M0, n = self.get_multiplier(M)
                    input = self.output_dict[inp_names[i]]
                    print("--", inp_names[i], input.shape)
                    layer_info[f"input{i+1}_shape"] = input.shape[1:]
                    layer_info[f"inp{i+1}_off"] = int(scales[i][0])
                    layer_info[f"multiplier_{i+1}"] = int(M0)
                    layer_info[f"shift_{i+1}"] = int(n)
                layer_info["out_off"] = int(Z)
                layer_info["output_height"] = output.shape[1] if len(output.shape) == 4 else 1
                layer_info["output_width"] = output.shape[2] if len(output.shape) == 4 else 1
                layer_info["output_depth"] = output.shape[-1]

            else:   ### Padding, Pooling, Reshape, Transpose, ...
                input_name = layer.input.name.split("/")[0]
                self.scale_dict[name] = self.scale_dict[input_name]

            all_layer_info[name] = layer_info
            print("-----------------")

        # Do for last layers
        if isinstance(self.model.output, list):
            last_layers = [out.name.split("/")[0] for out in self.model.output]
        else:
            last_layers = [self.model.output.name.split("/")[0]]
        all_layer_info["last_layers"] = {}
        for layer in last_layers:
            Z, S = self.scale_dict[layer]
            M, n = self.get_multiplier(S)
            all_layer_info["last_layers"][layer] = {}
            all_layer_info["last_layers"][layer]["out_multiplier"] = int(M)
            all_layer_info["last_layers"][layer]["shift"] = int(n)
            all_layer_info["last_layers"][layer]["out_off"] = int(Z)

        return all_layer_info

    def __call__(self, json_path):
        all_layer_info = self.quantize_model()
        with open(json_path, "w") as f:
            json.dump(all_layer_info, f)
