# For Keras model
import tensorflow as tf
from tensorflow import keras
from keras import Sequential, layers
from keras.layers import *
from keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img

# For Onnx model
import onnx
from onnx import numpy_helper, helper

# For Common
import numpy as np
import math
import copy

onnx_model = onnx.load("saved/yolov5face_n.onnx")
graph = onnx_model.graph


layer_store = {}
weight_store = {}
weight_shape = {}
for initializer in graph.initializer:
    weight_shape[initializer.name] = initializer.dims
    weight_store[initializer.name] = np.frombuffer(initializer.raw_data, dtype=np.float32).reshape(initializer.dims)


for node in graph.node:
    if "Constant" in node.name:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR:
                tensor = attr.t
                shape = tensor.dims
                value = numpy_helper.to_array(tensor)
                weight_store[node.output[0]] = value


class SplitLayer(layers.Layer):
    def __init__(self, num_splits, axis=-1, **kwargs):
        super(SplitLayer, self).__init__(**kwargs)
        self.num_splits = num_splits
        self.axis = axis

    def call(self, inputs):
        return tf.split(inputs, num_or_size_splits=self.num_splits, axis=self.axis)


def create_Conv(node):
    global weight_shape
    global layer_store
    padding = 'valid'

    filters = weight_shape[node.input[1]][0]
    kernel_size = tuple(helper.get_node_attr_value(node, "kernel_shape"))
    groups = helper.get_node_attr_value(node, "group")
    dilation_rate = helper.get_node_attr_value(node, "dilations")

    # Convert ONNX padding to kerras padding
    onnx_pads = helper.get_node_attr_value(node, "pads")
    top_pad = onnx_pads[0]
    bottom_pad = onnx_pads[2]
    left_pad = onnx_pads[1]
    right_pad = onnx_pads[3]

    strides = tuple(helper.get_node_attr_value(node, "strides"))
    layer_name = node.output[0]
    onnx_weight = weight_store[node.input[1]]  # Oc * Ic * H * W
    keras_weight = np.transpose(onnx_weight, (2, 3, 1, 0))
    onnx_bias = weight_store[node.input[2]]

    # Add Zeropadding layer
    zero_padding = ZeroPadding2D(padding=((top_pad, bottom_pad), (left_pad, right_pad)), data_format="channels_last")(
        layer_store[node.input[0]])

    conv2d_layer = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, groups=groups,
                          dilation_rate=dilation_rate)  # , activation=custom_max_activation)
    layer_store[layer_name] = conv2d_layer(zero_padding)
    conv2d_layer.set_weights([keras_weight, onnx_bias])
    # print(layer_name, helper.get_node_attr_value(node, "pads"), ((top_pad, bottom_pad), (left_pad, right_pad)))


def create_Mul(node):
    global layer_store

    layer_name = node.output[0]
    input_layer = []
    for inp in node.input:
        if "/model.22/Constant_12_output_0" == inp:
            const_array = weight_store[inp]
            dummy_input = Input(shape=(), dtype=const_array.dtype)
            const_layer = Lambda(lambda _: tf.constant(const_array))(dummy_input)
            input_layer.append(const_layer)
        elif "Constant" in inp:
            input_layer.append(tf.constant([1]))
        else:
            input_layer.append(layer_store[inp])
    layer_store[layer_name] = Multiply()(input_layer)


def create_Add(node):
    global layer_store

    layer_name = node.output[0]
    input_layer = []
    for inp in node.input:
        if "/model.22/Constant_10_output_0" == inp:
            const_array = weight_store[inp]
            dummy_input = Input(shape=(), dtype=const_array.dtype)
            const_layer = Lambda(lambda _: tf.constant(const_array))(dummy_input)
            input_layer.append(const_layer)
        elif "Constant" in inp:
            input_layer.append(tf.constant([1]))
        else:
            input_layer.append(layer_store[inp])
    layer_store[layer_name] = Add()(input_layer)


def create_Concat(node):
    global layer_store
    if "/model.22/Concat_4" != node.name:
        shape_onnx_2_keras = [0, 3, 1, 2]
    else:
        shape_onnx_2_keras = [0, 1, 2, 3]

    axis = shape_onnx_2_keras[helper.get_node_attr_value(node, "axis")]
    layer_name = node.output[0]
    input_layer = []
    for inp in node.input:
        if "/model.22/Sigmoid_output_0" == inp:
            layer_store[inp] = Reshape((1, 8400, 80))(layer_store[inp])
        else:
            input_layer.append(layer_store[inp])
    layer_store[layer_name] = Concatenate(axis=axis)(input_layer)


def create_Sigmoid(node):
    global layer_store

    layer_name = node.output[0]
    sigmoid_layer = Activation(keras.activations.sigmoid)
    layer_store[layer_name] = sigmoid_layer(layer_store[node.input[0]])


def create_MaxPool(node):
    global layer_store

    pool_size = tuple(helper.get_node_attr_value(node, "kernel_shape"))
    if sum(helper.get_node_attr_value(node, "pads")) == 0:
        padding = 'valid'
    else:
        padding = 'same'
    strides = tuple(helper.get_node_attr_value(node, "strides"))
    layer_name = node.output[0]

    layer_store[layer_name] = MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding)(
        layer_store[node.input[0]])


def create_Resize(node):
    global layer_store

    layer_name = node.output[0]
    layer_store[layer_name] = Resizing(height=int(layer_store[node.input[0]].shape[1] * 2),
                                       width=int(layer_store[node.input[0]].shape[2] * 2), interpolation='nearest')(
        layer_store[node.input[0]])


def create_Reshape(node):
    global layer_store
    global weight_store

    layer_name = node.output[0]
    shape_value = weight_store[str(node.input[1])]
    if len(shape_value) == 5:
        target_shape = [shape_value[3], shape_value[4], shape_value[1], shape_value[2]]
    elif len(shape_value) == 4:
        target_shape = [shape_value[2], shape_value[3], shape_value[1]]
    layer_store[layer_name] = Reshape(tuple(target_shape))(layer_store[node.input[0]])


def create_Transpose(node):
    global layer_store

    layer_name = node.output[0]
    perm = (helper.get_node_attr_value(node, "perm"))
    if len(perm) == 5:
        target_perm = [perm[2], perm[1], perm[4], perm[3]]
    elif len(perm) == 4:
        target_perm = perm
    layer_store[layer_name] = Permute(tuple(target_perm))(layer_store[node.input[0]])


def create_Softmax(node):
    global layer_store

    layer_name = node.output[0]
    layer_store[layer_name] = Softmax()(layer_store[node.input[0]])


def create_Gather(node):
    global layer_store
    indices = tf.constant([1])

    layer_name = node.output[0]
    # layer_store[layer_name] = tf.gather(layer_store[node.input[0]], indices, axis=0)
    layer_store[layer_name] = Lambda(lambda x: x[1], output_shape=lambda s: (1,))(layer_store[node.input[0]])


def create_Slice(node):
    global layer_store

    layer_name = node.output[0]
    data = layer_store[node.input[0]]
    if "Slice_1" in layer_name:
        starts = layer_store[node.input[1]]
    else:
        starts = tf.constant([0])
    ends = layer_store[node.input[2]]
    axes = tf.constant([1])
    # sizes = tf.math.subtract(ends, starts)
    sizes = Lambda(lambda x: tf.math.subtract(x[0], x[1]))([ends, starts])

    # layer_store[layer_name] = tf.keras.ops.slice(data, starts, sizes)
    layer_store[layer_name] = Lambda(
        lambda x: ops.slice(x[0], x[1], x[2]),
        output_shape=lambda input_shape: tuple(input_shape[2])  # Shape dựa trên sizes
    )([data, starts, sizes])


def create_Sub(node):
    global layer_store
    global weight_store

    layer_name = node.output[0]
    input_layer = []
    for inp in node.input:
        if "/model.22/Constant_9_output_0" == inp:
            const_array = weight_store[inp]
            dummy_input = Input(shape=(), dtype=const_array.dtype)
            const_layer = Lambda(lambda _: tf.constant(const_array))(dummy_input)
            input_layer.append(const_layer)
        else:
            input_layer.append(layer_store[inp])
    layer_store[layer_name] = Subtract()(input_layer)


def create_Div(node):
    global layer_store

    layer_name = node.output[0]
    div = tf.constant([2])
    layer_store[layer_name] = tf.keras.ops.divide(layer_store[node.input[0]], div)


def create_Split(node):
    global layer_store

    input_layer = layer_store[node.input[0]]
    if (len(input_layer.shape) == 4):
        shape_onnx_2_keras = [0, 3, 1, 2]
    elif (len(input_layer.shape) == 3):
        shape_onnx_2_keras = [0, 2, 1]

    axis = shape_onnx_2_keras[helper.get_node_attr_value(node, "axis")]

    # TODO: Not fixed (For and segment if node.name == ... or layer_name == ...)
    if (node.name == "/model.22/Split"):
        divisor = [64, 80]
    else:
        divisor = len(node.output)

    split_layer = Lambda(
        lambda x: tf.split(x, num_or_size_splits=divisor, axis=axis),
        name=f"split_{node.name.replace('/', '_')}"
    )(input_layer)

    if type(divisor) == type(list()):
        divisor = len(divisor)

    for i in range(divisor):
        layer_store[node.output[i]] = split_layer[i]


def create_Shape(node):
    global layer_store

    layer_name = node.output[0]
    # layer_store[layer_name] = tf.keras.ops.shape(layer_store[node.input[0]])(layer_store[node.input[0]])
    layer_store[layer_name] = Lambda(lambda x: tf.cast(ops.shape(x), tf.int64),
                                     output_shape=lambda input_shape: (len(input_shape),))(layer_store[node.input[0]])


def assert_split_block(onnx_graph, idx):
    return True


def create_split_block(node, i):
    global graph
    global layer_store
    input_layer = layer_store[node.input[0]]

    output1_name = graph.node[i + 10].output[0]
    output2_name = graph.node[i + 13].output[0]

    split_layer = SplitLayer(num_splits=2, axis=3)(input_layer)

    layer_store[output1_name] = split_layer[0]
    layer_store[output2_name] = split_layer[1]





layer_store["input"] = Input(shape= (640, 640, 3))
input = layer_store["input"]
layer_idx = 0

for i, node in enumerate(graph.node):
    if i < layer_idx:
        continue

    if "Shape" in node.name:
        is_split_block = assert_split_block(graph, i)
        if is_split_block:
            layer_idx = i + 14
            create_split_block(node, i)
            continue
        else:
            pass

    if "Constant" in node.name:
        continue
    # Skip the post process block
    elif "/model.20/cv3/act/Mul" in node.name:
        func_name = "create_" + node.op_type.split("/")[-1]
        # print("Call " + func_name)
        globals()[func_name](node)
        break
    else:
        func_name = "create_" + node.op_type.split("/")[-1]
        # print("Call " + func_name)
        globals()[func_name](node)

output = [layer_store["/model.20/cv3/act/Mul_output_0"], layer_store["/model.17/cv3/act/Mul_output_0"], layer_store["/model.14/cv3/act/Mul_output_0"]]
# output = layer_store["output0"]
keras_model = Model(input, output, name="yolov5face_n")
keras_model.save("saved/yolov5face_n.h5")

sub_model = Model(keras_model.input, keras_model.get_layer("concatenate_1").output)
sub_model.save("saved/sub_model.h5")


