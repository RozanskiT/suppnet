#!/usr/bin/python3
# -*- coding: utf-8 -*-

from itertools import accumulate
import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.backend import clear_session

from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Dropout, concatenate
from tensorflow.keras.layers import Conv1DTranspose, GlobalAveragePooling1D, Dense, Multiply, Add, Concatenate, add
from tensorflow.keras.layers import ReLU, LayerNormalization, Conv1DTranspose, AveragePooling1D
from tensorflow.keras.models import Model, load_model
import tensorflow.keras.backend as K


def residual_block(x, width, bottleneck_ratio=1, group_width=None, name=None):
    if group_width is None:
        group_width = width
    else:
        group_width = min(group_width, width)
    y = x

    name_1 = None if name is None else name+"_1"
    x = Conv1D(width, 1, padding='same', name=name_1)(x)
#     x = LayerNormalization()(x)
    x = ReLU()(x)

    layers_for_concat = []
    for i in range(width//bottleneck_ratio//group_width):
        name_2 = None if name is None else name+f"_2_gr{i}"
        _x = Conv1D(group_width, 3, padding='same', name=name_2)(x)
#         _x = LayerNormalization()(_x)
        _x = ReLU()(_x)
        layers_for_concat.append(_x)
    if len(layers_for_concat) > 1:
        x = Concatenate()(layers_for_concat)
    else:
        x = layers_for_concat[0]

    name_3 = None if name is None else name+f"_3"
    x = Conv1D(width, 1, padding='same', name=name_3)(x)
#     x = LayerNormalization()(x)
    x = ReLU()(x)

    x = Add()([x, y])
    return x


def residual_stride_block(x, width, bottleneck_ratio=1, group_width=None, stride=2):
    if group_width is None:
        group_width = width
    else:
        group_width = min(group_width, width)
    y = x

    x = Conv1D(width, 1, padding='same')(x)
#     x = LayerNormalization()(x)
    x = ReLU()(x)

    layers_for_concat = []
    for i in range(width//bottleneck_ratio//group_width):
        _x = Conv1D(group_width, 3, padding='same', strides=stride)(x)
#         _x = LayerNormalization()(_x)
        _x = ReLU()(_x)
        layers_for_concat.append(_x)
    if len(layers_for_concat) > 1:
        x = Concatenate()(layers_for_concat)
    else:
        x = layers_for_concat[0]

    x = Conv1D(width, 1, padding='same')(x)
#     x = LayerNormalization()(x)
    x = ReLU()(x)

    y = Conv1D(width, 1, padding='same', strides=stride)(y)
#     y = LayerNormalization()(y)
    y = ReLU()(y)

    x = Add()([x, y])
    return x


def residual_upsampling_block(x, width, bottleneck_ratio=1, group_width=None, stride=2):
    if group_width is None:
        group_width = width
    else:
        group_width = min(group_width, width)
    y = x

    x = Conv1D(width, 1, padding='same')(x)
#     x = LayerNormalization()(x)
    x = ReLU()(x)

    layers_for_concat = []
    for i in range(width//bottleneck_ratio//group_width):
        _x = Conv1DTranspose(group_width, 3, padding='same', strides=stride)(x)
#         _x = LayerNormalization()(_x)
        _x = ReLU()(_x)
        layers_for_concat.append(_x)
    if len(layers_for_concat) > 1:
        x = Concatenate()(layers_for_concat)
    else:
        x = layers_for_concat[0]

    x = Conv1D(width, 1, padding='same')(x)
#     x = LayerNormalization()(x)
    x = ReLU()(x)

    y = Conv1DTranspose(width, 1, padding='same', strides=stride)(y)
    y = ReLU()(y)

    x = Add()([x, y])
    return x


def head_segmentation(x, name):
    x = Conv1D(64, 1, activation='relu', padding='same')(x)
    x = Conv1D(32, 1, activation='relu', padding='same')(x)
    x = Conv1D(1, 1, activation='sigmoid', padding='same', name=name)(x)
    return x


def head_continuum(x, name):
    x = Conv1D(64, 1, activation='relu', padding='same')(x)
    x = Conv1D(32, 1, activation='relu', padding='same')(x)
    x = Conv1D(1, 1, activation='relu', padding='same', name=name)(x)
    return x


def stem(x):
    residual_block(x, width=16, bottleneck_ratio=1, group_width=16)
    return x


def PSPModule(in_features, compression, w, d):
    # list of pool's
    x = [in_features]
    for factor in compression:
        y = in_features
        if y.shape[-1] != w:
            y = Conv1D(w, 1, padding='same')(y)
        x.append(interp_block(factor, w, d)(y))
    if len(compression) > 1:
        x = Concatenate()(x)
    else:
        x = in_features
    return x


def interp_block(pool_size, w, d, b=1, g=None):
    def layer(x):
        strides = pool_size
        x = AveragePooling1D(pool_size, strides, padding='same')(x)
        x = PSP_block_net(x, width=w, depth=d,
                          bottleneck_ratio=b, group_width=g)
        x = UpSampling1D_layers(x, strides)
        return x
    return layer


def UpSampling1D_layers(inputs, size=2):
    x = tf.reshape(inputs, (-1, inputs.shape[1], 1, inputs.shape[2]))
    x = tf.keras.layers.UpSampling2D(
        size=(size, 1), data_format=None, interpolation="bilinear")(x)
    return tf.reshape(x, (-1, x.shape[1], x.shape[3]))


def PSP_block_net(x, width, depth, bottleneck_ratio=1, group_width=None):
    for i in range(depth):
        x = residual_block(x,
                           width,
                           bottleneck_ratio=bottleneck_ratio,
                           group_width=group_width,
                           name=None
                           )
    return x


def body_uppnet_suppnet(in_features, params):

    d_i = params["d_i"]
    w_i = params["w_i"]
    psp_bool = params["psp_bool"]
    g = params["g"]
    w_ppm = params["w_ppm"]
    d_ppm = params["d_ppm"]
    b = 1

    x = in_features

    # Backbone - Encoder
    forward_skip = []
    N = (len(d_i)-1)//2
    for i in range(N):
        d = d_i[i]
        w = w_i[i]
        w_next = w_i[i+1]

        for _ in range(d-1):
            x = residual_block(x, width=w, bottleneck_ratio=b, group_width=g)

        forward_skip.append(x)

        x = residual_stride_block(
            x, width=w_next, bottleneck_ratio=b, group_width=g)

    for _ in range(d_i[N]):
        x = residual_block(x, width=w_i[N], bottleneck_ratio=b, group_width=g)
    _x = x
    forward_skip.append(_x)

    # PSP Modules
    log2_input_length = 13
    for i in range(N+1):
        # In between PSPNets
        if psp_bool[i] > 0:
            compression = [2**(i+1) for i in range(log2_input_length-i)]
            forward_skip[i] = PSPModule(
                forward_skip[i], compression, w_ppm, d_ppm)

    x = forward_skip.pop()
    # Decoder
    for i in range(N):
        d = d_i[i+N+1]
        w = w_i[i+N+1]

        x = residual_upsampling_block(
            x, width=w, bottleneck_ratio=b, group_width=g)
        if psp_bool[i] > -1:
            x = Concatenate()([x, forward_skip[-1-i]])
            x = Conv1D(w, 1, padding='same')(x)
            x = ReLU()(x)

        for _ in range(d-1):
            x = residual_block(x, width=w, bottleneck_ratio=b, group_width=g)
    return x


def supp_model_template(params, input_shape=(8192, 1), no_forward_features=9):
    input_vec = Input(shape=input_shape)
    # -----------------
    x = input_vec
    # -----------------
    # Part 1 :
    x = body_uppnet_suppnet(x, params)
    decoded_cont_1 = head_continuum(x, name="cont_1")
    decoded_seg_1 = head_segmentation(x, name="seg_1")

    # -----------------
    # Part 2 :
    x = Conv1D(no_forward_features, 1, activation='relu', padding='same')(x)
    x = Concatenate()([x, decoded_cont_1, decoded_seg_1, input_vec])
    x = body_uppnet_suppnet(x, params)

    decoded_cont_2 = head_continuum(x, name="cont_2")
    decoded_seg_2 = head_segmentation(x, name="seg_2")

    model = Model(input_vec, [decoded_cont_1,
                  decoded_seg_1, decoded_cont_2, decoded_seg_2])

    nadam = Nadam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999)

    loss_weights = {f"cont_1": 400, f"seg_1": 1, f"cont_2": 400, f"seg_2": 1}
    loss = {f"cont_1": 'mse', f"seg_1": 'binary_crossentropy',
            f"cont_2": 'mse', f"seg_2": 'binary_crossentropy'}
    metrics = {f"cont_1": 'mae', f"seg_1": 'accuracy',
               f"cont_2": 'mae', f"seg_2": 'accuracy'}

    model.compile(loss=loss,
                  metrics=metrics,
                  loss_weights=loss_weights,
                  optimizer=nadam)
    return model


def create_SUPPNet_model(input_shape=(8192, 1)):
    params = {"d_i": np.array([1,  1,  1,  2,  2,  5,  6,  7, 10,  7,  6,  5,  2,  2,  1,  1,  1]),
              "w_i": np.array([12, 16, 16, 20, 24, 32, 44, 44, 44, 44, 44, 32, 24, 20, 16, 16, 12]),
              "psp_bool": np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
              "g": 64,
              "w_ppm": 4,
              "d_ppm": 1,
              }
    model = supp_model_template(params, input_shape=(8192, 1))
    return model


class modelWrapper:
    def __init__(self, model, norm_only=True):
        self.model = model
        self.norm_only = norm_only

    def predict(self, X):
        results = self.model.predict(X)
        if self.norm_only:
            return results[2]
        else:
            return {"cont": results[2], "seg": results[3]}


def get_suppnet_model(norm_only=True, which_weights="active"):
    script_directory = os.path.dirname(os.path.realpath(__file__))

    clear_session()
    print("Start creating SUPPNet model!")
    SUPPNet_model = create_SUPPNet_model(input_shape=(8192, 1))
    print("SUPPNet model created!")

    print("Start loading weights!")
    if which_weights == "synth":
        SUPPNet_synth_weights_relative_path = 'supp_weights/SUPPNet_synth'
        SUPPNet_model.load_weights(os.path.join(script_directory, SUPPNet_synth_weights_relative_path))
        print("SUPPNet (synth)")
    elif which_weights == "active":
        SUPPNet_active_weights_relative_path = 'supp_weights/SUPPNet_active'
        SUPPNet_model.load_weights(os.path.join(script_directory, SUPPNet_active_weights_relative_path))
        print("SUPPNet (active)")
    elif which_weights == "emission":
        SUPPNet_powr_weights_relative_path = 'supp_weights/SUPPNet_18_powr'
        SUPPNet_model.load_weights(os.path.join(script_directory, SUPPNet_powr_weights_relative_path))
        print("SUPPNet (emission, active+PoWR)")
    else:
        raise ValueError("Unknown model type")

    print("Weights loaded!")
    return modelWrapper(SUPPNet_model, norm_only=norm_only)


if __name__ == "__main__":
    print("Num GPUs Available: ", len(
        tf.config.experimental.list_physical_devices('GPU')))
