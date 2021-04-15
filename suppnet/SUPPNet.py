#!/usr/bin/python3
# -*- coding: utf-8 -*-

from itertools import accumulate

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, AveragePooling1D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.backend import clear_session


def UpSampling1D_layers(inputs, size=2):
    x = tf.keras.layers.Reshape((inputs.shape[1], 1, inputs.shape[2]))(inputs)
    x = tf.keras.layers.UpSampling2D(
        size=(size, 1), data_format=None, interpolation="bilinear")(x)
    return tf.keras.layers.Reshape((x.shape[1], x.shape[3]))(x)


def conv_block(x, no_channels, no_layers=2):

    for i in range(no_layers):
        x = Conv1D(no_channels, 3, activation='relu', padding='same')(x)
    return x


def pspnet_module_unet(input_features, compression):
    x = [input_features]
    for factor in compression:
        x.append(interp_block_unet(factor)(input_features))
    x = Concatenate()(x)
    return x


def interp_block_unet(pool_size):
    def layer(x):
        strides = pool_size
        x = AveragePooling1D(pool_size, strides, padding='same')(x)
        x = conv_block(x, no_channels=8, no_layers=3)
        x = UpSampling1D_layers(x, strides)
        return x
    return layer


def UNet_PSPNet(input_features, block_name):
    # Backbone
    sqeeze_rate = 13*[2]
    no_channels = [16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 128]
    no_layers = [2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    output_blocks = []
    _x = input_features
    for sq, no_channel, no_lay in zip(sqeeze_rate, no_channels, no_layers):
        _x = conv_block(_x, no_channel, no_layers=no_lay)
        output_blocks.append(_x)
        _x = AveragePooling1D(sq, sq, padding='same')(_x)

    # Bottom
    _bottom = conv_block(_x, 256, no_layers=3)

    # In between
    left_layer_forward = []
    for i, left_layer in enumerate(output_blocks):
        compression = list(accumulate(sqeeze_rate[i:], lambda x, y: x*y))
        left_layer_forward.append(pspnet_module_unet(
            left_layer, compression=compression))

    # Up
    _x = _bottom
    for i, (sq, no_channel, no_lay, left_layer) in enumerate(zip(reversed(sqeeze_rate), reversed(no_channels), reversed(no_layers), reversed(left_layer_forward))):
        _x = Conv1D(no_channel, 2, activation='relu', name=f"UP_{i}_"+block_name, padding='same')(UpSampling1D_layers(_x, sq))
        _x = Concatenate()([_x, left_layer])
        _x = conv_block(_x, no_channel, no_layers=no_lay)
    head = _x

    head_seg = Conv1D(128, 1, activation='relu', padding='same', name="seg_0_"+block_name)(head)
    head_seg = Conv1D(64, 1, activation='relu', padding='same', name="seg_1_"+block_name)(head_seg)
    head_seg = Conv1D(1, 1, activation='sigmoid', padding='same', name="seg_"+block_name)(head_seg)

    head_norm = Conv1D(128, 1, activation='relu', padding='same', name="norm_0_"+block_name)(head)
    head_norm = Conv1D(64, 1, activation='relu', padding='same', name="norm_1_"+block_name)(head_norm)
    head_norm = Conv1D(1, 1, activation='relu', padding='same', name="norm_"+block_name)(head_norm)

    narrow_head = Conv1D(8, 1, activation='relu', padding='same', name="forward_"+block_name)(head)
    forward_head = Concatenate()([head_norm, head_seg, narrow_head, input_features])

    return head_norm, head_seg, forward_head


def UNet_PSPNet_refine(input_features, block_name):
    # Backbone
    sqeeze_rate = 13*[2]
    no_channels = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]
    no_layers = [2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    output_blocks = []
    _x = input_features
    for sq, no_channel, no_lay in zip(sqeeze_rate, no_channels, no_layers):
        _x = conv_block(_x, no_channel, no_layers=no_lay)
        output_blocks.append(_x)
        _x = AveragePooling1D(sq, sq, padding='same')(_x)

    # Bottom
    _bottom = conv_block(_x, 256, no_layers=3)

    # Up
    _x = _bottom
    for i, (sq, no_channel, no_lay, left_layer) in enumerate(zip(reversed(sqeeze_rate), reversed(no_channels), reversed(no_layers), reversed(output_blocks))):
        _x = Conv1D(no_channel, 2, activation='relu', name=f"UP_{i}_"+block_name, padding='same')(UpSampling1D_layers(_x, sq))
        _x = Concatenate()([_x, left_layer])
        _x = conv_block(_x, no_channel, no_layers=no_lay)
    head = _x

    head_seg = Conv1D(32, 1, activation='relu', padding='same', name="seg_0_"+block_name)(head)
    head_seg = Conv1D(16, 1, activation='relu', padding='same', name="seg_1_"+block_name)(head_seg)
    head_seg = Conv1D(1, 1, activation='sigmoid', padding='same', name="seg_"+block_name)(head_seg)

    head_norm = Conv1D(32, 1, activation='relu', padding='same', name="norm_0_"+block_name)(head)
    head_norm = Conv1D(16, 1, activation='relu', padding='same', name="norm_1_"+block_name)(head_norm)
    head_norm = Conv1D(1, 1, activation='relu', padding='same', name="norm_"+block_name)(head_norm)

    return head_norm, head_seg


def create_SUPPNet_model(input_shape=(8192, 1)):
    input_features = Input(shape=input_shape)

    head_norm_0, head_seg_0, forward_head = UNet_PSPNet(input_features, '0')
    head_norm_1, head_seg_1 = UNet_PSPNet_refine(forward_head, '1')

    outputs = [head_norm_0, head_norm_1, head_seg_0, head_seg_1]
    model = Model(input_features, outputs, name='SUPPNet')

    nadam = Nadam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999)
    relative_weights = [1, 1]
    loss_weights = {**{f"norm_{i}": 100*val for i, val in enumerate(relative_weights)},
                    **{f"seg_{i}": val for i, val in enumerate(relative_weights)}}
    loss = {**{f"norm_{i}": 'mse' for i, _ in enumerate(relative_weights)},
            **{f"seg_{i}": 'binary_crossentropy' for i, _ in enumerate(relative_weights)}}
    metrics = {**{f"norm_{i}": 'mae' for i, _ in enumerate(relative_weights)},
               **{f"seg_{i}": 'accuracy' for i, _ in enumerate(relative_weights)}}

    model.compile(loss=loss,
                  metrics=metrics,
                  loss_weights=loss_weights,
                  optimizer=nadam)
    return model


class modelWrapper:
    def __init__(self, model, norm_only=True):
        self.model = model
        self.norm_only = norm_only

    def predict(self, X):
        results = self.model.predict(X)
        if self.norm_only:
            return results[1]
        else:
            return {"cont": results[1], "seg": results[3]}


def get_suppnet_model(norm_only=True):
    clear_session()
    SUPPNet_model = create_SUPPNet_model(input_shape=(8192, 1))

    # SUPPNet_model.load_weights('suppnet/supp_weights/SUPP_synth')
    SUPPNet_model.load_weights('suppnet/supp_weights/SUPP_active')
    return modelWrapper(SUPPNet_model, norm_only=norm_only)


if __name__ == "__main__":
    print("Num GPUs Available: ", len(
        tf.config.experimental.list_physical_devices('GPU')))
