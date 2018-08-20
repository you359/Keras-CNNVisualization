#!/usr/bin/env python
# coding=utf-8
###############################################
# File Name: DeconvNet2D.py
# Author: Liang Jiang
# mail: jiangliang0811@gmail.com
# Created Time: Sun 30 Oct 2016 09:52:15 PM CST
# Description: Code for Deconvnet based on keras
###############################################

import sys
sys.path.insert(0, '../utils')

import numpy as np
import sys
from PIL import Image
from keras.layers import (
    Input,
    InputLayer,
    Flatten,
    Activation,
    Dense)
from keras.layers.convolutional import (
    Convolution2D,
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D)
from keras.applications import vgg16, imagenet_utils
import keras.backend as K

import cv2
from utils import deprocess_image


class DConvolution2D(object):
    '''
    A class to define forward and backward operation on Convolution2D
    '''

    def __init__(self, layer):
        '''
        # Arguments
            layer: an instance of Convolution2D layer, whose configuration 
                   will be used to initiate DConvolution2D(input_shape, 
                   output_shape, weights)
        '''
        self.layer = layer

        weights = layer.get_weights()
        W = weights[0]
        b = weights[1]

        config = layer.get_config()
        print(config)

        # Set up_func for DConvolution2D
        nb_up_filter = W.shape[3]
        nb_up_row = W.shape[0]
        nb_up_col = W.shape[1]
        input = Input(shape=layer.input_shape[1:])
        # output = Convolution2D(
        #     nb_filter=nb_up_filter,
        #     nb_row=nb_up_row,
        #     nb_col=nb_up_col,
        #     border_mode='same',
        #     weights=[W, b]
        # )(input)
        output = Conv2D(
            filters=nb_up_filter,
            kernel_size=(nb_up_row, nb_up_col),
            padding='same',
            weights=[W, b]
        )(input)
        self.up_func = K.function([input, K.learning_phase()], [output])

        # Flip W horizontally and vertically,
        # and set down_func for DConvolution2D
        # W = np.transpose(W, (1, 0, 2, 3))
        # W = W[:, :, ::-1, ::-1]
        # print(W.shape)
        nb_down_filter = W.shape[2]
        nb_down_row = W.shape[0]
        nb_down_col = W.shape[1]
        b = np.zeros(nb_down_filter)
        input = Input(shape=layer.output_shape[1:])
        # output = Convolution2D(
        #     nb_filter=nb_down_filter,
        #     nb_row=nb_down_row,
        #     nb_col=nb_down_col,
        #     border_mode='same',
        #     weights=[W, b]
        # )(input)
        output = Conv2DTranspose(
            filters=nb_down_filter,
            kernel_size=(nb_down_row, nb_down_col),
            padding='same',
            weights=[W, b]
        )(input)
        self.down_func = K.function([input, K.learning_phase()], [output])

    def up(self, data, learning_phase=0):
        '''
        function to compute Convolution output in forward pass
        # Arguments
            data: Data to be operated in forward pass
            learning_phase: learning_phase of Keras, 1 or 0
        # Returns
            Convolved result
        '''
        self.up_data = self.up_func([data, learning_phase])[0]
        return self.up_data

    def down(self, data, learning_phase=0):
        '''
        function to compute Deconvolution output in backward pass
        # Arguments
            data: Data to be operated in backward pass
            learning_phase: learning_phase of Keras, 1 or 0
        # Returns
            Deconvolved result
        '''
        self.down_data = self.down_func([data, learning_phase])[0]
        return self.down_data


class DDense(object):
    '''
    A class to define forward and backward operation on Dense
    '''

    def __init__(self, layer):
        '''
        # Arguments
            layer: an instance of Dense layer, whose configuration 
                   will be used to initiate DDense(input_shape, 
                   output_shape, weights)
        '''
        self.layer = layer
        weights = layer.get_weights()
        W = weights[0]
        b = weights[1]

        # Set up_func for DDense
        input = Input(shape=layer.input_shape[1:])
        output = Dense(output_dim=layer.output_shape[1],
                       weights=[W, b])(input)
        self.up_func = K.function([input, K.learning_phase()], [output])

        # Transpose W and set down_func for DDense
        W = W.transpose()
        self.input_shape = layer.input_shape
        self.output_shape = layer.output_shape
        b = np.zeros(self.input_shape[1])
        flipped_weights = [W, b]
        input = Input(shape=self.output_shape[1:])
        output = Dense(
            output_dim=self.input_shape[1],
            weights=flipped_weights)(input)
        self.down_func = K.function([input, K.learning_phase()], [output])

    def up(self, data, learning_phase=0):
        '''
        function to compute dense output in forward pass
        # Arguments
            data: Data to be operated in forward pass
            learning_phase: learning_phase of Keras, 1 or 0
        # Returns
            Result of dense layer
        '''
        self.up_data = self.up_func([data, learning_phase])[0]
        return self.up_data

    def down(self, data, learning_phase=0):
        '''
        function to compute dense output in backward pass
        # Arguments
            data: Data to be operated in forward pass
            learning_phase: learning_phase of Keras, 1 or 0
        # Returns
            Result of reverse dense layer
        '''
        # data = data - self.bias
        self.down_data = self.down_func([data, learning_phase])[0]
        return self.down_data


class DPooling(object):
    '''
    A class to define forward and backward operation on Pooling
    '''

    def __init__(self, layer):
        '''
        # Arguments
            layer: an instance of Pooling layer, whose configuration 
                   will be used to initiate DPooling(input_shape, 
                   output_shape, weights)
        '''
        self.layer = layer
        self.poolsize = layer.pool_size
        # self.poolsize = layer.poolsize

    def up(self, data, learning_phase=0):
        '''
        function to compute pooling output in forward pass
        # Arguments
            data: Data to be operated in forward pass
            learning_phase: learning_phase of Keras, 1 or 0
        # Returns
            Pooled result
        '''
        [self.up_data, self.switch] = \
            self.__max_pooling_with_switch(data, self.poolsize)
        return self.up_data

    def down(self, data, learning_phase=0):
        '''
        function to compute unpooling output in backward pass
        # Arguments
            data: Data to be operated in forward pass
            learning_phase: learning_phase of Keras, 1 or 0
        # Returns
            Unpooled result
        '''
        self.down_data = self.__max_unpooling_with_switch(data, self.switch)
        return self.down_data

    def __max_pooling_with_switch(self, input, poolsize):
        '''
        Compute pooling output and switch in forward pass, switch stores 
        location of the maximum value in each poolsize * poolsize block
        # Arguments
            input: data to be pooled
            poolsize: size of pooling operation
        # Returns
            Pooled result and Switch
        '''
        switch = np.zeros(input.shape)
        out_shape = list(input.shape)

        row_poolsize = int(poolsize[0])
        col_poolsize = int(poolsize[1])
        out_shape[1] = out_shape[1] // poolsize[0]
        out_shape[2] = out_shape[2] // poolsize[1]
        pooled = np.zeros(out_shape)

        for sample in range(input.shape[0]):
            for dim in range(input.shape[3]):
                for row in range(out_shape[1]):
                    for col in range(out_shape[2]):
                        patch = input[sample,
                                row * row_poolsize: (row + 1) * row_poolsize,
                                col * col_poolsize: (col + 1) * col_poolsize,
                                dim]
                        max_value = patch.max()
                        pooled[sample, row, col, dim] = max_value
                        max_col_index = patch.argmax(axis=1)
                        max_cols = patch.max(axis=1)
                        max_row = max_cols.argmax()
                        max_col = max_col_index[max_row]
                        switch[sample,
                               row * row_poolsize + max_row,
                               col * col_poolsize + max_col,
                               dim] = 1
        return [pooled, switch]

    # Compute unpooled output using pooled data and switch
    def __max_unpooling_with_switch(self, input, switch):
        '''
        Compute unpooled output using pooled data and switch
        # Arguments
            input: data to be pooled
            poolsize: size of pooling operation
            switch: switch storing location of each elements
        # Returns
            Unpooled result
        '''
        tile = np.ones((switch.shape[1] // input.shape[1],
                        switch.shape[2] // input.shape[2]))

        input = np.transpose(input, (0, 3, 1, 2))
        out = np.kron(input, tile)
        out = np.transpose(out, (0, 2, 3, 1))

        unpooled = out * switch
        return unpooled


class DActivation(object):
    '''
    A class to define forward and backward operation on Activation
    '''

    def __init__(self, layer, linear=False):
        '''
        # Arguments
            layer: an instance of Activation layer, whose configuration 
                   will be used to initiate DActivation(input_shape, 
                   output_shape, weights)
        '''
        self.layer = layer
        self.linear = linear
        self.activation = layer.activation
        input = K.placeholder(shape=layer.output_shape)

        output = self.activation(input)
        # According to the original paper,
        # In forward pass and backward pass, do the same activation(relu)
        self.up_func = K.function(
            [input, K.learning_phase()], [output])
        self.down_func = K.function(
            [input, K.learning_phase()], [output])

    # Compute activation in forward pass
    def up(self, data, learning_phase=0):
        '''
        function to compute activation in forward pass
        # Arguments
            data: Data to be operated in forward pass
            learning_phase: learning_phase of Keras, 1 or 0
        # Returns
            Activation
        '''
        self.up_data = self.up_func([data, learning_phase])[0]
        return self.up_data

    # Compute activation in backward pass
    def down(self, data, learning_phase=0):
        '''
        function to compute activation in backward pass
        # Arguments
            data: Data to be operated in backward pass
            learning_phase: learning_phase of Keras, 1 or 0
        # Returns
            Activation
        '''
        self.down_data = self.down_func([data, learning_phase])[0]
        return self.down_data


class DFlatten(object):
    '''
    A class to define forward and backward operation on Flatten
    '''

    def __init__(self, layer):
        '''
        # Arguments
            layer: an instance of Flatten layer, whose configuration 
                   will be used to initiate DFlatten(input_shape, 
                   output_shape, weights)
        '''
        self.layer = layer
        self.shape = layer.input_shape[1:]
        self.up_func = K.function(
            [layer.input, K.learning_phase()], layer.output)

    # Flatten 2D input into 1D output
    def up(self, data, learning_phase=0):
        '''
        function to flatten input in forward pass
        # Arguments
            data: Data to be operated in forward pass
            learning_phase: learning_phase of Keras, 1 or 0
        # Returns
            Flattened data
        '''
        self.up_data = self.up_func([data, learning_phase])[0]
        return self.up_data

    # Reshape 1D input into 2D output
    def down(self, data, learning_phase=0):
        '''
        function to unflatten input in backward pass
        # Arguments
            data: Data to be operated in backward pass
            learning_phase: learning_phase of Keras, 1 or 0
        # Returns
            Recovered data
        '''
        new_shape = [data.shape[0]] + list(self.shape)
        assert np.prod(self.shape) == np.prod(data.shape[1:])
        self.down_data = np.reshape(data, new_shape)
        return self.down_data


class DInput(object):
    '''
    A class to define forward and backward operation on Input
    '''

    def __init__(self, layer):
        '''
        # Arguments
            layer: an instance of Input layer, whose configuration 
                   will be used to initiate DInput(input_shape, 
                   output_shape, weights)
        '''
        self.layer = layer

    # input and output of Inputl layer are the same
    def up(self, data, learning_phase=0):
        '''
        function to operate input in forward pass, the input and output
        are the same
        # Arguments
            data: Data to be operated in forward pass
            learning_phase: learning_phase of Keras, 1 or 0
        # Returns
            data
        '''
        self.up_data = data
        return self.up_data

    def down(self, data, learning_phase=0):
        '''
        function to operate input in backward pass, the input and output
        are the same
        # Arguments
            data: Data to be operated in backward pass
            learning_phase: learning_phase of Keras, 1 or 0
        # Returns
            data
        '''
        self.down_data = data
        return self.down_data


def visualize(model, data, layer_name, feature_to_visualize, visualize_mode):
    '''
    function to visualize feature
    # Arguments
        model: Pre-trained model used to visualize data
        data: image to visualize
        layer_name: Name of layer to visualize
        feature_to_visualize: Featuren to visualize
        visualize_mode: Visualize mode, 'all' or 'max', 'max' will only pick 
                        the greates activation in a feature map and set others
                        to 0s, this will indicate which part fire the neuron 
                        most; 'all' will use all values in a feature map,
                        which will show what image the filter sees. For 
                        convolutional layers, There is difference between 
                        'all' and 'max', for Dense layer, they are the same
    # Returns
        The image reflecting feature
    '''
    deconv_layers = []
    # Stack layers
    for i in range(len(model.layers)):
        if isinstance(model.layers[i], Convolution2D):
            deconv_layers.append(DConvolution2D(model.layers[i]))
            deconv_layers.append(
                DActivation(model.layers[i]))
        elif isinstance(model.layers[i], MaxPooling2D):
            deconv_layers.append(DPooling(model.layers[i]))
        elif isinstance(model.layers[i], Dense):
            deconv_layers.append(DDense(model.layers[i]))
            deconv_layers.append(
                DActivation(model.layers[i]))
        elif isinstance(model.layers[i], Activation):
            deconv_layers.append(DActivation(model.alyers[i]))
        elif isinstance(model.layers[i], Flatten):
            deconv_layers.append(DFlatten(model.layers[i]))
        elif isinstance(model.layers[i], InputLayer):
            deconv_layers.append(DInput(model.layers[i]))
        else:
            print('Cannot handle this type of layer')
            print(model.layers[i].get_config())
            sys.exit()
        if layer_name == model.layers[i].name:
            break

    # Forward pass
    deconv_layers[0].up(data)
    for i in range(1, len(deconv_layers)):
        deconv_layers[i].up(deconv_layers[i - 1].up_data)

    output = deconv_layers[-1].up_data
    assert output.ndim == 2 or output.ndim == 4
    if output.ndim == 2:
        feature_map = output[:, feature_to_visualize]
    else:
        feature_map = output[:, :, :, feature_to_visualize]
    if 'max' == visualize_mode:
        max_activation = feature_map.max()
        temp = feature_map == max_activation
        feature_map = feature_map * temp
    elif 'all' != visualize_mode:
        print('Illegal visualize mode')
        sys.exit()
    output = np.zeros_like(output)
    if 2 == output.ndim:
        output[:, feature_to_visualize] = feature_map
    else:
        output[:, :, :, feature_to_visualize] = feature_map

    # Backward pass
    deconv_layers[-1].down(output)
    for i in range(len(deconv_layers) - 2, -1, -1):
        deconv_layers[i].down(deconv_layers[i + 1].down_data)
    deconv = deconv_layers[0].down_data
    deconv = deconv.squeeze()

    return deconv


if "__main__" == __name__:
    image_path = "../images/cat.jpg"
    layer_name = "block5_conv3"
    feature_to_visualize = 256
    visualize_mode = "max"

    model = vgg16.VGG16(weights='imagenet', include_top=True)
    print(model.summary())
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    if layer_name not in layer_dict:
        print('Wrong layer name')
        sys.exit()

    # Load data and preprocess
    img = Image.open(image_path)
    img = img.resize((224, 224), Image.BILINEAR)
    img_array = np.array(img)
    img_array = img_array[np.newaxis, :]
    img_array = img_array.astype(np.float)
    img_array = imagenet_utils.preprocess_input(img_array)

    deconv = visualize(model, img_array,
                       layer_name, feature_to_visualize, visualize_mode)

    result = cv2.cvtColor(deprocess_image(deconv), cv2.COLOR_RGB2BGR)
    cv2.imshow('deconv_result', result)
    cv2.waitKey()
    cv2.destroyAllWindows()
