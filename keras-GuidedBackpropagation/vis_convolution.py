import sys
sys.path.insert(0, '../utils')

import keras
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image
import keras.backend as K

import tensorflow as tf
from tensorflow.python.framework import ops

import numpy as np
import cv2

from utils import deprocess_image


def load_image(path, target_size=(224, 224)):
    x = image.load_img(path, target_size=target_size)
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    return x


class VisConvolution:
    def __init__(self, model, model_func, activation_layer, method='GuidedBackProp'):
        self.model = model
        self.model_func = model_func
        self.activation_layer = activation_layer

        if method == 'BackProp':
            self._register_backprop_gradient()
            self.guided_model = self._modify_graph('BackProp')
        elif method == 'DeconvNet':
            self._register_deconvnet_gradient()
            self.guided_model = self._modify_graph('DeconvNet')
        elif method == 'GuidedBackProp':
            self._register_guidedbackprop_gradient()
            self.guided_model = self._modify_graph('GuidedBackProp')
        else:
            sys.exit('method must be (BackProp, DeconvNet, GuidedBackProp)')

        self.tensor_function = self.get_tensor_function()

    # register gradient
    def _register_backprop_gradient(self):
        if "BackProp" not in ops._gradient_registry._registry:
            @ops.RegisterGradient("BackProp")
            def _BackProp(op, grad):
                dtype = op.inputs[0].dtype
                return grad * tf.cast(op.inputs[0] > 0., dtype)

    def _register_deconvnet_gradient(self):
        if "DeconvNet" not in ops._gradient_registry._registry:
            @ops.RegisterGradient("DeconvNet")
            def _DeconvNet(op, grad):
                dtype = op.inputs[0].dtype
                return grad * tf.cast(grad > 0., dtype)

    def _register_guidedbackprop_gradient(self):
        if "GuidedBackProp" not in ops._gradient_registry._registry:
            @ops.RegisterGradient("GuidedBackProp")
            def _GuidedBackProp(op, grad):
                dtype = op.inputs[0].dtype
                return grad * tf.cast(grad > 0., dtype) * \
                       tf.cast(op.inputs[0] > 0., dtype)

    # modify model graph
    def _modify_graph(self, name):
        g = tf.get_default_graph()
        with g.gradient_override_map({'Relu': name}):

            # get layers that have an activation
            layer_dict = [layer for layer in self.model.layers[1:]
                          if hasattr(layer, 'activation')]

            # replace relu activation
            for layer in layer_dict:
                if layer.activation == keras.activations.relu:
                    layer.activation = tf.nn.relu

            # re-instanciate a new model
            K.reset_uids()
            new_model = self.model_func(weights='imagenet')
        return new_model

    # get partial tensor graph of CNN model
    def get_tensor_function(self, method='max', channel=0):
        model_input = self.guided_model.input
        layer_output = self.guided_model.get_layer(self.activation_layer).output

        if method == 'max':
            output = K.max(layer_output, axis=3)
        elif method == 'one':
            output = layer_output[:, :, :, channel]
        else:
            sys.exit('method must be (max, one)')

        tensor_function = K.function([model_input], [K.gradients(output, model_input)[0]])
        return tensor_function

    # generate saliency map(gradient)
    def generate(self, input_tensor):
        saliency = self.tensor_function([input_tensor])

        return saliency[0]


if __name__ == "__main__":
    img_width = 224
    img_height = 224

    model = ResNet50(weights='imagenet')
    activation_layer = 'activation_49'

    img_path = '../images/cat.jpg'
    img = load_image(path=img_path, target_size=(img_width, img_height))

    preds = model.predict(img)
    predicted_class = preds.argmax(axis=1)[0]
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    print("predicted top1 class:", predicted_class)
    print('Predicted:', decode_predictions(preds, top=1)[0])
    # Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]

    vis_conv = VisConvolution(model, ResNet50, activation_layer)
    gradient = vis_conv.generate(img)

    cv2.imshow('vis_conv', deprocess_image(gradient))
    cv2.waitKey()
    cv2.destroyAllWindows()

