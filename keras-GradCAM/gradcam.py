from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions

from keras.preprocessing import image
import keras.backend as K

import numpy as np
import cv2


def load_image(path, target_size=(224, 224)):
    x = image.load_img(path, target_size=target_size)
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    return x


class GradCAM:
    def __init__(self, model, activation_layer, class_idx):
        self.model = model
        self.activation_layer = activation_layer
        self.class_idx = class_idx
        self.tensor_function = self._get_gradcam_tensor_function()

    # get partial tensor graph of CNN model
    def _get_gradcam_tensor_function(self):
        model_input = self.model.input
        y_c = self.model.outputs[0].op.inputs[0][0, self.class_idx]
        A_k = self.model.get_layer(self.activation_layer).output

        tensor_function = K.function([model_input], [A_k, K.gradients(y_c, A_k)[0]])
        return tensor_function

    # generate Grad-CAM
    def generate(self, input_tensor):
        [conv_output, grad_val] = self.tensor_function([input_tensor])
        conv_output = conv_output[0]
        grad_val = grad_val[0]

        weights = np.mean(grad_val, axis=(0, 1))

        grad_cam = np.zeros(dtype=np.float32, shape=conv_output.shape[0:2])
        for k, w in enumerate(weights):
            grad_cam += w * conv_output[:, :, k]

        grad_cam = np.maximum(grad_cam, 0)

        return grad_cam, weights


class CAM:
    def __init__(self, model, activation_layer, class_idx):
        self.model = model
        self.activation_layer = activation_layer
        self.class_idx = class_idx
        self.tensor_function = self._get_cam_tensor_function()

    # get partial tensor graph of CNN model
    def _get_cam_tensor_function(self):
        model_input = self.model.input

        A_k = self.model.get_layer(self.activation_layer).output

        tensor_function = K.function([model_input], [A_k])
        return tensor_function

    # generate CAM
    def generate(self, input_tensor):
        [conv_output] = self.tensor_function([input_tensor])
        conv_output = conv_output[0]

        weights = self.model.layers[-1].get_weights()[0][:, self.class_idx]

        cam = np.zeros(dtype=np.float32, shape=conv_output.shape[0:2])
        for k, w in enumerate(weights):
            cam += w * conv_output[:, :, k]

        return cam, weights


if __name__ == "__main__":
    img_width = 224
    img_height = 224

    model = ResNet50(weights='imagenet')
    activation_layer = 'activation_49'

    img_path = '../images/elephant.jpg'
    img = load_image(path=img_path, target_size=(img_width, img_height))

    preds = model.predict(img)
    predicted_class = preds.argmax(axis=1)[0]
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    print("predicted top1 class:", predicted_class)
    print('Predicted:', decode_predictions(preds, top=1)[0])
    # Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]

    cam_generator = CAM(model, activation_layer, predicted_class)
    gardcam_generator = GradCAM(model, activation_layer, predicted_class)

    gradcam, gradcam_weight = gardcam_generator.generate(img)
    cam, cam_weight = cam_generator.generate(img)

    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_width, img_height))
    cv2.imshow('image', img)

    cam = cam / cam.max()
    cam = cam * 255
    cam = cv2.resize(cam, (img_width, img_height))
    cam = np.uint8(cam)

    cv_cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    fin = cv2.addWeighted(cv_cam, 0.5, img, 0.5, 0)
    cv2.imshow('cam', fin)

    gradcam = gradcam / gradcam.max()
    gradcam = gradcam * 255
    gradcam = cv2.resize(gradcam, (img_width, img_height))
    gradcam = np.uint8(gradcam)

    cv_cam = cv2.applyColorMap(gradcam, cv2.COLORMAP_JET)
    fin = cv2.addWeighted(cv_cam, 0.5, img, 0.5, 0)
    cv2.imshow('gradcam', fin)

    cv2.waitKey()
    cv2.destroyAllWindows()

    # validate
    print(model.get_layer(activation_layer).output_shape[1:3])
    Z = model.get_layer(activation_layer).output_shape[1] * model.get_layer(activation_layer).output_shape[2]

    print(gradcam_weight * Z)
    print(cam_weight)

