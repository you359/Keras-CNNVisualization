from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing import image

import sys
sys.path.insert(0, '../utils')
sys.path.insert(0, '../keras-GuidedBackpropagation')

import numpy as np
import cv2

from utils import deprocess_image
from vis_convolution import VisConvolution
from gradcam import GradCAM


def load_image(path, target_size=(224, 224)):
    x = image.load_img(path, target_size=target_size)
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    return x


if __name__ == "__main__":
    img_width = 224
    img_height = 224

    model = VGG16(weights='imagenet')
    activation_layer = 'block5_conv3'

    img_path = '../images/cat_dog.jpg'
    img = load_image(path=img_path, target_size=(img_width, img_height))

    preds = model.predict(img)
    predicted_class = preds.argmax(axis=1)[0]
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    print("predicted top1 class:", predicted_class)
    print('Predicted:', decode_predictions(preds, top=1)[0])
    # Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]

    # create Grad-CAM generator
    gradcam_generator = GradCAM(model, activation_layer, predicted_class)
    grad_cam, grad_val = gradcam_generator.generate(img)

    # create Convolution Visualizer
    vis_conv = VisConvolution(model, VGG16, activation_layer)
    gradient = vis_conv.generate(img)

    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_width, img_height))

    grad_cam = grad_cam / grad_cam.max()
    grad_cam = grad_cam * 255
    grad_cam = cv2.resize(grad_cam, (img_width, img_height))
    grad_cam = np.uint8(grad_cam)

    cv_cam = cv2.applyColorMap(grad_cam, cv2.COLORMAP_JET)
    fin = cv2.addWeighted(cv_cam, 0.5, img, 0.5, 0)
    cv2.imshow('image', img)
    cv2.imshow('gradcam', fin)

    guided_gradcam = gradient * grad_cam[..., np.newaxis]
    guided_gradcam = deprocess_image(guided_gradcam)
    guided_gradcam = cv2.cvtColor(guided_gradcam, cv2.COLOR_RGB2BGR)
    cv2.imshow('guided_gradcam', guided_gradcam)
    cv2.waitKey()
    cv2.destroyAllWindows()
