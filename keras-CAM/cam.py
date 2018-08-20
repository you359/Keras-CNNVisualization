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


class CAM:
    def __init__(self, model, activation_layer):
        self.model = model
        self.activation_layer = activation_layer
        self.tensor_function = self._get_cam_tensor_function()

    # get partial tensor graph of CNN model
    def _get_cam_tensor_function(self):
        model_input = self.model.input

        f_k = self.model.get_layer(self.activation_layer).output

        tensor_function = K.function([model_input], [f_k])
        return tensor_function

    # generate class activation map
    def generate(self, input_tensor, class_idx):
        [last_conv_output] = self.tensor_function([input_tensor])
        last_conv_output = last_conv_output[0]

        class_weight_k = self.model.layers[-1].get_weights()[0][:, class_idx]

        cam = np.zeros(dtype=np.float32, shape=last_conv_output.shape[0:2])
        for k, w in enumerate(class_weight_k):
            cam += w * last_conv_output[:, :, k]

        return cam


if __name__ == "__main__":
    img_width = 224
    img_height = 224

    # you must use CNN models which has GAP(Global Average Pooling) or GMP(Global Max Pooling)
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

    cam_generator = CAM(model, activation_layer)
    cam = cam_generator.generate(img, predicted_class)

    cam = cam / cam.max()
    cam = cam * 255

    cam = cv2.resize(cam, (img_width, img_height))
    cam = np.uint8(cam)

    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_width, img_height))
    cv_cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    fin = cv2.addWeighted(cv_cam, 0.7, img, 0.3, 0)
    cv2.imshow('cam', cv_cam)
    cv2.imshow('image', img)
    cv2.imshow('cam on image', fin)
    cv2.waitKey()
    cv2.destroyAllWindows()
