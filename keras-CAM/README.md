# CAM(Class Activation Map)
re-implementation Class Activation Map using Keras

## usage
check jupyter notebook [CAM Visualization](https://github.com/you359/Keras-CNNVisualization/blob/master/keras-CAM/CAM%20Visualization.ipynb)

first, define your model and load image<br/>
your model must containing GAP(Global Average Pooling) or GMP(Global Max Pooling)
```python
img_path = '/path/to/img.jpg'

model = ResNet50(weights='imagenet')
img = load_image(path=img_path, target_size=(img_width, img_height))
```

second, create CAM generator and define activation_layer<br/>
activation_layer must be name of last conv layer(before GAP or GMP)
```python
activation_layer = 'activation49'
cam_generator = CAM(model, activation_layer)
```

finaliy, you can generate CAM using cam_generator.generate(...)<br/>
```python
cam = cam_generator.generate(img, predicted_class) 
```

## result
<img src=./result.png>

## Reference
[1] [Learning Deep Features for Discriminative Localization, 2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf) <br/>
<!-- [5] []() <br/> -->