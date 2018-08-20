# Grad-CAM(Grad Class Activation Map)
re-implementation Grad-CAM using Keras

## usage
check jupyter notebook [Grad-CAM Visualization](https://github.com/you359/Keras-CNNVisualization/blob/master/keras-GradCAM/Grad-CAM%20Visualization.ipynb)

first, define your model and load image<br/>
```python
img_path = '/path/to/img.jpg'

model = ResNet50(weights='imagenet')
img = load_image(path=img_path, target_size=(img_width, img_height))
```

second, create Grad-CAM generator and define activation_layer<br/>
```python
activation_layer = 'activation49'
gardcam_generator = GradCAM(model, activation_layer, predicted_class)
```

finaliy, you can generate Grad-CAM using gardcam_generator.generate(...)<br/>
```python
cam = gardcam_generator.generate(img) 
```

## result
<img src=./result.png>

## Reference
[1] [Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization, 2017](http://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf) <br/>
<!-- [5] []() <br/> -->