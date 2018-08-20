# all convolutional net
re-implementation guided-backpropagation for visualize conv features

## usage
check jupyter notebook [GuidedBackprop Visualization](https://github.com/you359/Keras-CNNVisualization/tree/master/keras-GuidedBackpropagation/GuidedBackprop%20Visualization.ipynb)

first, define your model and load image<br/>
```python
img_path = '/path/to/img.jpg'

model = ResNet50(weights='imagenet')
img = load_image(path=img_path, target_size=(img_width, img_height))
```

second, create VisConvolution and define activation_layer<br/>
```python
activation_layer = 'activation49'
visconv = VisConvolution(model, ResNet50, activation_layer, method='GuidedBackProp')
```
method must be one of the following (BackProp, DeconvNet, GuidedBackProp)

finaliy, you can generate gradient using visconv.generate(...)<br/>
```python
gradient = visconv.generate(img)
```

## result
<img src=./result.png>

## Reference
[1] [Striving for simplicity: The all convolutional net, 2014](https://arxiv.org/pdf/1412.6806.pdf) <br/>
[2] [https://github.com/jacobgil/keras-grad-cam](https://github.com/jacobgil/keras-grad-cam)
<!-- [5] []() <br/> -->