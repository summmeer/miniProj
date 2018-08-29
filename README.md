## miniProj
cifar10 classification based on alexnet and vgg16 using TensorFlow
### miniProj

#### Build a basic Alex net using tensorflow from scratch

- **model.py** Alexnet model, change the size and number of kernels and add dropout and batch normalization.
- **train.py** train Alexnet, and use Tensorboard to help visualize.

#### Use pre-trained deep model VGG to build image classification model

- **vgg16.py**  vgg model, add another fc layer with 10 output and add batch normalization. Reference: [Davi Frossard, 2016, VGG16 implementation in TensorFlow](http://www.cs.toronto.edu/~frossard/post/vgg16/ )
- **trainVGG.py** train vgg model, nearly the same as train.py
- **cifar10_input.py** load data, including data augmentation.

Best result:85%
