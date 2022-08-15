# Despeckling SAR images

Repository contains an algorithm that removes speckle noise from SAR images.
The algorithm based on using convolutional neural network. Using PyTorch framework.
This repository contains the implementation of the network from [Kaggle](https://www.kaggle.com/code/javidtheimmortal/sar-image-despeckling-using-a-convolutional-neural/notebook) on the framework PyTorch.

# Description of CNN

Despeckle filter consist from convolution layers (CNN).
Example, Summary for image (256, 256, 3):

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 256, 256]             640
        Conv2dSame-2         [-1, 64, 256, 256]               0
         LeakyReLU-3         [-1, 64, 256, 256]               0
            Conv2d-4         [-1, 64, 256, 256]          36,928
        Conv2dSame-5         [-1, 64, 256, 256]               0
       BatchNorm2d-6         [-1, 64, 256, 256]             128
         LeakyReLU-7         [-1, 64, 256, 256]               0
            Conv2d-8         [-1, 64, 256, 256]          36,928
        Conv2dSame-9         [-1, 64, 256, 256]               0
      BatchNorm2d-10         [-1, 64, 256, 256]             128
        LeakyReLU-11         [-1, 64, 256, 256]               0
           Conv2d-12         [-1, 64, 256, 256]          36,928
       Conv2dSame-13         [-1, 64, 256, 256]               0
      BatchNorm2d-14         [-1, 64, 256, 256]             128
        LeakyReLU-15         [-1, 64, 256, 256]               0
           Conv2d-16         [-1, 64, 256, 256]          36,928
       Conv2dSame-17         [-1, 64, 256, 256]               0
      BatchNorm2d-18         [-1, 64, 256, 256]             128
        LeakyReLU-19         [-1, 64, 256, 256]               0
           Conv2d-20         [-1, 64, 256, 256]          36,928
       Conv2dSame-21         [-1, 64, 256, 256]               0
      BatchNorm2d-22         [-1, 64, 256, 256]             128
             ReLU-23         [-1, 64, 256, 256]               0
           Conv2d-24         [-1, 64, 256, 256]          36,928
       Conv2dSame-25         [-1, 64, 256, 256]               0
      BatchNorm2d-26         [-1, 64, 256, 256]             128
             ReLU-27         [-1, 64, 256, 256]               0
           Conv2d-28         [-1, 64, 256, 256]          36,928
       Conv2dSame-29         [-1, 64, 256, 256]               0
      BatchNorm2d-30         [-1, 64, 256, 256]             128
             ReLU-31         [-1, 64, 256, 256]               0
           Conv2d-32         [-1, 64, 256, 256]          36,928
       Conv2dSame-33         [-1, 64, 256, 256]               0
      BatchNorm2d-34         [-1, 64, 256, 256]             128
             ReLU-35         [-1, 64, 256, 256]               0
           Conv2d-36          [-1, 1, 256, 256]             577
       Conv2dSame-37          [-1, 1, 256, 256]               0
             ReLU-38          [-1, 1, 256, 256]               0
           Lambda-39          [-1, 1, 256, 256]               0
================================================================
Total params: 297,665
Trainable params: 297,665
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.25
Forward/backward pass size (MB): 1122.00
Params size (MB): 1.14
Estimated Total Size (MB): 1123.39
----------------------------------------------------------------
```

You can check architecture of neural network using the following lines of code:

```
from torchsummary import summary
model = DespeckleFilter(1)
summary(model, input_size=(1, 256, 256))
```

where 1 is number channels of image. Current repository using 1-channel images for training and testing neural network.

# Dataset

Dataset should contain image without noise (clean images) and with noise (noise images). The data set must have the following file structure:

```
dataset/
├── test
│   ├── clean
│   └── noise
├── train
│   ├── clean
│   └── noise
└── val
    ├── clean
    └── noise
```

There is [link](https://disk.yandex.ru/d/SH8-sVZkq6z23w) on dataset (dataset shortened and formed) for training and testing network. Full dataset you can find on [Kaggle](https://www.kaggle.com/code/javidtheimmortal/sar-image-despeckling-using-a-convolutional-neural/data).

# Train and Test

Start the training of neural network with command:

```
python3 train.py
```

Testing of neural network with command:

```
python3 test.py
```

# Results

Will be later

# Examples

Will be later
