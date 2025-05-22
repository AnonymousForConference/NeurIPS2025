# Hybrid-Query-Strategy-with-Diversity-Weighted-Metropolis-Adjusted-Langevin-Algorithm

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) for CIFAR-10, CIFAR-100, and Tiny-Imagenet datasets, run these commands respectively:

```train
python3 DWMALA10.py
python3 DWMALA100.py
python3 DWMALAI.py
```

## Results

Our model achieves the following performance on the image classification task:

|  Dataset     |    CIFAR-10  |   CIFAR-100   |  Tiny-ImageNet |
| ---------------|--------------|-------------- | ---------------| 
|    Test Accuracy     |     80.40%   |     46.65%    |     31.13%     | 

