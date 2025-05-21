# Hybrid-Query-Strategy-with-Diversity-Weighted-Metropolis-Adjusted-Langevin-Algorithm

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, run this command:

```train
python DWMALA10.py
python DWMALA100.py
python DWMALAI.py
```

## Results

Our model achieves the following performance on image classification task:

|  Optimizer     |    CIFAR-10  |   CIFAR-100   |  Tiny-ImageNet |
| ---------------|--------------|-------------- | ---------------| 
|    RMSProp     |     80.40%   |     46.65%    |     31.13%     | 

