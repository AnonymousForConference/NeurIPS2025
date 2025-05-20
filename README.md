# Hybrid-Query-Strategy-with-Diversity-Weighted-Metropolis-Adjusted-Langevin-Algorithm

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, run this command:

```train
python3 main.py --data Cifar10 --acqMode Max_Diversity --optimizer_name Adam --isInit True
python3 main.py --data Cifar10 --acqMode Max_Diversity --optimizer_name SAM --isInit False
```

## Evaluation

- Data will be downloaded to folder 'dataset'.
- Result will be recorded to folder 'Results'.

## Results

Our model achieves the following performance on image classification task:

|  Optimizer   | CIFAR-10  |      CIFAR-100     |    Tiny-ImageNet   |
| ------------ |-------------- | ------------- | ------------- | 
|    Adam      |     85.8%     |     76.8%     |     54.4%     | 

