# SURNet
code for the paper "Fish Detection and Segmentation using Convolutional Neural Networks with Limited Training Data"

### Quick start
1. Dataset can get from [Here!](https://drive.google.com/file/d/1GiDg6XTCgQfQD8gkJHKiTksRjI8abRKW/view?usp=sharing)
2. Install dependencies
```python =
pip install -r requirement.txt
```

### Train 
```python =
python train.py -p mosaic -l BCE -n 1000 -m vgg16

optional arguments:
  -b, --batch-size     input batch size for training (default: 5)
  -e, --epochs         number of epochs to train (default: 100)
  -n, --dataset_num    number of dataset to train (default: 1000)
  -p, --dataset_place  place of dataset to train (default: mosaic)
  -l, --loss           loss name(defaulf : BinaryCrossEntropy)
  -m, --model          model back bone name(defaulf : vgg16)
```

