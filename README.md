# MathModel-2020C

## Preprocess P300 data
```
python prepare_data.py
```

## Run train_subject.py to train the model
* ```--subject``` [1, 5], int, which subject to train or test
* [optional]```--is_train``` [0, 1], int, default 1, 0 test 1 train
* [deprecated]```--is_with_average``` [True, False], bool, default False
#### train subject
```
python train_subject.py --subject 1 --is_train 1
```

#### test subject
```
python train_subject.py --subject 1 --is_train 0
```

#### if you have a GPU
```
CUDA_VISIBLE_DEVICES=0 python train_subject.py --subject 1
```
