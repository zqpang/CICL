# CICL

## Environment
Python >= 3.6

PyTorch >= 1.1

## Installation
```python
git clone https://github.com/zqpang/CICL
cd CICL
```
## Data preparation
Please modify the "data_dir" in the train.py file.

## Training
```python
python train.py
```
You can make modifications according to the specific device, with the command --cuda.

For convenience in training, we have integrated data augmentation into the code. Researchers can switch data augmentation to offline mode to speed up the training process.

## Model
The pre-trained [model](https://pan.baidu.com/s/1_AR-X7WRmgBznWrgCYU9kA)(code: bzkt) is available for access.

## Acknowledgements
Some parts of the code is borrowed from [MaskCL](https://github.com/MingkunLishigure/MaskCL). Thanks to [MaskCL](https://github.com/MingkunLishigure/MaskCL) for the opening source.
