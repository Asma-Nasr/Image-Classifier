# Project 2: Udacity AI Programming with Python Nanodegree

This project consists of two parts:

## Part 1: Image Classifier
- **Objective**: Classify images of flowers using PyTorch.
- **Environment**: Jupyter Notebook.

## Part 2: Command Line Program
- **Objective**: Classify images of flowers using PyTorch.
- **Environment**: Command line interface.

- ### How to Run the Command Line Program
To run the command line program, use the following command in your terminal:
To train the model
```bash
python train.py data_dir --save_dir save_dir --arch architecture(vgg16 or densenet121) --learning_rate lr --hidden_units 2048 --epochs epochs --gpu true/false
```
- or in my case

```bash
python train.py flowers --save_dir save --arch vgg16 --learning_rate 0.01 --hidden_units 2048 --epochs 8 --gpu true
```
- to predict an image 
```bash
python predict.py flowers/test/90/image_04405.jpg save --top_k 3  --category_names cat_to_name.json --gpu true
```
