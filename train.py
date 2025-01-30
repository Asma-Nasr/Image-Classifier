import argparse
import torch
import model1
from model1 import train_model, save_checkpoint
from torchvision import transforms, datasets, models
from dataloader import DataLoader
import warnings
warnings.filterwarnings("ignore")






if __name__ == '__main__':
    parser = argparse.ArgumentParser('train.py to train the model',usage='python train.py flowers --save_dir save --arch vgg16 --learning_rate 0.01 --hidden_units 2048 --epochs 8 --gpu true')
    parser.add_argument('data_dir',type=str,help='Dataset Directory')
    parser.add_argument('--save_dir',type=str,default='save',help='Path to save the trained model')
    parser.add_argument('--arch',default='vgg16',type=str,help='Choose a model architecture')
    parser.add_argument('--learning_rate',type=float,default=0.01,help='Learning rate of the model')
    parser.add_argument('--hidden_units',type=int,default=201,help='Hidden units of the NN')
    parser.add_argument('--epochs',type=int,default=8,help='Epochs to train the model')
    parser.add_argument('--gpu',type=bool,default='true',help='To train the model on GPU/CPU')

    # python train.py flowers --save_dir save --arch densenet121 --learning_rate 0.01 --hidden_units 2048 --epochs 1 --gpu true

    args = parser.parse_args()
    data_dir = args.data_dir
    save_dir = args.save_dir
    arch = args.arch
    lr = args.learning_rate
    units = args.hidden_units
    epochs = args.epochs
    gpu = args.gpu

    device = torch.device('cuda' if gpu else 'cpu')

    print(f'device is {device}')
    print(f'learning rate is {lr}, type of {type(lr)}')
  

    data_loader = DataLoader(data_dir=data_dir)
    trainloader, train_data  = data_loader.load_trainloader()
    validloader = data_loader.load_validloader()

    print('-----------------Data Loaded Correctly--------------------')
    model = model1.model(arch,units)
    print('-------------------Model initialzed-----------------------')
    train_model(model, device, trainloader, validloader, epochs, lr)
    print('-------------------Model Done Training -------------------')
    print('---------------------Saving the Model --------------------')
    save_checkpoint(save_dir,model,train_data)
    print('----------------------Model is saved----------------------')


