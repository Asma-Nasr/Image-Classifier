import torch
from torchvision import datasets, transforms

class DataLoader():
    '''
    Class for loading the data: train, valid, and test.
    '''
    def __init__(self, data_dir, batch_size=64):
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.train_transforms = transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        self.valid_transforms = transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        self.test_transforms = transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
   
    def load_trainloader(self):
        train_dir = f'{self.data_dir}/train'
        train_data = datasets.ImageFolder(train_dir, transform=self.train_transforms)
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        print('-----------------Train loader success-----------------')
        return trainloader, train_data

    def load_validloader(self):
        valid_dir = f'{self.data_dir}/valid'
        valid_data = datasets.ImageFolder(valid_dir, transform=self.valid_transforms)
        validloader = torch.utils.data.DataLoader(valid_data, batch_size=self.batch_size)
        print('------------------Valid loader success----------------')

        return validloader

    def load_testloader(self):
        test_dir = f'{self.data_dir}/test'
        test_data = datasets.ImageFolder(test_dir, transform=self.test_transforms)
        testloader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size)
        print('-----------------Test loader success------------------')

        return testloader
