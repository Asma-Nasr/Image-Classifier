import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from torchvision import models


def model(arch, units):
    '''
    Function for the model 
    Inputs : 
        arch : The model name, only two options are supported here (vgg16 or densenet121).
            type : str
        units : The number of hidden units for the model classifier layer.
            type : int
    Output : 
        The created custom model. 

    '''

    model = None  # Initialize the model variable
    # i'm using only two options vgg16 or densenet121
    if arch.lower() == 'vgg16':
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, units)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(p=0.2)),
            ('fc2', nn.Linear(units, 102)),
            ('output', nn.LogSoftmax(dim=1)) ]))
        model.classifier = classifier  # Assigning the classifier to the model

    elif arch.lower() == 'densenet121':
        model = models.densenet121(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(1024, units)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(p=0.2)),
            ('fc2', nn.Linear(units, 102)),
            ('output', nn.LogSoftmax(dim=1))  ]))
        model.classifier = classifier  # Assigning the classifier to the model

    else:
        raise ValueError("Unsupported architecture: choose 'vgg16' or 'densenet121'") #Raise error if the arc choosen isn't from the options declared

    return model

def validation(model, validloader, criterion, gpu):

    print('------------------started validating----------------------') 
    device = torch.device('cuda' if gpu else 'cpu')
    model.to(device)
    
    validloss = 0
    accuracy = 0

    for inputs, labels in validloader:
        inputs, labels = inputs.to(device), labels.to(device)  
        output = model(inputs)   
        validloss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return validloss, accuracy

def train_model(model, device, trainloader, validloader, epochs, lr):
    '''
    Function to train the model
    Inputs:
        model : the custom model from the function model
        device : the choosen device gpu/cpu
        trainloader : the training dataset
        validloader: the validation dataset
        epochs : number of epochs for the model
            type : int
        lr : learning rate for the model
            type : float
    
    '''
    print('-------------------started training-----------------------')
    model.to(device)
    training_loss = 0
    steps = 0
    print_every = 50
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr) # Adam optimizer
    
    for e in range(epochs):
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            
            inputs, labels = inputs.to(device), labels.to(device)  
            
            optimizer.zero_grad() 
            outputs = model(inputs)  
            loss = criterion(outputs, labels) 
            loss.backward() 
            optimizer.step()
            
            training_loss += loss.item() 
            
            if steps % print_every == 0:
                model.eval() 
                
                with torch.no_grad():
                    validloss, accuracy = validation(model, validloader, criterion, gpu=(device.type == 'cuda'))
                print("Epoch: {}/{}.. ".format(e + 1, epochs),
                      "Training Loss: {:.3f}.. ".format(training_loss / print_every),
                      "Validation Loss: {:.3f}.. ".format(validloss / len(validloader)),
                      "Validation Accuracy: {:.3f}%".format(accuracy / len(validloader) * 100))
                
                training_loss = 0
                model.train()


def save_checkpoint(save_dir,model,train_data):
    ''' Function  to save the trained model (checkpoint)
        inputs : 
            save_dir :save directory
            model : the trained model
            train_data : datasets.ImageFolder(train_dir, transform=self.train_transforms)
        output :
            saved model saved in save_dir/Project2_checkpoint.pth 
    '''
    model.to ('cpu') 
    model.class_to_idx = train_data.class_to_idx 

    checkpoint = {'classifier': model.classifier,
                'state_dict': model.state_dict (),
                'mapping':    model.class_to_idx  }        
    torch.save(checkpoint, save_dir + '.pth')

    #torch.save(checkpoint, save_dir + '/Project2_checkpoint.pth')
