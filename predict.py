import argparse
import json
import torch
from torchvision import models
from preprocess import process_image, imshow
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

def load_model(save_dir):
    checkpoint = torch.load(save_dir+ '.pth', weights_only=False) 

    #checkpoint = torch.load(save_dir+ '/Project2_checkpoint.pth', weights_only=False) 
    model = models.vgg16(pretrained = True) 
    #model = models.densenet121(pretrained = True) 

    model.classifier = checkpoint ['classifier']
    model.load_state_dict (checkpoint ['state_dict'])
    model.class_to_idx = checkpoint ['mapping']
    
    for param in model.parameters(): 
        param.requires_grad = False 
    
    return model

  

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained model (VGG16).
    '''
        
    image = process_image(image_path) 
    
    im = torch.from_numpy(image).type(torch.FloatTensor)
    
    im = im.unsqueeze(dim = 0) 
        
    with torch.no_grad ():
        output = model.forward (im)
    output_prob = torch.exp (output) 
    
    probs, indeces = output_prob.topk (topk)
    probs = probs.numpy () 
    indeces = indeces.numpy () 
    
    probs = probs.tolist () [0] 
    indeces = indeces.tolist () [0]
    
    
    mapping = {value: key for key, value in
                model.class_to_idx.items()
                }
    
    classes = [mapping [item] for item in indeces]
    classes = np.array (classes)
    
    return probs, classes



if __name__ == "__main__":
    parser = argparse.ArgumentParser('predict.py to pass in a single image /path/to/image and return the flower name and class probability.',usage='python predict.py flowers/test/90/iimage_04405.jpg save --top_k 3  --category_names cat_to_name.json --gpu true')
    parser.add_argument('image_path',type=str,help='Path to image')
    parser.add_argument('checkpoint',type=str,help='Path to the saved model')
    parser.add_argument('--top_k',type=int,default=3,help='Top k most likely classes')
    parser.add_argument('--category_names',type=str,help='category names')
    parser.add_argument('--gpu',type=bool,help='To train the model on GPU/CPU')

    args = parser.parse_args()
    image_path= args.image_path
    checkpoint = args.checkpoint
    top_k = args.top_k

    category_names = args.category_names
    gpu = args.gpu
    print(f'image path is {image_path}')
    print(f'checkpoint name is {checkpoint}')
    print(f'top k value is {top_k}')
    print(f'gpu value is {gpu}')


    print('-------------------Loading the Saved Model------------------')
    saved_model = load_model(checkpoint) #save_dir
    #python predict.py flowers/test/90/iimage_04405.jpg save --top_k 3  --category_names cat_to_name.json --gpu true

    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    model = saved_model #the saved model


    probs, classes = predict(image_path, model, top_k)
    class_names = [cat_to_name[item] for item in classes]

    prob = max(probs)
    index = probs.index(prob)
    name = class_names[index]

    print(f'The class name of the image is {name}, with a probability of {prob}')
    for class_name, prob in zip(class_names, probs):
        print(f"{class_name}: {prob:.4f}")

    #plt.figure(figsize = (6,10))
    #plt.subplot(2,1,2)

    #sns.barplot(x=probs, y=class_names, color= 'blue')

    #plt.show()

    #
    
