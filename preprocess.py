from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def process_image(image_path):
    ''' 
    This function Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Loading the image
    pil_image = Image.open(image_path)

    # Resizing the image to 256 256
    pil_image.thumbnail((256, 256))

    # Crop the center 224x224 
    width, height = pil_image.size
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2
    bottom = (height + 224) / 2
    pil_image = pil_image.crop((left, top, right, bottom))

    # Converting and normalizing the image to numpy array
    np_image = np.array(pil_image).astype(np.float32) / 255.0  # Convert to float and scale to [0, 1]

    # Normalize with mean and std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    np_image = (np_image - mean) / std

    # Reorder dimensions to (C, H, W)
    np_image = np_image.transpose((2, 0, 1))

    return np_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    #image = image.numpy().transpose((1, 2, 0))
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax