import json
import torch
import argparse
import numpy as np
from PIL import Image
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import OrderedDict
from torchvision import datasets, transforms, models

def get_label_map_dict(cat2name_path):
    with open(cat2name_path, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def load_model(save_dir):
    arch_names = {'VGG': 'vgg16',
              'densenet': 'densenet121'}
    
    checkpoint = torch.load(save_dir)
    model_name = arch_names[checkpoint['model_name']]
        
    model = eval('models.{}(pretrained=True)'.format(model_name))
    model.classifier = checkpoint['classifier']
    model.load_state_dict = checkpoint['state_dict']
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def predict(image_path, model, topk=5, gpu=False, category_names=None):   
    image = process_image(image_path)

    image = image.unsqueeze_(0)
    with torch.no_grad():
        if (torch.cuda.is_available()):
            device = 'cuda'
        else:
            device = 'cpu'
        model.to(device)
        output = model.forward(image.to(device))
        possibilities, predicted = torch.topk(output, topk)

    if (category_names is not None):
        labels = []
        cat_to_name = get_label_map_dict(category_names)
        for c in np.array(predicted[0]):
            labels.append(cat_to_name[str(c)])
        predicted=labels
        
    return predicted, F.softmax(possibilities, dim=1).data


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(image_path)
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                        [0.229, 0.224, 0.225])])
    image = transform(image)
    
    return image

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, default='flowers/test/10/image_07117.jpg', help='the image to be predicted')
    
    parser.add_argument('checkpoint', type=str, default='./checkpoint.pth', help = 'trained model saving path')
    
    parser.add_argument('--top_k', type=int, default=3,
                 help='the top k possible classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='the category map path')
    parser.add_argument('--gpu', default=False, help='Applying gpu or not')
    
    return parser.parse_args()


def main():
    in_arg = get_input_args()
    model = load_model(in_arg.checkpoint+'.pth')
    predicted, prob = predict(image_path=in_arg.input, model=model, topk=in_arg.top_k, gpu=in_arg.gpu, category_names=in_arg.category_names)
    print ('The most top {} possible class:'.format(in_arg.top_k))
    for i in range(len(predicted)):
        print ('Class: {}, Possibility: {}'.format(predicted[i], prob[0][i]))

    
if __name__=='__main__':
    main()
    
