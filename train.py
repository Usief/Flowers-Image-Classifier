import torch
import argparse
import numpy as np
from PIL import Image
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import OrderedDict
from torchvision import datasets, transforms, models


def process_data(data_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    
    train_datasets = datasets.ImageFolder(data_dir, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    
    return train_loader, train_datasets


def setup_model(model_name='densenet121', hidden_layers=120, output=102, learning_rate=0.001, dropout=0.5):
    name2inputs = {'densenet121': 1024,
              'vgg16': 25088}
    
    try:
        hidden_layers = [hidden_layers] 
        hidden_layers.insert(0, name2inputs[model_name])
    except:
        print ('Invalid model name with {}'.format(model_name))
        return
               
    model = eval('models.{}(pretrained=True)'.format(model_name))
    
    for param in model.parameters():
        param.requires_grad = False
    
    layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
    structures = [('dropout', nn.Dropout(dropout))]
    for i, size in enumerate(layer_sizes):
        layer_name = 'fc{}'.format(i+1)
        structures.extend([(layer_name, nn.Linear(size[0], size[1])),
                           ('relu', nn.ReLU())])
    structures.extend([('fc_last', nn.Linear(hidden_layers[-1], output)),('output', nn.LogSoftmax(dim=1))])
    classifier = nn.Sequential(OrderedDict(structures))
               
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    model.cuda()
    
    return model, criterion, optimizer


def train(train_loader, model, criterion, optimizer, epochs=10, gpu=False):
    print_every = 10
    steps = 0

    # change to cuda
    if (gpu):
        print ('Training start with GPU')
        model.to('cuda')
    else:
        print ('Training start with CPU')

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(train_loader):
            steps += 1
            if (gpu):
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every))
                running_loss = 0
                
    print ('Train Completed')
    return model
   
def save_model(model, train_datasets):
    model.class_to_idx = train_datasets.class_to_idx
    
    try:
        torch.save({'model_name': model.__class__.__name__,
            'classifier': model.classifier,
            'state_dict': model.state_dict(),
            'class_to_idx': model.class_to_idx}
           , 'checkpoint.pth')
        print ('Save model at chckpoint.pth Successfully')
    except e:
        print ('Save model with error')

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, default='flowers/train', help='the train data path')
    
    parser.add_argument('--save_dir', type=str, default='checkpoint .pkt', help = 'trained model saving path')
    
    parser.add_argument('--arch', type=str, default='vgg16',
                 help='chosen model vgg or densenet121')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='set the learning rate')
    parser.add_argument('--hidden_units', type=int, default=120, help='hidden units a number')
    parser.add_argument('--dropout', type=float, default=0.5, help='the dropout rate')
    parser.add_argument('--epochs', type=int, default='10', help='the training epochs')
    parser.add_argument('--gpu', default=False, help='Applying gpu or not')
    
    return parser.parse_args()
    
def main():
    in_arg = get_input_args()
    
    train_loader, train_datasets = process_data(in_arg.data_dir)
    model, criterion, optimizer = setup_model(model_name=in_arg.arch, hidden_layers=in_arg.hidden_units, learning_rate=in_arg.learning_rate, dropout = in_arg.dropout)
    
    model = train(train_loader, model, criterion, optimizer, in_arg.epochs, in_arg.gpu)
    save_model(model, train_datasets)
    
if __name__ == '__main__':
    print ('running')
    main()
    