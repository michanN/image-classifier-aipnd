import torch
import torch.nn.functional as F
from collections import OrderedDict
from torchvision import models
from torch import nn


def create_model(model, hidden_units):
    model_name = model.lower()

    if(model_name == 'densenet'):
        model = models.__dict__['densenet201'](pretrained=True)
        num_features = model.classifier.in_features
    elif(model_name == 'vgg'):
        model = models.__dict__['vgg16'](pretrained=True)
        num_features = model.classifier[0].in_features

    for param in model.parameters():
        param.requires_grad = False

    hidden_units = hidden_units.split(',')
    hidden_units = [int(i) for i in hidden_units]

    hidden = [nn.Linear(num_features, hidden_units[0])]

    layer_sizes = zip(hidden_units[:-1], hidden_units[1:])
    hidden.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

    hidden_layers = []

    for i in range(len(hidden)):
        hidden_layers.append(hidden[i])
        hidden_layers.append(nn.Dropout(p=0.5))
        hidden_layers.append(nn.ReLU())

    hidden_layers.append(nn.Linear(hidden_units[-1], 102))
    hidden_layers.append(nn.LogSoftmax(dim=1))

    hidden_layers = nn.ModuleList(hidden_layers)

    classifier = nn.Sequential(*hidden_layers)

    model.classifier = classifier

    return model


def save_checkpoint(state, filename="checkpoint.pth"):
    torch.save(state, filename)
