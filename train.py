import json
import time
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models, transforms, datasets
from command_utils import train_input, print_model
from model_utils import create_model, save_checkpoint
from data_loading import create_dataloaders

# EXAMPLE COMMAND
# python train.py --arch densenet --save_dir densenet.pth --hidden_units 1000,500 --learning_rate 0.0005 --epochs 10 --gpu true


def main():
    input_args = train_input()
    print_model(input_args)

    device = torch.device("cuda:0" if torch.cuda.is_available()
                          and input_args.gpu == True else "cpu")

    model = create_model(input_args.arch, input_args.hidden_units)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),
                           input_args.learning_rate,)

    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.1)

    image_datasets, dataloaders = create_dataloaders(
        input_args.data_dir)

    train(model, dataloaders, image_datasets, criterion, optimizer,
          exp_lr_scheduler, device, input_args.epochs)

    if input_args.save_dir:
        model.cpu()
        save_checkpoint({
            'epoch': input_args.epochs,
            'arch': input_args.arch,
            'classifier': model.classifier,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'mapping': image_datasets['train'].class_to_idx
        }, input_args.save_dir)


def train(model, dataloaders, datasets, criterion, optimizer, scheduler, device, epochs):
    start = time.time()
    print(f'Training with {device}\n')

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch+1, epochs))
        print('-' * 20)

        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()
                model.to(device)
            else:
                model.eval()

            running_error = 0.0
            running_corrects = 0

            for inputs, targets in dataloaders[phase]:
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model.forward(inputs)
                    _, predictions = torch.max(outputs, 1)
                    error = criterion(outputs, targets)

                    if phase == 'train':
                        error.backward()
                        optimizer.step()

                running_error += error.item() * inputs.size(0)
                running_corrects += torch.sum(predictions == targets.data)

            epoch_error = running_error / len(datasets[phase])
            epoch_acc = running_corrects.double() / len(datasets[phase])

            print('{} Error: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_error, epoch_acc))
            print('-' * 20)

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


if __name__ == "__main__":
    main()
