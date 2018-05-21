import os
import torch
import json
from torchvision import models
from torch.autograd import Variable
from data_loading import process_image
from command_utils import predict_input, print_predict


# EXAMPLE COMMAND
# python predict.py --checkpoint densenet.pth --gpu true
# --input flowers/test/2/image_05100.jpg --category_names cat_to_name.json --top_k 5

def main():
    input_args = predict_input()

    if input_args.checkpoint:
        if os.path.isfile(input_args.checkpoint):
            print("==> loading checkpoint '{}'".format(input_args.checkpoint))
            checkpoint = torch.load(input_args.checkpoint)

            if checkpoint['arch'] == 'densenet':
                model = models.__dict__['densenet201'](pretrained=True)
            elif checkpoint['arch'] == 'vgg':
                model = models.__dict__['vgg16'](pretrained=True)

            model.classifier = checkpoint['classifier']
            model.load_state_dict(checkpoint['state_dict'])
            model.class_to_idx = checkpoint['mapping']
        else:
            print("Couldn't find any checkpoint")
    else:
        print("Please provide a checkpoint")

    device = torch.device("cuda:0" if torch.cuda.is_available()
                          and input_args.gpu == True else "cpu")
    image_to_predict = process_image(input_args.input)
    prediction = predict(image_to_predict, model, device,
                         input_args.top_k, input_args.category_names)

    print_predict(prediction)


def predict(image, model, device, top_k, category_names):
    model.eval()
    image.to(device)

    with torch.no_grad():
        output = model.forward(Variable(image))

    top_prob, top_labels = torch.topk(output, top_k)
    top_prob = top_prob.exp()

    top_prob_array = top_prob.data.numpy()[0]
    top_labels_data = top_labels.data.numpy()

    inv_class_to_idx = {v: k for k, v in model.class_to_idx.items()}

    top_labels_list = top_labels_data[0].tolist()

    top_classes = [inv_class_to_idx[x] for x in top_labels_list]

    if category_names == None:
        mapped = zip(top_prob_array, top_classes)
    else:
        with open(f'{category_names}', 'r') as f:
            cat_to_name = json.load(f)
        class_names = [cat_to_name[x] for x in top_classes]
        mapped = zip(top_prob_array, top_classes, class_names)

    return mapped


if __name__ == '__main__':
    main()
