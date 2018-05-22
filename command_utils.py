import argparse


def predict_input():
    parser = argparse.ArgumentParser(
        description='Provide input, checkpoint, top_k, category_names and gpu')

    parser.add_argument('--input', type=str, help='path to input image')
    parser.add_argument('--checkpoint', type=str, help='path to checkpoint')
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of top_k to show')
    parser.add_argument('--category_names', type=str,
                        help='path to cat names file')
    parser.add_argument('--gpu', type=bool_helper,
                        default='gpu', help='whether gpu should be used for or not')

    return parser.parse_args()


def train_input():
    parser = argparse.ArgumentParser(
        description='Provide image_dir, save_dir, architecture, hyperparameters such as learningrate, num of hidden_units, epochs and whether to use gpu or not')

    parser.add_argument('--data_dir', type=str, default='flowers/',
                        help='path to image folder')
    parser.add_argument('--save_dir', type=str, default='',
                        help='folder where model checkpoints gets saved to')
    parser.add_argument('--arch', type=str,
                        default='densenet121', help='choose between vgg and densenet')
    parser.add_argument('--learning_rate', type=float,
                        default=0.001, help='learning_rate for model')
    parser.add_argument('--hidden_units', type=str,
                        default="500", help='hidden_units for model')
    parser.add_argument('--epochs', type=int,
                        default=10, help='epochs for model')
    parser.add_argument('--gpu', type=bool_helper,
                        default='gpu', help='whether gpu should be used for or not')

    return parser.parse_args()


def print_model(args):
    print("==> creating {} model with:\n".format(args.arch),
          "* learning_rate: {}\n".format(args.learning_rate),
          "* hidden_units: {}\n".format(args.hidden_units),
          "* epochs: {}\n".format(args.epochs),
          "* gpu enabled: {}\n".format(args.gpu))


def print_predict(mapped):
    print("\n******PREDICTION******")
    for x, y, *z in mapped:
        print(
            f'Probability: {x} || Class: {y}{"" if z == [] else " || Flower: {}".format(z[0])}')


def bool_helper(str):
    if str.lower() in ['yes', 'y', 'true', 'gpu', 't', '1']:
        return True
    elif str.lower() in ['no', 'n', 'false', 'cpu', 'f', '0']:
        return False
    else:
        return argparse.ArgumentTypeError('Boolean value was expected as input')
