import numpy as np
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import viz
import argparse
from explainers import CASO, VanillaGradExplainer, \
    IntegrateGradExplainer, SmoothGradExplainer
import torchvision.models as models
import torch.nn.functional as F

import matplotlib
import matplotlib.pyplot as plt

model_names = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

image_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()])

def load_images(example_ids=None):
	images = []
	for example in example_ids:
		image = image_transform(viz.pil_loader(example))
		images.append(image)
	images = torch.stack(images, dim=0).cuda()
	return images

parser = argparse.ArgumentParser(description='Model Interpretation')
parser.add_argument('--image_path', metavar='PATH', help='path to image', 
					default='duck.jpeg')
parser.add_argument('--arch', metavar='ARCH', default='resnet50',
                    choices=model_names, help='model architecture: ' +
                    ' | '.join(model_names) + ' (default: resnet18)')
parser.add_argument('--second-order', default=False, action='store_true',
                    help='use hessian')
parser.add_argument('--smooth', default=False, action='store_true',
                    help='use smooth interpretation')
parser.add_argument('--relu-hessian', default=False, action='store_true',
                    help='use hessian formula for a relu network')
parser.add_argument('--lambda1', default=1e-4, type=float, \
					help='L1 regularization coefficient', dest='lambda1')
parser.add_argument('--lambda2', default=4, type=float, \
					help='L2 regularization coefficient', dest='lambda2')
parser.add_argument('--n_iter', default=10, type=int, \
					help='number of iterations', dest='n_iter')
parser.add_argument('--optimizer', default='proximal', type=str, \
					choices= ['sgd', 'lbfgs', 'adam', 'proximal'], \
					help='optimizer type', dest='optimizer')
parser.add_argument('--lr', default=0.05, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--init', default='zero', type=str, choices= ['zero', 'random'], 
                    help='initialization type', dest='init')
parser.add_argument('--n_samples', default=16, type=int, \
					help='number of samples', dest='n_samples')
parser.add_argument('--stddev', default=0.15, type=float, \
					help='standard deviation', dest='stddev')
parser.add_argument('--times_input', default=False, action='store_true',
                    help='multiply with input')
parser.add_argument('--magnitude', default=False, action='store_true',
                    help='normalize the interpretation')

def main():
    args = parser.parse_args()
    example_ids = [args.image_path]

    model = models.__dict__[args.arch](pretrained=True)
    model.eval()
    model.cuda()

    images = load_images(example_ids=example_ids)

    caso = CASO(second_order=args.second_order, smooth=args.smooth, \
                full_hessian=args.relu_hessian)
    delta = caso.explain(model, images, lambda1=args.lambda1, lambda2=args.lambda2, \
                 n_iter=args.n_iter, optim=args.optimizer, lr=args.lr, 
                 init=args.init, n_samples=args.n_samples, stddev_spread=args.stddev, \
                 times_input=args.times_input, magnitude=args.magnitude)
    delta_np = delta.detach().cpu().numpy()
    delta_viz = viz.agg_default(delta_np)
    delta_viz = viz.clip(delta_viz)
    plt.imshow(delta_viz[0, :], cmap='gray')
    plt.axis('off')
    plt.savefig('delta.png')
    plt.close()

if __name__ == '__main__':
    main()