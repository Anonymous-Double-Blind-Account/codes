########################## Build silo datasets for different data ########################

from dataset.cifar10 import CIFAR10_silo
from dataset.fmnist import FMNIST_silo
from dataset.mnist import MNIST_silo
from dataset.cifar100 import CIFAR100_silo
from dataset.gtsrb import GTSRB_silo

def build_slios(args):
    root = '../datasets'
    silotr = []
    silote = []

    assert args.data == 'cifar10' or args.data == 'fmnist' or args.data == 'mnist' or args.data == 'gtsrb' or args.data == 'cifar100', "Not supported dataset!"

    for i in range(args.num_silo):
        if args.data == 'cifar10':
            datatr = CIFAR10_silo(root, args, 'train', i)
            datate = CIFAR10_silo(root, args, 'val', i)
        elif args.data == 'cifar100':
            datatr = CIFAR100_silo('../datasets/cifar100', args, 'train', i)
            datate = CIFAR100_silo('../datasets/cifar100', args, 'val', i)
        elif args.data == 'fmnist':
            datatr = FMNIST_silo(root, args, 'train', i)
            datate = FMNIST_silo(root, args, 'val', i)
        elif args.data == 'gtsrb':
            datatr = GTSRB_silo('../datasets/GTSRB', args, 'train', i)
            datate = GTSRB_silo('../datasets/GTSRB', args, 'val', i)
        else:
            datatr = MNIST_silo(root, args, 'train', i)
            datate = MNIST_silo(root, args, 'val', i)
        silotr.append(datatr)
        silote.append(datate)

    return silotr, silote
