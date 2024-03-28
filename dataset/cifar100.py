import numpy as np
from torch.utils.data import Dataset
import pickle
import os, cv2
import platform

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

img_rows, img_cols = 32, 32
input_shape = (img_rows, img_cols, 3)
def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000,3072)
        X = X.reshape(10000, *input_shape, order="F")
        X = X.transpose((0, 2, 1, 3))
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte
    
# load cifar100
def load_CIFAR100_batch(filename, size):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['fine_labels']
        X = X.reshape(size,3072)
        X = X.reshape(size, *input_shape, order="F")
        X = X.transpose((0, 2, 1, 3))
        Y = np.array(Y)
        return X, Y
    
def load_CIFAR100(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,2):
        f = os.path.join(ROOT, 'train')
        X, Y = load_CIFAR100_batch(f, 50000)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR100_batch(os.path.join(ROOT, 'test'), 10000)
    return Xtr, Ytr, Xte, Yte

class CIFAR100_silo(Dataset):
    """
    load cifar100 dataset from local directory, with silo partition
    """
    def __init__(self, root, args, folder, silo):
        super().__init__()
        self.noise = args.noise
        self.num_silo = args.num_silo
        cifar10_dir = '../datasets/cifar-100-python/'
        x_train, y_train, x_test, y_test = load_CIFAR100(cifar10_dir) # load cifar100 dataset

        # partition the dataset into silos
        xtr = x_train[silo*(args.sample):(silo+1)*(args.sample)].astype(np.float32)
        ytr = y_train[silo*(args.sample):(silo+1)*(args.sample)].astype(np.int64)
        xte = x_test[silo*(args.sample//4):(silo+1)*(args.sample//4)].astype(np.float32)
        yte = y_test[silo*(args.sample//4):(silo+1)*(args.sample//4)].astype(np.int64)
        if folder == 'val':
            self.data = self.transform(xte, silo)
            self.label = yte
        else:
            self.data = self.transform(xtr, silo)
            self.label = ytr
        print('{} folder image and label shape:'.format(folder), self.data.shape, self.label.shape)

    # rotate the image
    def augment_silos(self, x, silo, num_silo):
        angles = [0, 0, 0, -50, 120, 0, -65] if num_silo == 5 else [0, 0, 0, -50, -50, -50, 120, 120, 120]
        
        angle = angles[silo]
        if len(x.shape) == 3:
            h, w, c = x.shape
            gaussian = np.random.normal(0, 100**0.5, (h,w, c))
        else:
            h, w = x.shape
            gaussian = np.random.normal(0, 100**0.5, (h,w))
        if self.noise:
            if silo > 5:
                x = x+gaussian.astype(np.float32)
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        return cv2.warpAffine(x, M, (w, h))

    def transform(self, x, silo):
        rotate_x = []
        for i in range(x.shape[0]):
            rotate_x.append(self.augment_silos(x[i], silo, self.num_silo)) # augment data silos with different angles
        rotate_x = np.asarray(rotate_x)
        rotate_x = np.transpose(rotate_x, (0, 3, 1, 2))
        return rotate_x
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.label[index]
