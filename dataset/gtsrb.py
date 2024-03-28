import os
import cv2
from torch.utils.data import Dataset
import numpy as np
from sklearn.utils import shuffle

class GTSRB_silo(Dataset):
    """
    load gtsrb dataset from local directory, with silo partition
    """
    def __init__(self, root, args, folder, silo):
        super().__init__()
        self.noise = args.noise
        self.num_silo = args.num_silo
        datadir = '../datasets/GTSRB'

        # load the data
        if os.path.exists(os.path.join(datadir, 'x_train.npy')):
            x_train = np.load(os.path.join(datadir, 'x_train.npy'))
            y_train = np.load(os.path.join(datadir, 'y_train.npy'))
            x_test = np.load(os.path.join(datadir, 'x_test.npy'))
            y_test = np.load(os.path.join(datadir, 'y_test.npy'))
        else:
            print('No data file found!')
            x_train, y_train = self.load_GTSRB(datadir, 'Final_Training') # (39209, 32, 32, 3) | (39209,) (43 classes)
            x_test, y_test = self.load_GTSRB(datadir, 'Final_Test') # (12630, 32, 32, 3) | (12630,) (43 classes)

            print('Data loaded!')
            # shuffle the data
            x_train, y_train = shuffle(x_train, y_train)
            x_test, y_test = shuffle(x_test, y_test)
            print('x_train, y_train, x_test, y_test shape:', x_train.shape, y_train.shape, x_test.shape, y_test.shape)

            # save the data
            np.save(os.path.join(datadir, 'x_train.npy'), x_train)
            np.save(os.path.join(datadir, 'y_train.npy'), y_train)
            np.save(os.path.join(datadir, 'x_test.npy'), x_test)
            np.save(os.path.join(datadir, 'y_test.npy'), y_test)
        
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
    
    def load_GTSRB(self, datadir, folder):
        x, y = [], []
        if folder == 'Final_Training':
            datafolder = datadir + '/' + folder + '/' + 'Images'
            for dirname in os.listdir(datafolder):
                for filename in os.listdir(datafolder + '/' + dirname):
                    if filename == 'GT-{}.csv'.format(dirname):
                        continue
                    img = cv2.imread(datafolder + '/' + dirname + '/' + filename)
                    img = cv2.resize(img, (32, 32))
                    x.append(img)
                    y.append(int(dirname))
            return np.asarray(x), np.asarray(y)
        elif folder == 'Final_Test':
            datafolder = datadir + '/' + folder + '/' + 'Images'
            labelpath = datadir + '/' + folder + '/' + 'GT-final_test.csv'
            labels = []
            with open(labelpath, 'r') as f:
                for line in f.readlines()[1:]:
                    labels.append(int(line.split(';')[-1].strip()))
            for filename in os.listdir(datafolder):
                if filename == 'GT-final_test.test.csv':
                    continue
                img = cv2.imread(datafolder + '/' + filename)
                img = cv2.resize(img, (32, 32))
                x.append(img)
                y.append(labels[int(filename.split('.')[0])])
            return np.asarray(x), np.asarray(y)
    
    # augment data silos with different methods to create different distributions
    def augment_silos(self, x, silo, num_silo):
        angles = [0, 0, 0, -50, 120] if num_silo == 5 else [0, 0, 0, -50, -50, -50, 120, 120, 120]
        
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
            rotate_x.append(self.augment_silos(x[i], silo, self.num_silo))
        rotate_x = np.asarray(rotate_x)
        rotate_x = rotate_x.transpose(0, 3, 1, 2)
        return rotate_x
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.label[index]
