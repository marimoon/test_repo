import numpy as np
import os

class Dataset_mnist(object):
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_val   = None
        self.y_val   = None

    def load(self):
        try:
            from tensorflow.keras.datasets import mnist
        except:
            from keras.datasets import mnist

        (x_train, y_train), (x_val, y_val) = mnist.load_data()
        self.x_train = x_train
        self.y_train = y_train
        self.x_val   = x_val
        self.y_val   = y_val

    def savez(self, file=None):
        if file is None:
            if os.name == 'nt':
                file = '../colab/data/mnist'
            else:
                file = '/content/drive/My Drive/colab/data/mnist'
        np.savez(file, self.x_train, self.y_train, self.x_val, self.y_val)

    def load_wo_web(self, file=None):
        if file is None:
            if os.name == 'nt':
                file = '../colab/data/mnist.npz'
            else:
                file = '/content/drive/My Drive/colab/data/mnist.npz'

        npzfile = np.load(file)
        self.x_train = npzfile['arr_0']
        self.y_train = npzfile['arr_1']
        self.x_val   = npzfile['arr_2']
        self.y_val   = npzfile['arr_3']

    def save_for_textlinedataset(self, num=30):
        if os.name == 'nt':
            work_dir = '../colab/data/mnist/'
        else:
            work_dir = '/content/drive/My Drive/colab/data/mnist/'
        if os.path.exists(work_dir) is False:
            os.makedirs(work_dir)

        with open(work_dir + 'input.txt', 'w') as fp:
            for i in range(num):
                fp.writelines('{0}{1:03d}.npy,{2}\n'.format(work_dir, i, self.y_train[i]) )
                np.save('{0}{1:03d}.npy'.format(work_dir, i), self.x_train[i,:,:])
        return work_dir + 'input.txt'

    def dbg_check(self):
        print(self.x_train.shape)
        print(self.y_train.shape)
        print(self.x_val.shape)
        print(self.y_val.shape)
