import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation
"""
import keras as keras
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.layers import Activation
"""

import numpy as np

"""
Reference: Github http://github.com/gorogoroyasu/mnist-Grad-CAM
"""
class Model_mnist_gradcam(Model):
    def __init__(self, b_training=False, model_path=None):
        self.b_training = b_training
        self.build_model()

        if self.b_training is True:
            super(Model_mnist_gradcam, self).__init__(
                inputs=[self.inputs], 
                outputs=[self.predictions])
            if not model_path is None:
                self.load_weights(model_path)
            self.summary()
        else:
            super(Model_mnist_gradcam, self).__init__(
                inputs=[self.labels, self.inputs], 
                outputs=[self.predictions, self.g, self.a, self.gb_grad])
            if not model_path is None:
                self.load_weights(model_path)
            self.summary()

    def build_model(self):
        self.inputs = Input(shape=(28, 28, 1 ),name='imgs')
        self.aa = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same' )(self.inputs) # 28,28,32
        self.bb = MaxPooling2D(pool_size=(2, 2))(self.aa) # 14,14,32
        self.cc = Conv2D(64, (3, 3), activation='relu')(self.bb) # 12,12,64
        self.dd = MaxPooling2D(pool_size=(2, 2))(self.cc) # 6,6,64
        self.ee = Conv2D(128, (3, 3), activation='relu')(self.dd) # 4,4,128
        self.a = Conv2D(128, (3, 3), activation='relu')(self.ee) # 2,2,128
        self.b = MaxPooling2D(pool_size=(2, 2))(self.a) # 1,1,128

        self.c = Flatten()(self.b)

        self.d = Dense(64, activation='relu')(self.c)
        self.e = Dropout(0.5)(self.d)
        self.before_soft_max = Dense(10)(self.e)
        self.predictions = Activation('softmax')(self.before_soft_max)

        self.labels = Input((10,),name='labels')
        self.g = Lambda(lambda x: K.gradients(x[0] * x[2], x[1]), output_shape=list(self.a.shape))([self.before_soft_max, self.a, self.labels])
        self.cost = Lambda(lambda x: (-1) * K.sum(x[0] * K.log(x[1]), axis=1), output_shape=list(self.labels.shape))([self.labels, self.predictions])
        self.gb_grad = Lambda(lambda x: K.gradients(x[0], x[1]), output_shape=list(self.inputs.shape))([self.cost, self.inputs])

class Model_mnist_classification(Model):
    def __init__(self, model_path=None):
        self.build_model()
        super(Model_mnist_classification, self).__init__(
            inputs=[self.inputs], 
            outputs=[self.predictions])
        if not model_path is None:
            self.load_weights(model_path)
        self.summary()
    
    def __conv_pooling(self, x, kernels=[3], depths=[16], pool=3, is_need_conv=False):
        conv = x
        for (kernel, depth) in zip(kernels, depths):
            conv = Conv2D(depth, kernel_size=(kernel, kernel), activation='relu', padding='same')(conv)
        output = MaxPooling2D(pool_size=(pool, pool))(conv)
        if is_need_conv is True:
            return (output, conv)
        else:
            return output

    def build_model(self):
        self.inputs = Input(shape=(28, 28, 1 ),name='imgs')
        h1 = self.__conv_pooling(self.inputs, kernels=[3], depths=[32], pool=2) # h1.shape=(14,14,32)
        h2 = self.__conv_pooling(h1,          kernels=[3], depths=[64], pool=2) # h2.shape=(7,7,64)
        h3 = self.__conv_pooling(h2,          kernels=[3,3], depths=[128,128], pool=7) #1,1,128
        h4 = Flatten()(h3)
        h5 = Dense(64, activation='relu')(h4)
        h6 = Dropout(0.5)(h5)
        h7 = Dense(10)(h6)
        self.predictions = Activation('softmax')(h7)

class Model_mnist_classification_train(Model):
    def __init__(self, input_tensor, model_path=None):
        self.build_model(input_tensor)
        super(Model_mnist_classification, self).__init__(
            inputs=[self.inputs], 
            outputs=[self.predictions])
        if not model_path is None:
            self.load_weights(model_path)
        self.summary()
    
    def __conv_pooling(self, x, kernels=[3], depths=[16], pool=3, is_need_conv=False):
        conv = x
        for (kernel, depth) in zip(kernels, depths):
            conv = Conv2D(depth, kernel_size=(kernel, kernel), activation='relu', padding='same')(conv)
        output = MaxPooling2D(pool_size=(pool, pool))(conv)
        if is_need_conv is True:
            return (output, conv)
        else:
            return output

    def build_model(self, input_tensor):
        self.inputs = Input(tensor=input_tensor)
        h1 = self.__conv_pooling(self.inputs, kernels=[3], depths=[32], pool=2) # h1.shape=(14,14,32)
        h2 = self.__conv_pooling(h1,          kernels=[3], depths=[64], pool=2) # h2.shape=(7,7,64)
        h3 = self.__conv_pooling(h2,          kernels=[3,3], depths=[128,128], pool=7) #1,1,128
        h4 = Flatten()(h3)
        h5 = Dense(64, activation='relu')(h4)
        h6 = Dropout(0.5)(h5)
        h7 = Dense(10)(h6)
        self.predictions = Activation('softmax')(h7)


class Model_mnist_pix2pix(Model):
    def __init__(self, b_training=False, model_path=None):
        self.b_training = b_training
        self.build_model()
        super(Model_mnist_pix2pix, self).__init__(inputs=[self.x], outputs=[self.pred])
        if not model_path is None:
            self.load_weights(model_path)
        self.summary()

    def build_model(self):
        self.x = Input(shape=(28, 28, 1 ),name='imgs')
        self.h1 = Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same' )(self.x)
        self.h2 = Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same' )(self.h1)
        self.h3 = Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same' )(self.h2)
        self.pred = Conv2D(1, kernel_size=(5, 5), activation='sigmoid', padding='same' )(self.h3)
