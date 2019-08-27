try:
    import tensorflow.keras as keras
    from tensorflow.keras.layers import Input, Dense, Lambda
    from tensorflow.keras.models import Model
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Flatten, LeakyReLU, Reshape
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D
    from tensorflow.keras.layers import GlobalAveragePooling2D
    from tensorflow.keras import backend as K
    from tensorflow.keras.layers import Activation
except:
    import keras as keras
    from keras.layers import Input, Dense, Lambda
    from keras.models import Model
    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten, LeakyReLU, Reshape
    from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D
    from keras import backend as K
    from keras.layers import Activation
import numpy as np

""" -------------------------------------------------------------
Grad-CAM for MNIST
Reference: Github http://github.com/gorogoroyasu/mnist-Grad-CAM
------------------------------------------------------------- """
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

""" -------------------------------------------------------------
Image classification for MNIST
------------------------------------------------------------- """
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

""" -------------------------------------------------------------
pix2pix for MNIST
------------------------------------------------------------- """
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

""" -------------------------------------------------------------
Generator of DCGAN for MNIST
Reference : Matsuo Lab(DL4US) of Tokyo univ, Lesson.6-section.1
------------------------------------------------------------- """
class Model_mnist_GAN_Generator(Model):
    def __init__(self, model_path=None, dim_latent=100, dim_hidden=200):
        self.dim_latent = dim_latent
        self.dim_hidden = dim_hidden
        self.build_model()
        super(Model_mnist_GAN_Generator, self).__init__(inputs=[self.input_latent], outputs=[self.pred])
        if not model_path is None:
            self.load_weights(model_path + '/model_mnist_gan_generator.h5')
        self.summary()

    def build_model(self):
        self.input_latent = Input( shape = [self.dim_latent] )
        nch = self.dim_hidden
        x = Dense(nch*14*14, kernel_initializer='glorot_normal')(self.input_latent) # 100 -> 200*14*14
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Reshape( [14, 14, nch] )(x) # 200*14*14 -> 14x14x200
        x = UpSampling2D(size=(2, 2))(x) # 14x14x200 -> 28x28x200
        x = Conv2D(int(nch/2), (3, 3), padding='same', kernel_initializer='glorot_uniform')(x) # 28x28x200 -> 28x28x100
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(int(nch/4), (3, 3), padding='same', kernel_initializer='glorot_uniform')(x) # 28x28x100 -> 28x28x50
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(1, (1, 1), padding='same', kernel_initializer='glorot_uniform')(x) # 28x28x50 -> 28x28x1
        self.pred = Activation('sigmoid')(x)

""" -------------------------------------------------------------
Discriminator of DCGAN for MNIST
Reference : Matsuo Lab(DL4US) of Tokyo univ, Lesson.6-section.1
------------------------------------------------------------- """
class Model_mnist_GAN_Discriminator(Model):
    def __init__(self, model_path=None, shape=(28,28,1), dropout_rate=0.25):
        self.shape = shape
        self.dropout_rate = dropout_rate
        self.build_model()
        super(Model_mnist_GAN_Discriminator, self).__init__(self.input_img, self.pred)
        if not model_path is None:
            self.load_weights(model_path + '/model_mnist_gan_discriminator.h5')
        self.summary()

    def build_model(self):
        self.input_img = Input(shape=self.shape) # 28x28x1
        x = Conv2D(256, (5, 5), padding = 'same', kernel_initializer='glorot_uniform', strides=(2, 2))(self.input_img) # 28x28x1 -> 14x14x256
        x = LeakyReLU(0.2)(x)
        x = Dropout(self.dropout_rate)(x)
        x = Conv2D(512, (5, 5), padding = 'same', kernel_initializer='glorot_uniform', strides=(2, 2))(x) # 14x14x256 -> 7x7x512
        x = LeakyReLU(0.2)(x)
        x = Dropout(self.dropout_rate)(x)
        x = Flatten()(x) # 7x7x512 -> 7*7*512
        x = Dense(256)(x) # 7*7*512 -> 256
        x = LeakyReLU(0.2)(x)
        x = Dropout(self.dropout_rate)(x)
        self.pred = Dense(2, activation='softmax')(x) # 256 -> 2

""" -------------------------------------------------------------
Discriminator of DCGAN for MNIST
------------------------------------------------------------- """
class Model_mnist_GAN_conbined(Model):
    def __init__(self, generator, discriminator):
        self.build_model(generator, discriminator)
        super(Model_mnist_GAN_conbined, self).__init__(self.input_latent, self.pred)
        self.summary()

    def build_model(self, generator, discriminator):
        self.input_latent = Input( shape=(generator.dim_latent,) )
        image_gene = generator(self.input_latent)
        self.pred  = discriminator(image_gene)

""" -------------------------------------------------------------
Generator of DCGAN with pix2pix for CIFAR10
------------------------------------------------------------- """
class Model_pix2pixGAN_Generator(Model):
    def __init__(self, model_path=None, num_channel=3, dim_hidden=128):
        self.num_channel = num_channel
        self.dim_hidden = dim_hidden
        self.build_model()
        super(Model_pix2pixGAN_Generator, self).__init__(inputs=[self.input_img], outputs=[self.pred])
        if not model_path is None:
            self.load_weights(model_path)
        self.summary()

    def build_model(self):
        nch = self.dim_hidden
        self.input_img = Input( shape = (None, None, self.num_channel) )
        x = BatchNormalization()(self.input_img)
        x = Conv2D(nch, (5, 5), padding='same', kernel_initializer='glorot_uniform')(x)
        x = Activation('relu')(x)
        x = Conv2D(int(nch/2), (5, 5), padding='same', kernel_initializer='glorot_uniform')(x) # 28x28x100 -> 28x28x50
        x = Activation('relu')(x)
        x = Conv2D(nch, (5, 5), padding='same', kernel_initializer='glorot_uniform')(x)
        x = Activation('relu')(x)
        x = Conv2D(self.num_channel, (1, 1), padding='same', kernel_initializer='glorot_uniform')(x) # 28x28x50 -> 28x28x1
        self.pred = Activation('sigmoid')(x)

""" -------------------------------------------------------------
Discriminator of DCGAN with pix2pix for CIFAR10
------------------------------------------------------------- """
class Model_pix2pixGAN_Discriminator(Model):
    def __init__(self, model_path=None, num_channel=3, dropout_rate=0.25):
        self.num_channel = num_channel
        self.dropout_rate = dropout_rate
        self.build_model()
        super(Model_pix2pixGAN_Discriminator, self).__init__(self.input_img, self.pred)
        if not model_path is None:
            self.load_weights(model_path)
        self.summary()

    def build_model(self):
        self.input_img = Input(shape=(None, None, self.num_channel) )
        x = Conv2D(256, (5, 5), padding = 'same', kernel_initializer='glorot_uniform', strides=(2, 2))(self.input_img)
        x = LeakyReLU(0.2)(x)
        x = Dropout(self.dropout_rate)(x)
        x = Conv2D(256, (5, 5), padding = 'same', kernel_initializer='glorot_uniform', strides=(2, 2))(x)
        x = LeakyReLU(0.2)(x)
        x = Dropout(self.dropout_rate)(x)
        x = GlobalAveragePooling2D(data_format='channels_last')(x) # 任意の画像サイズに対応するためGAPを使用
        x = Flatten()(x)
        x = Dense(256)(x)
        x = LeakyReLU(0.2)(x)
        x = Dropout(self.dropout_rate)(x)
        self.pred = Dense(2,activation='softmax')(x)

""" -------------------------------------------------------------
Discriminator of DCGAN for CIFAR10
comment : This model is used for training of generator.
------------------------------------------------------------- """
class Model_pix2pixGAN_conbined(Model):
    def __init__(self, generator, discriminator):
        self.build_model(generator, discriminator)
        super(Model_pix2pixGAN_conbined, self).__init__(self.input_img, self.pred)
        self.summary()

    def build_model(self, generator, discriminator):
        self.input_img = Input( shape=(None, None, generator.num_channel) )
        image_gene = generator(self.input_img)
        self.pred  = discriminator(image_gene)

