import os, time
from PIL import Image
import numpy as np
import tensorflow as tf
try:
    import tensorflow.keras as keras
except:
    import keras

from .hyper_params   import *
from .models         import *
from .datasets       import *
from .predict_data   import *
from .               import model_utils

class Engine_Base(object):
    def __init__(self):
        pass
    def init(self, hyper_param):
        pass
    def load(self, hyper_param):
        pass
    def predict(self, input_data):
        pass
    def train(self, listImageFile, epochs):
        pass
    def save(self, save_path):
        pass

#--------------------------------------------------------------------
# 通常のpix2pixでは像にぼけが生じやすいため、
# Discriminatorで画像を評価することでsharpな画像を生成させる学習モデル。
class Engine_pix2pixGAN(Engine_Base):
    def __init__(self):
        self.name_generator     = '/model_pix2pixgan_generator.h5'
        self.name_discriminator = '/model_pix2pixgan_discriminator.h5'

    def init(self, hyper_param):
        self.hyper_param = hyper_param

        if self.hyper_param.model_path is not None:
            model_gene_file = hyper_param.model_path + self.name_generator
            model_disc_file = hyper_param.model_path + self.name_discriminator

        # create models
        self.generator = Model_pix2pixGAN_Generator(
                            model_path   = model_gene_file, 
                            num_channel  = hyper_param.num_channel, 
                            dim_hidden   = hyper_param.gene_dim_hidden )

        self.generator.compile(
            loss = 'mean_squared_error',
            optimizer = keras.optimizers.Adam( lr = self.hyper_param.gene_lr)
        )

        self.discriminator = Model_pix2pixGAN_Discriminator(
                            model_path   = hyper_param.model_path + self.name_discriminator, 
                            num_channel  = hyper_param.num_channe, 
                            dropout_rate = hyper_param.disc_dropout_rate )
        self.discriminator.compile(
            loss = 'categorical_crossentropy',
            optimizer = keras.optimizers.Adam( lr = self.hyper_param.disc_lr)
        )
        model_utils.make_trainable(self.discriminator, False)

        self.gan = Model_pix2pixGAN_conbined(
                            self.generator,
                            self.discriminator )
        self.gan.compile(
            loss = 'categorical_crossentropy',
            optimizer = keras.optimizers.Adam( lr = self.hyper_param.gene_lr)
        )

    def load(self, hyper_param):
        self.generator = Model_pix2pixGAN_Generator(
                            model_path   = hyper_param.model_path + self.name_generator , 
                            num_channel  = hyper_param.num_channel, 
                            dim_hidden   = hyper_param.gene_dim_hidden )

        self.discriminator = Model_pix2pixGAN_Discriminator(
                            model_path   = hyper_param.model_path + self.name_discriminator , 
                            num_channel  = hyper_param.num_channel, 
                            dropout_rate = hyper_param.disc_dropout_rate )
        
        self.gan = Model_pix2pixGAN_conbined(
                            self.generator,
                            self.discriminator )

    def dump(self, batch, epoch, w=6, h=6, gene=True):
        from PIL import Image
        dir = './debug/'
        if gene is True:
            dir += 'pix2pixgan_train_fake'
        else:
            dir += 'pix2pixgan_train_true'

        if os.path.exists(dir) is False:
            os.makedirs(dir)

        tile = np.zeros((28*h, 28*w, 3))
        for j in range(h*w):
            x = j%w
            y = int( (j - x) / w )
            tile[(28*y):(28*y+28), (28*x):(28*x+28), 0] = batch[j, :, :, 0]
            tile[(28*y):(28*y+28), (28*x):(28*x+28), 1] = batch[j, :, :, 0]
            tile[(28*y):(28*y+28), (28*x):(28*x+28), 2] = batch[j, :, :, 0]
        img = Image.fromarray( np.uint8( tile * 255.0 ))
        img.save( dir + '/tile_epoch{0:03}.tif'.format(epoch))

    def predict(self, input_data, mode='generator'):
        if mode == 'generator':
            pred = self.generator.predict( input_data.data )
            pred = Data_Image(pred)
        else:
            pred = self.discriminator.predict( input_data.data )
            pred = Data_Scalar(pred)
        return pred

    # x_train : 低画質画像, y_train : 高画質画像
    def train(self, x_train, y_train, epochs, initial_epoch=0):
        batch_size_true  = self.hyper_param.batch_size
        batch_size_false = self.hyper_param.batch_size
        batch_size = batch_size_true + batch_size_false

        num_data = x_train.shape[0]
        step_per_epoch = int( num_data / batch_size_true )

        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print('Start Training pix2pixGAN')
        print('num_data         = {0:10d}'.format(num_data) )
        print('batch_size       = {0:10d}'.format(batch_size) )
        print('batch_size_false = {0:10d}'.format(batch_size_false) )
        print('batch_size_true  = {0:10d}'.format(batch_size_true) )
        print('step_per_epoch   = {0:10d}'.format(step_per_epoch) )
        print('initial_epoch    = {0:10d}'.format(initial_epoch) )
        print('epochs           = {0:10d}'.format(epochs) )
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

        # pretrain generator
        print('pretrain generator')
        self.generator.fit( x_train, y_train, batch_size=batch_size, epochs=epochs )

        # pretrain discriminator
        images_train = np.concatenate((y_train, x_train))
        labels_train = np.zeros([x_train.shape[0]*2,2])
        labels_train[:x_train.shape[0],1] = 1     # true
        labels_train[x_train.shape[0]:,0] = 1     # false        
        model_utils.make_trainable(self.discriminator, True)
        self.discriminator.fit( images_train, labels_train, batch_size=batch_size, epochs=int(epochs/2) )
        model_utils.make_trainable(self.discriminator, False)

        for e in range(epochs):
            time_gene_fake  = 0.0
            time_train_gene = 0.0
            time_train_disc = 0.0
            loss_train_gene = 0.0
            loss_train_disc = 0.0
            for i in range(step_per_epoch):
                #------------------------------------------------------------
                # 1. train discriminator
                shuffled_index = np.random.randint(
                                    0,                  # MIN
                                    x_train.shape[0],   # MAX
                                    size=batch_size_true )
                x_train_latent = x_train[shuffled_index,:,:,:]
                x_train_true   = y_train[shuffled_index,:,:,:]
                
                st = time.time()
                x_train_fake = self.generator.predict(x_train_latent)  # create by generator
                time_gene_fake += time.time()-st
                if i == 0: # debug
                    self.dump(x_train_fake, e, gene=True, w=8, h=8)

                model_utils.make_trainable(self.discriminator, True)

                images_train = np.concatenate((x_train_true, x_train_fake))

                labels_train = np.zeros([batch_size,2])
                labels_train[:batch_size_true,1] = 1     # true
                labels_train[batch_size_true:,0] = 1     # false

                st = time.time()
                loss_disc = self.discriminator.train_on_batch(images_train, labels_train)
                loss_train_disc += loss_disc
                time_train_disc += time.time()-st

                #------------------------------------------------------------
                # 2. train generator
                model_utils.make_trainable(self.discriminator, False)

                shuffled_index = np.random.randint(
                                    0,                  # MIN
                                    x_train.shape[0],   # MAX
                                    size=batch_size_true )
                x_train_true   = y_train[shuffled_index,:,:,:]

                labels_train = np.zeros([batch_size_false,2])
                labels_train[:,1] = 1

                loss_gene = self.gan.train_on_batch(x_train_true, labels_train )
                loss_train_gene += loss_gene
                time_train_gene += time.time()-st

            fmt = "[{0:3d} / {1:3d}] Time=[Prep={2:.1f}, G={3:.1f}, D={4:.1f}] Loss=[G:{5:.3f}, D:{6:.3f}]"
            print(fmt.format(
                        initial_epoch+e+1,      # 0
                        initial_epoch+epochs,   # 1
                        time_gene_fake,         # 2
                        time_train_gene,        # 3
                        time_train_disc,        # 4
                        loss_train_gene,        # 5
                        loss_train_disc,        # 6
                        ))

    def save(self, save_path):
        if os.path.exists(save_path) is False:
            os.makedirs( save_path )
        self.generator.save_weights( save_path + self.name_generator )
        self.discriminator.save_weights( save_path + self.name_discriminator )
        # model付帯情報
        cond = Model_Condition()
        cond.save(save_path)

#--------------------------------------------------------------------
class Engine_mnist_GAN(Engine_Base):
    def __init__(self):
        self.name_generator     = '/model_mnist_gan_generator.h5'
        self.name_discriminator = '/model_mnist_gan_discriminator.h5'

    def init(self, hyper_param):
        self.hyper_param = hyper_param

        # create models
        self.generator = Model_mnist_GAN_Generator(
                            model_path   = hyper_param.model_path, 
                            dim_latent   = hyper_param.gene_dim_latent, 
                            dim_hidden   = hyper_param.gene_dim_hidden )

        self.discriminator = Model_mnist_GAN_Discriminator(
                            model_path   = hyper_param.model_path, 
                            shape        = hyper_param.disc_shape, 
                            dropout_rate = hyper_param.disc_dropout_rate )
        self.discriminator.compile(
            loss = 'categorical_crossentropy',
            optimizer = keras.optimizers.Adam( lr = self.hyper_param.disc_lr)
        )
        model_utils.make_trainable(self.discriminator, False)

        self.gan = Model_mnist_GAN_conbined(
                            self.generator,
                            self.discriminator )
        self.gan.compile(
            loss = 'categorical_crossentropy',
            optimizer = keras.optimizers.Adam( lr = self.hyper_param.gene_lr)
        )

    def load(self, hyper_param):
        if self.hyper_param.model_path is not None:
            model_gene_file = hyper_param.model_path + self.name_generator
            model_disc_file = hyper_param.model_path + self.name_discriminator

        self.generator = Model_mnist_GAN_Generator(
                            model_path   = model_gene_file, 
                            dim_latent   = hyper_param.gene_dim_latent, 
                            dim_hidden   = hyper_param.gene_dim_hidden )

        self.discriminator = Model_mnist_GAN_Discriminator(
                            model_path   = hmodel_disc_file, 
                            shape        = hyper_param.disc_shape, 
                            dropout_rate = hyper_param.disc_dropout_rate )
        
        self.gan = Model_mnist_GAN_conbined(
                            self.generator,
                            self.discriminator )
    
    def save(self, save_path):
        if os.path.exists(save_path) is False:
            os.makedirs( save_path )
        self.generator.save_weights( save_path + self.name_generator )
        self.discriminator.save_weights( save_path + self.name_discriminator )
        # model付帯情報
        cond = Model_Condition()
        cond.save(save_path)

    def predict(self, input_data, mode='generator'):
        if mode == 'generator':
            pred = self.generator.predict( input_data.data )
            pred = Data_Image(pred)
        else:
            pred = self.discriminator.predict( input_data.data )
            pred = Data_Scalar(pred)
        return pred

    def dump(self, batch, epoch, w=6, h=6, gene=True):
        from PIL import Image
        dir = './debug/'
        if gene is True:
            dir += 'mnist_gan_train_false'
        else:
            dir += 'mnist_gan_train_true'

        if os.path.exists(dir) is False:
            os.makedirs(dir)

        tile = np.zeros((28*h, 28*w, 3))
        for j in range(h*w):
            x = j%w
            y = int( (j - x) / w )
            tile[(28*y):(28*y+28), (28*x):(28*x+28), 0] = batch[j, :, :, 0]
            tile[(28*y):(28*y+28), (28*x):(28*x+28), 1] = batch[j, :, :, 0]
            tile[(28*y):(28*y+28), (28*x):(28*x+28), 2] = batch[j, :, :, 0]
        img = Image.fromarray( np.uint8( tile * 255.0 ))
        img.save( dir + '/tile_epoch{0:03}.tif'.format(epoch))

    def train(self, x_train, epochs, initial_epoch=0):
        batch_size_true  = self.hyper_param.batch_size
        batch_size_false = self.hyper_param.batch_size
        batch_size = batch_size_true + batch_size_false

        num_data = x_train.shape[0]
        step_per_epoch = int( num_data / batch_size_true )

        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print('Start Training GAN')
        print('num_data         = {0:10d}'.format(num_data) )
        print('batch_size       = {0:10d}'.format(batch_size) )
        print('batch_size_false = {0:10d}'.format(batch_size_false) )
        print('batch_size_true  = {0:10d}'.format(batch_size_true) )
        print('step_per_epoch   = {0:10d}'.format(step_per_epoch) )
        print('initial_epoch    = {0:10d}'.format(initial_epoch) )
        print('epochs           = {0:10d}'.format(epochs) )
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        for e in range(epochs):
            time_gene_fake  = 0.0
            time_train_gene = 0.0
            time_train_disc = 0.0
            loss_train_gene = 0.0
            loss_train_disc = 0.0
            for i in range(step_per_epoch):
                #------------------------------------------------------------
                # 1. train discriminator
                shuffled_index = np.random.randint(
                                    0,                          # MIN
                                    x_train.shape[0],   # MAX
                                    size=batch_size_true )
                image_batch_true = x_train[shuffled_index,:,:,:]
                
                st = time.time()
                noise_gen = np.random.uniform(0,1,size=[batch_size_false,100])
                image_batch_fake = self.generator.predict(noise_gen)  # create by generator
                time_gene_fake += time.time()-st
                if i == 0: # debug
                    self.dump(image_batch_fake, e, gene=True, w=8, h=8)

                model_utils.make_trainable(self.discriminator, True)

                images_train = np.concatenate((image_batch_true, image_batch_fake))

                labels_train = np.zeros([batch_size,2])
                labels_train[:batch_size_true,1] = 1     # true
                labels_train[batch_size_true:,0] = 1     # false

                st = time.time()
                loss_disc = self.discriminator.train_on_batch(images_train, labels_train)
                loss_train_disc += loss_disc
                time_train_disc += time.time()-st

                #------------------------------------------------------------
                # 2. train generator
                model_utils.make_trainable(self.discriminator, False)

                noise_train = np.random.uniform(0,1,size=[batch_size_false,100])

                labels_train = np.zeros([batch_size_false,2])
                labels_train[:,1] = 1

                loss_gene = self.gan.train_on_batch(noise_train, labels_train )
                loss_train_gene += loss_gene
                time_train_gene += time.time()-st

            fmt = "[{0:3d} / {1:3d}] Time=[Prep={2:.1f}, G={3:.1f}, D={4:.1f}] Loss=[G:{5:.3f}, D:{6:.3f}]"
            print(fmt.format(
                        initial_epoch+e+1,      # 0
                        initial_epoch+epochs,   # 1
                        time_gene_fake,         # 2
                        time_train_gene,        # 3
                        time_train_disc,        # 4
                        loss_train_gene,        # 5
                        loss_train_disc,        # 6
                        ))


#--------------------------------------------------------------------
# Engine No
ENGINE_NO_MNIST_GAN  = 900
ENGINE_NO_PIX2PIXGAN = 901

dispatch_engine = {
    ENGINE_NO_MNIST_GAN  : Engine_mnist_GAN,
    ENGINE_NO_PIX2PIXGAN : Engine_pix2pixGAN,
}

import datetime
# モデル付帯情報クラス
class Model_Condition(object):
    def __init__(self):
        self.file            = '/model_cond.txt'
        self.dt_fmt          = '%Y/%m/%d %H:%M:%S'
        self.dispatch_engine = dispatch_engine
        self.engine_no       = -1
        self.dt              = datetime.datetime.now()

    def set_cond(self, instance_engine):
        # engine特定
        self.engine_no = -1
        for (key, engine) in self.dispatch_engine:
            if type(instance_engine) == engine:
                self.engine_no = key

        # 保存時刻取得
        self.dt = datetime.datetime.now()

    def save(self, save_path):
        with open( save_path + self.file, 'w') as fp:
            fp.writelines('Date     {0}\n'.format(self.dt.strftime(self.dt_fmt))) 
            fp.writelines('Engine   {0}\n'.format( self.engine_no )) 

    def load(self, path):
        with open( path + self.file, 'r' ) as fp:
            for line in fp.readlines():
                if (line[0] == '#') or \
                   (line[0] == '\n') or\
                   (line[0] == '\t'):
                   continue
                
                buf = line
                # delete \n,\t
                if (buf in '\n') or (buf in '\t'):
                    buf = buf[:-1]
                # delete comment
                if (buf in '#'):
                    buf[ : buf.index('#') ]

                if 'Date ' in buf:
                    self.dt = datetime.datetime.strptime(
                                buf[4:].rstrip(' ').lstrip(' ') )
                if 'Engine ' in buf:
                    self.engine_no = int( buf[6:].rstrip(' ').lstrip(' ') )

    def get_engine_class(self):
        return self.dispath_engine[self.engine_no]

#--------------------------------------------------------------------
