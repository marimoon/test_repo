import numpy as np
import sys, os
from PIL import Image
import scipy.ndimage as ndimage

from PyLib.engines      import *
from PyLib.models       import *
from PyLib.datasets     import *
from PyLib.predict_data import *
from PyLib.hyper_params import *
from PyLib              import model_utils

def tile_image(arr, w=3, h=3):
    shape = arr.shape
    tile = np.zeros( (h * shape[1], w * shape[2]) )
    for i in range( shape[0] ):
        i_w = i % w
        i_h = int( (i-i_w) / w )

        s_h = i_h * shape[1]
        s_w = i_w * shape[2]
        tile[ s_h:(s_h+shape[1]), s_w:(s_w+shape[2]) ] = arr[i, :, : , 0]
    return tile

def main( model_path ):
    # network構造定義に必要なパラメータ
    num_channel     = 1
    gene_dim_hidden = 200
    disc_dropout_rate   = 0.25
    hyper_param = Hyper_Param_pix2pixGAN(
            model_path          = model_path, 
            num_channel         = num_channel,
            gene_dim_hidden     = gene_dim_hidden,
            disc_dropout_rate   = disc_dropout_rate
    )

    engine = Engine_pix2pixGAN()
    engine.load(hyper_param)

    dataset = Dataset_mnist()
    dataset.load()    # download mnist data
    dataset.x_train = np.reshape( dataset.x_train, dataset.x_train.shape + (1,) ) / 255.0
    
    from scipy import misc
    face = misc.face(gray=True)
    x_train_blur = np.zeros( dataset.x_train.shape )
    for i in range(dataset.x_train.shape[0]):
        x_train_blur[i, :, :, 0] = ndimage.gaussian_filter(dataset.x_train[i, :, :, 0], sigma=1.5)

    print('exec prediction')
    # prediction
    w = 10
    h = 10
    generated_image = engine.predict( x_train_blur[:(w*h), :, :, :], mode='generator' )
    prob            = engine.predict( x_train_blur[:(w*h), :, :, :], mode='discriminator' )

    print("Discriminator: prob =", prob.data)

    dir ='./predict'
    if os.path.exists(dir) is False:
        os.makedirs(dir)

    img = tile_image(generated_image.data, w=w, h=h)
    pil = Image.fromarray( np.uint8( img * 255 ) )
    pil.save( dir + 'pix2pixgan_generated.tif')

    img = tile_image(x_train_blur.data, w=w, h=h)
    pil = Image.fromarray( np.uint8( img * 255 ) )
    pil.save( dir + 'pix2pixgan_input.tif')

if __name__ == '__main__':
    model_path = './model/pix2pixgan/epoch010'

    argv = sys.argv
    if "--model_path" in argv:
        model_path = argv[argv.index("--model_path")+1]

    main( model_path )
