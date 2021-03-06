import numpy as np
import sys, os
from PIL import Image

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

def main( model_path, epochs, batch_size, gene_lr, disc_lr ):
    hyper_param = Hyper_Param_mnist_GAN(
            model_path          = model_path, 
            batch_size          = batch_size,
            gene_dim_latent     = 100,
            gene_dim_hidden     = 200,
            disc_shape          = (28,28,1),
            disc_dropout_rate   = 0.25
    )

    engine = Engine_mnist_GAN()
    engine.load(hyper_param)

    # generate latent variables
    w = 10
    h = 10
    noise_gen = Data_Scalar( np.random.uniform(0,1,size=[w*h,100]) )

    # prediction
    generated_image = engine.predict( noise_gen,       mode='generator' )
    prob            = engine.predict( generated_image, mode='discriminator' )

    print("Discriminator: prob =", prob.data)

    img = tile_image(generated_image.data, w=w, h=h)
    pil = Image.fromarray( np.uint8( img * 255 ) )
    pil.save('predict/mnist_gan_generated.tif')

    import matplotlib.pyplot as plt
    plt.plot( range(w*h), prob.data[:, 0] , range(w*h), prob.data[:, 1])
    plt.show()

if __name__ == '__main__':
    model_path = './model/mnist_gan/mnist_epoch050'
    epochs     = 10
    batch_size = 128
    gene_lr    = 0.001
    disc_lr    = 0.0001

    argv = sys.argv
    if "--model_path" in argv:
        model_path = argv[argv.index("--model_path")+1]
    if "--epochs" in argv:
        epochs = int( argv[argv.index("--epochs")+1] )
    if "--batch_size" in argv:
        batch_size = int( argv[argv.index("--batch_size")+1] )
    if "--gene_lr" in argv:
        gene_lr = float( argv[argv.index("--gene_lr")+1] )
    if "--disc_lr" in argv:
        disc_lr = float( argv[argv.index("--disc_lr")+1] )

    main( model_path, epochs, batch_size, gene_lr, disc_lr )
