import numpy as np
import sys

from PyLib.engines      import *
from PyLib.models       import *
from PyLib.datasets     import *
from PyLib.predict_data import *
from PyLib.hyper_params import *
from PyLib              import model_utils

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

    noise_gen = np.random.uniform(0,1,size=[batch_size_false,100])
    generated_image = engine.predict( noise_gen,       mode='generator' )
    prob            = engine.predict( generated_image, mode='discriminator' )

if __name__ == '__main__':
    model_path = None
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
