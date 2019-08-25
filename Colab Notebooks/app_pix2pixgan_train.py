import numpy as np
import sys

from PyLib.engines      import *
from PyLib.models       import *
from PyLib.datasets     import *
from PyLib.predict_data import *
from PyLib.hyper_params import *
from PyLib              import model_utils

def main( model_path, epochs, batch_size, gene_lr, disc_lr ):
    hyper_param = Hyper_Param_pix2pixGAN(
            model_path          = model_path, 
            batch_size          = batch_size,
            gene_lr             = gene_lr,
            gene_dim_latent     = 100,
            gene_dim_hidden     = 200,
            disc_lr             = disc_lr,
            disc_shape          = (28,28,1),
            disc_dropout_rate   = 0.25
    )

    engine = Engine_pix2pixGAN()
    engine.init(hyper_param)

    # download mnist data
    dataset = Dataset_mnist()
    dataset.load()
    dataset.x_train = np.reshape( dataset.x_train, dataset.x_train.shape + (1,) ) / 255.0

    epochs = epochs
    engine.train( dataset.x_train, epochs, initial_epoch=0 )
    engine.save( './model/pix2pixgan/epoch{0:03d}'.format(epochs) )

if __name__ == '__main__':
    model_path = None
    epochs     = 10
    batch_size = 128
    num_channel = 3
    gene_lr    = 0.001
    disc_lr    = 0.0001

    argv = sys.argv
    if "--model_path" in argv:
        model_path = argv[argv.index("--model_path")+1]
    if "--epochs" in argv:
        epochs = int( argv[argv.index("--epochs")+1] )
    if "--batch_size" in argv:
        batch_size = int( argv[argv.index("--batch_size")+1] )
    if "--num_channel" in argv:
        num_channel = int( argv[argv.index("--num_channel")+1] )
    if "--gene_lr" in argv:
        gene_lr = float( argv[argv.index("--gene_lr")+1] )
    if "--disc_lr" in argv:
        disc_lr = float( argv[argv.index("--disc_lr")+1] )

    main( model_path, epochs, batch_size, gene_lr, disc_lr )
