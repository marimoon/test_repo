class Hyper_Param_Base(object):
    def __init__(self):
        pass

#--------------------------------------------------------------------
class Hyper_Param_pix2pixGAN(Hyper_Param_Base):
    def __init__(self, 
            model_path          = None, 
            batch_size          = 128,
            num_channel         = 3,
            gene_lr             = 1e-4,
            gene_dim_hidden     = 200,
            disc_lr             = 1e-4,
            disc_dropout_rate   = 0.25
            ):
        self.model_path         = model_path
        self.batch_size         = batch_size
        self.num_channel        = num_channel
        self.gene_lr            = gene_lr
        self.gene_dim_hidden    = gene_dim_hidden
        self.disc_lr            = disc_lr
        self.disc_dropout_rate  = disc_dropout_rate

#--------------------------------------------------------------------
class Hyper_Param_mnist_GAN(Hyper_Param_Base):
    def __init__(self, 
            model_path          = None, 
            batch_size          = 128,
            gene_lr             = 1e-4,
            gene_dim_latent     = 100,
            gene_dim_hidden     = 200,
            disc_lr             = 1e-4,
            disc_shape          = (28,28,1),
            disc_dropout_rate   = 0.25
            ):
        self.model_path         = model_path
        self.batch_size         = batch_size
        self.gene_lr            = gene_lr
        self.gene_dim_latent    = gene_dim_latent
        self.gene_dim_hidden    = gene_dim_hidden
        self.disc_lr            = disc_lr
        self.disc_shape         = disc_shape
        self.disc_dropout_rate  = disc_dropout_rate



