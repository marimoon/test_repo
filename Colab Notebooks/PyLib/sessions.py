import tensorflow.keras as keras
import tensorflow as tf

"""
実行環境の参考:
>>> import tensorflow.keras as keras
>>> import tensorflow as tf
>>> tf.__version_W_
'1.13.1'
>>> keras.__version__
'2.2.4-tf'
"""

# Singlton Class
class Single_Session(object):
    _instance = None

    def __init__(self):
        pass

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)

            self.config = tf.ConfigProto()
            self.config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=self.config)
            K.set_session(self.sess)

            return cls._instance

