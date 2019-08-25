try:
    import tensorflow.keras as keras
    from tensorflow.keras.models import Model
except:
    import keras as keras
    from keras.models import Model

# Modelの学習属性を切り替え
# val = True  : 学習ON
#     = False : 学習OFF
def make_trainable(model, val):
    model.trainable = val
    for l in model.layers:
        l.trainable = val


