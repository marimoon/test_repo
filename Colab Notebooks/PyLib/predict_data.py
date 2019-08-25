class Predict_Data(object):
    def __init__(self):
        pass

class Data_Image(Predict_Data):
    def __init__(self, data):
        self.data = data

class Data_Scalar(Predict_Data):
    def __init__(self, data):
        self.data = data

