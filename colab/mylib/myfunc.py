def test():
    print('myfunc.test')

def load_npz(file):
    a = np.load(file)
    return [ val[1] for val in a.items() ]
