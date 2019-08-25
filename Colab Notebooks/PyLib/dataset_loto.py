import pandas as pd
from keras.utils import np_utils

num = {6:6, 7:7}
max_num = {6:43, 7:37}

def get_max_num(ver):
    return max_num[ver]

def read_csv_loto(ver):
    list_col = list()
    if ver == 6:
        df = pd.read_csv('loto{0}.csv'.format(ver), encoding="shift-jis")
        fmt_col = '第{0}数字'
        list_col = [fmt_col.format(i+1) for i in range(num[ver])]
        list_col += [ 'BONUS数字' ]
    else:
        df = pd.read_csv('loto{0}.csv'.format(ver), encoding="cp932")
        fmt_col = '第{0}数字'
        list_col = [fmt_col.format(i+1) for i in range(num[ver])]
        fmt_col = 'BONUS数字{0}'
        list_col += [ fmt_col.format(i+1) for i in range(2) ]
    df2 = df[list_col]
    return df2

def to_categorical(arr, num_class):
    # (num_data, 7, 43)になるため、axis=1でreshapeする
    return np_utils.to_categorical(arr, num_class).sum(axis=1)
    