#-*- coding:utf-8 -*-

import urllib.request
import sys

def download(url, save_name):
    urllib.request.urlretrieve(url,save_name)

def loto(version=6):
    url= 'https://loto{0}.thekyo.jp/data/loto{0}.csv'.format(version)
    file = 'loto{0}.csv'.format(version)

    download(url, file)

    