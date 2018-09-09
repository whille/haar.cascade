#!/usr/bin/env python

from matplotlib import pyplot as plt
import numpy as np
from my_ada import *


def test_feature():
    dic_fea = feature_dic(19)
    print len(dic_fea)
    print dic_fea[48531]


def test_image():
    dic_fea = feature_dic()
    white_img = np.ones((FrameSize, FrameSize), dtype=np.int8)
    res = transfer_img(white_img, dic_fea)
    plt.plot(res)
    plt.show()


def test_find_best():
    M, N = 3, 5
    np.random.seed(1911)
    F = np.random.randint(0, 10, (M, N))
    ys = np.array([1, 1, 0, 0, 1])
    wt = np.array([1, 2, 3, 4, 5], np.float)
    wt /= wt.sum()
    print 'F:', F
    print 'wt:', wt
    print 'ys:', ys
    print find_best(F, ys, wt)


def test_ada():
    T = 10
    N = 100
    frameSize=19
    dic_fea = feature_dic(frameSize)
    F, ys = cache_load(dic_fea, N)
    print 'F.shape:', F.shape
    alpha, H = train(F, ys, T)
    print alpha, H
    img, y = next(gen_data())
    print predict(dic_fea, img, alpha, H), y


def test_loadimg():
    for img, y in gen_data():
        assert (19,19) == img.shape
        print type(img)
        break
