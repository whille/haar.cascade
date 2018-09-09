#!/usr/bin/env python
import numpy as np
import os


def image_integral(a):
    return a.cumsum(axis=0).cumsum(axis=1)


FrameSize = 24
# {sizeX, sizeY): (weights of (xi,yi)}
dic_feature_templates = {
    (2, 1): (1, -1, -2, 2, 1, -1),
    (1, 2): (-1, 2, -1, 1,-2,1),
    (3, 1): (1, -1, -2, 2, 2, -2, -1, 1),
    (1, 3): (1, -2, 2, -1, -1, 2, -2, 1),
    (2, 2): (1, -2, 1, -2, 4, -2, 1, -2, 1),
}


def gen_features(frameSize):
    for (sizeX, sizeY), weights in dic_feature_templates.iteritems():
        for startX in range(frameSize - sizeX):
            for startY in range(frameSize - sizeY):
                for width in range(sizeX, frameSize - startX, sizeX):
                    for height in range(sizeY, frameSize - startY, sizeY):
                        yield feature_ops(width, height, sizeX, sizeY, startX, startY, weights)


def feature_ops(width, height, sizeX, sizeY, startX, startY, weights):
    i = 0
    lst = []
    for xi in range(startX, startX + width +1, width / sizeX):
        for yi in range(startY, startY + height +1, height / sizeY):
            weight = weights[i]
            lst.append((weight, xi, yi))
            i += 1
    return lst


def feature_dic(frameSize=FrameSize):
    dic = {}
    for i, v in enumerate(gen_features(frameSize)):
        dic[i] = v
    return dic


def transfer_img(img, dic_fea):
    """transfer image to feature vector"""
    ii = image_integral(img)
    res = np.zeros(len(dic_fea), dtype=np.int16)
    for i in range(len(dic_fea)):
        lst = dic_fea[i]
        fea_i = 0
        for ratio, xi, yi in lst:
            fea_i += ratio * ii[xi, yi]
        res[i] = fea_i
    return res


def train(F, ys, T):
    """
    xs: features of images
    ys: 0/1, classification
    """
    M, N = F.shape
    W = np.ones((T, N))
    alpha = np.zeros(T)
    # initialze w0
    m = (ys == 0).sum()
    W[0, ys == 0] /= m
    W[0, ys != 0] /= (N - m)
    H = []
    for t in range(T):
        # normalize weights to probability distribution
        W[t] /= W[t].sum()
        e_t, (j, p, theta) = find_best(F, ys, W[t])
        H.append((j, p, theta))
        # update weights
        beta_t = e_t / (1 - e_t)
        alpha[t] = -np.log(beta_t)
        h = weak_classifier(p, F[j], theta)
        if t < T - 1:
            W[t + 1] = W[t]
            W[t + 1, h != ys] *= beta_t
    return alpha, H


def predict(dic_fea, img, alpha, H):
    f = transfer_img(img, dic_fea)
    hs = [weak_classifier(p, f[j], theta) for (j, p, theta) in H]
    return alpha.dot(hs) >= 0.5 * alpha.sum()


def weak_classifier(p, f_j, theta):
    return p * f_j < p * theta


def find_best(F, ys, wt):
    """
    iterate all args(theta, parity, feature) to find best weak classifier minimize error e_t
    args:
        F: features[M,N], M = len(features), N = len(ys)
    return:
        best: j, p, theta
    """
    M, N = F.shape
    arr = F * wt
    # seems reasonable
    best_theta = (
        arr[:, ys == 1].sum(axis=1) + arr[:, ys != 1].sum(axis=1)) / 2  # M * 1
    print 'best_theta:', best_theta
    e_t, best_args = None, []
    for p in (-1, 1):
        for j in range(M):
            h = weak_classifier(p, F[j], best_theta[j])
            e = (wt * np.abs(h - ys)).sum()
            if e_t is None or e < e_t:
                e_t = e
                best_args = (j, p, best_theta[j])
    return e_t, best_args


def load_img(fname):
    from matplotlib import image
    im = image.imread(fname)
    return im


def gen_data(maxn = 1e4):
    for i, key in enumerate(['NFACES', 'FACES']):
        print i, key
        c = 0
        folder ="./TrainingImages/{}/".format(key)
        for fname in os.listdir(folder):
            img = load_img(os.sep.join((folder, fname)))
            print '.',
            yield img, i
            c += 1
            if c >= maxn:
                break
        print


def save_data(dic_fea, N, fname):
    F = np.zeros((2*N, len(dic_fea)), dtype=np.int16)
    ys = np.zeros(2*N, dtype=np.int8)
    for i, (img, y) in enumerate(gen_data(N)):
        f = transfer_img(img, dic_fea)
        F[i] = f
        ys[i] = y
    F = F.T
    np.save(fname, (F, ys))
    return F, ys


def cache_load(dic_fea, N):
    fname = './features/train.{}.npy'.format(N)
    if os.path.exists(fname):
        return np.load(fname)
    else:
        return save_data(dic_fea, N, fname)
