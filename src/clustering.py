# orignal Author: Dr. Renato Cordeiro de Amorim, r.amorim@glyndwr.ac.uk
# modified by Chase Chai, modify wkmeans and mwkmeans methods and add proclus method

import numpy as np
import random as rd
from mymath import MyMath
from kmeans import KMeans
from wkmeans import WKMeans
from mwkmeans import MWKMeans
from proclus import PROClus


class Clustering(object):

    def __init__(self):
        self.my_math = MyMath()

    def k_means(self, data, k, replicates=1, init_centroids=None, dist='SqEuclidean', p=None, max_ite=100):
        km = KMeans(self.my_math)
        return km.k_means(data, k, replicates, init_centroids, dist, p, max_ite)

    def ik_means(self, data, k=None, theta=0, distance='SqEuclidean', p=None):
        km = KMeans(self.my_math)
        return km.ik_means(data, k, theta, distance, p)

    def wk_means(self, data, k, beta, init_centroids=None, init_weights=None, distance='SqEuclidean', replicates=1, p=None, max_ite=10, init_weights_method="random", is_sparse=0, threshold=0.9):
        wkm = WKMeans(self.my_math)
        return wkm.wk_means(data, k, beta, init_centroids, init_weights, distance, replicates, p, max_ite, init_weights_method, is_sparse, threshold)

    def mwk_means(self, data, k, p, init_centroids=None, init_weights=None, replicates=1, max_ite=100, init_weights_method="random", is_sparse=0, threshold=0.9):
        mwk = MWKMeans(self.my_math)
        return mwk.mwk_means(data, k, p, init_centroids, init_weights, replicates, max_ite, init_weights_method, is_sparse, threshold)

    def imwk_means(self, data, p, k=None, theta=0):
        mwk = MWKMeans(self.my_math)
        return mwk.imwk_means(data, p, k, theta)

    def proclus(self, data, target, k=2, l=2, minDeviation=0.1, A=30, B=3, niters=30):
        _proclus = PROClus(self.my_math)
        seed = np.random.randint(low = 0, high = 1239831)
        M, D, A = _proclus.proclus(data, k, l, minDeviation, A, B, niters, seed)
        acc = _proclus.computeBasicAccuracy(A, target)
        return M, D, A, acc
