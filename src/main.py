# author: Chase Chai, made some modifications to Dr. Renato Cordeiro de Amorim and Cássio M. M. Pereira's code
# author of wkmeans and mwkmeans: Dr. Renato Cordeiro de Amorim, author of proclus: Cássio M. M. Pereira

# Subspace Clustering algorithms: include K-means, iK-means, WK-means, MWK-means, iMWK-means, Proclus

# parameters:
# datapath: txt format data set's path
# ypath: txt format target's path
# algorithm: algorithm's name
# k: number of clusters
# sep: separate delimiter used in reading data
# preprocess_method: preprocess_method to standardize the data
# replicates: number of times to run algorithms. The method will return the best clustering result
# max_ite: the maximum number of iterations
# beta: the weight exponent in wkmeans and mwkmeans
# init_centroids: initial centroids
# init_weights: initial weights
# init_weights_method: if init_weights is none, generate weights in random or in a fixed way, can either be "random" or "fixed"
# is_sparse: used in wk-means or mwk-means, if set to be 1, the weight of features with more than $threshold zeroes will be set to 0
# threshold: used in is_sparse
# l: parameter in proclus algorithm, average dimensions for each cluster
# minDeviation: parameter in proclus algorithm, used in selection of bad medoids
# A: parameter in proclus algorithm, the number of initial set of medoids
# B: parameter in proclus algorithm, the set of medoids found by greedy method, smaller than A


from clustering import Clustering
import numpy as np
import time
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

def data_preprocess(data, preprocess_method="standard"):
    if preprocess_method == "standard":
        return StandardScaler().fit_transform(data)
    elif preprocess_method == "maxmin":
        min_max_scaler = preprocessing.MinMaxScaler()
        return min_max_scaler.fit_transform(data)

def subspace_cluster(datapath, ypath, algorithm, k, sep=",", preprocess_method="standard", replicates=10, max_ite=100, beta=2, init_weights=None, init_centroids=None, init_weights_method="random", is_sparse=0, threshold=0.9, l=2, minDeviation=0.1, A=30, B=10):
    np.seterr(all='raise')
    cl = Clustering()
    data_ori = np.genfromtxt(datapath, delimiter=sep)
    y = np.genfromtxt(ypath)
    data = data_preprocess(data_ori, preprocess_method)

    if algorithm == 'K-Means':
        print 'using K-Means'
        start = time.time()
        [u, centroids, ite, dist_tmp] = cl.k_means(data, k, replicates)
        time_elapsed = time.time()-start
        acc = cl.my_math.compare_categorical_vectors(u, y)[0]
        print 'Time elapsed: ', time_elapsed
        print 'Accuracy: ', acc
        return [u, centroids, ite, dist_tmp], time_elapsed, acc
    elif algorithm == 'iK-Means':
        print 'using iK-Means'
        start = time.time()
        [u, centroids, ite, dist_tmp, init_centroids] = cl.ik_means(data, k)
        time_elapsed = time.time()-start
        acc = cl.my_math.compare_categorical_vectors(u, y)[0]
        print 'Time elapsed: ', time_elapsed
        print 'Accuracy: ', acc
        return [u, centroids, ite, dist_tmp, init_centroids], time_elapsed, acc
    elif algorithm == 'WK-Means':
        print 'using WK-Means'
        start = time.time()
        [u, centroids, weights, ite, dist_tmp] = cl.wk_means(data, k, beta, init_centroids=init_centroids, init_weights=init_weights, replicates=replicates, max_ite=max_ite, init_weights_method=init_weights_method, is_sparse=is_sparse, threshold=threshold)
        time_elapsed = time.time()-start
        acc = cl.my_math.compare_categorical_vectors(u, y)[0]
        print 'Time elapsed: ', time_elapsed
        print 'Accuracy: ', acc
        return [u, centroids, weights, ite, dist_tmp], time_elapsed, acc
    elif algorithm == 'MWK-Means':
        print 'using MWK-Means'
        start = time.time()
        [u, centroids, weights, ite, dist_tmp] = cl.mwk_means(data, k, beta, init_centroids=init_centroids, init_weights=init_weights, replicates=replicates, max_ite=max_ite, init_weights_method=init_weights_method, is_sparse=is_sparse, threshold=threshold)
        time_elapsed = time.time()-start
        acc = cl.my_math.compare_categorical_vectors(u, y)[0]
        print 'Time elapsed: ', time_elapsed
        print 'Accuracy: ', acc
        return [u, centroids, weights, ite, dist_tmp], time_elapsed, acc
    elif algorithm == 'iMWK-Means':
        print 'using iMWK-Means'
        start = time.time()
        [u, centroids, weights, ite, dist_tmp] = cl.imwk_means(data, beta, k)
        time_elapsed = time.time()-start
        acc = cl.my_math.compare_categorical_vectors(u, y)[0]
        print 'Time elapsed: ', time_elapsed
        print 'Accuracy: ', acc
        return [u, centroids, weights, ite, dist_tmp], time_elapsed, acc
    elif algorithm == 'proclus':
        print 'using proclus'
        start = time.time()
        M, D, A, acc = cl.proclus(data, y, k, l, minDeviation=minDeviation, A=A, B=B, niters=max_ite)
        time_elapsed = time.time()-start
        print 'Time elapsed: ', time_elapsed
        print "Accuracy: %.4f" % acc
        return [M, D, A], time_elapsed, acc
