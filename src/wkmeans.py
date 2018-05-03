# original author: Dr. Renato Cordeiro de Amorim, r.amorim@glyndwr.ac.uk
# modified by Chase Chai

# Weighted K-Means - subspace clustering with cluster dependant weight
#
# paper:
#
#Huang, J.Z., Ng, M.K., Rong, H., Li, Z.: Automated variable weighting in k-means type clustering. Pattern Analysis and
## Machine Intelligence, IEEE Transactions on 27(5), 657-668 (2005)
#
#Huang, J.Z., Xu, J., Ng, M., Ye, Y.: Weighting method for feature selection in k-#means. Computational Methods of
#Feature Selection, Chapman & Hall/CRC pp. 193-209 (2008)


import random as rd
import numpy as np


class WKMeans(object):

    def __init__(self, _my_math):
        self.my_math = _my_math

    def _get_dispersion_based_weights(self, data, centroids, k, beta, u, n_features, distance, p, dispersion_update='mean', is_sparse=0, threshold=0.9):
        dispersion = np.zeros([k, n_features])
        for k_i in range(k):
            for f_i in range(n_features):
                dispersion[k_i, f_i] = self.my_math.get_distance(data[u == k_i, f_i], centroids[k_i, f_i], distance, p)
        if dispersion_update == 'mean':
            dispersion += dispersion.mean()
        else:
            dispersion += dispersion_update
        #calculate the actual weight
        weights = np.zeros([k, n_features])
        if is_sparse == 1:
            for k_i in range(k):
                index = np.where(u == k_i)[0]
                fea, cnts = np.unique(np.where(data[index] == 0)[1], return_counts=True)
                dispersion[k_i, fea[np.where(cnts > threshold * data[index[0]].shape[0])]] = 0
        if beta != 1:
            exp = 1/float(beta - 1)
            for k_i in range(k):
                for f_i in range(n_features):
                    weights[k_i, f_i] = 1/((dispersion[k_i, f_i].repeat(n_features)/dispersion[k_i, :])**exp).sum() if dispersion[k_i, f_i] != 0 else 0
        else:
            for k_i in range(k):
                weights[k_i, dispersion[k_i, :].argmin()] = 1
        return weights

    def __wk_means(self, data, k, beta, centroids=None, weights=None, max_ite=10, distance='SqEuclidean', p=None, init_weights_method="random", is_sparse=0, threshold=0.9):
        #runs WK-Means (or MWK-Means) once
        #returns -1, -1, -1, -1, -1 if there is an empty cluster
        n_entities, n_features = data.shape
        if centroids is None:
            centroids = data[rd.sample(range(n_entities), k), :]
        if weights is None:
            if init_weights_method == "random":
                weights = np.random.rand(k, n_features)
                weights = weights/weights.sum(axis=1).reshape([k, 1])
            elif init_weights_method == 'fixed':
                weights = np.zeros([k, n_features])
                weights[:, :] = 1/float(n_features)
        previous_u = np.array([])
        ite = 1
        while ite <= max_ite:
            dist_tmp = np.zeros([n_entities, k])
            #assign entities to cluster
            for k_i in range(k):
                dist_tmp[:, k_i] = self.my_math.get_distance(data, centroids[k_i, :], distance, p, weights[k_i, :]**beta)
            u = dist_tmp.argmin(axis=1)
            #put the sum of distances to centroids in dist_tmp
            dist_tmp = np.sum(dist_tmp[np.arange(dist_tmp.shape[0]), u])
            #stop if there are no changes in the partitions
            if np.array_equal(u, previous_u):
                return u, centroids, weights, ite, dist_tmp
            #update centroids
            cnt = 0
            for k_i in range(k):
                entities_in_k = u == k_i
                #Check if cluster k_i has lost all its entities
                if sum(entities_in_k) == 0:
                    cnt += 1
                    continue
                centroids[k_i, :] = self.my_math.get_center(data[entities_in_k, :], distance, p)
            #update weights
            if cnt == k:
                return np.array([-1]), np.array([-1]), np.array([-1]), np.array([-1]), np.array([-1])
            weights = self._get_dispersion_based_weights(data, centroids, k, beta, u, n_features, distance, p, is_sparse, threshold)
            previous_u = u[:]
            #print weights,centroids
            ite += 1
        return u, centroids, weights, ite, dist_tmp

    def wk_means(self, data, k, beta, init_centroids=None, init_weights=None, distance='SqEuclidean', replicates=1, p=None, max_ite=10, init_weights_method="random", is_sparse=0, threshold=0.9):
        #Weighted K-Means
        final_dist = float("inf")
        for replication_i in range(replicates):
            #loops up to max_ite to try to get a successful clustering for this replication
            for i in range(max_ite):
                [u, centroids, weights, ite, dist_tmp] = self.__wk_means(data, k, beta, init_centroids, init_weights, max_ite, distance, p, init_weights_method, is_sparse, threshold)
                if u[0] != -1:
                    break
            if u[0] == -1:
                raise Exception('Cannot generate a single successful clustering')
            #given a successful clustering, check if its the best
            if dist_tmp < final_dist:
                final_u = u[:]
                final_centroids = centroids[:]
                final_ite = ite
                final_dist = dist_tmp
        return final_u, final_centroids, weights, final_ite, final_dist
