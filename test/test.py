import sys
sys.path.append('../src/') 
from main import subspace_cluster

[u, centroids, weights, ite, dist_tmp], time_elapsed, acc = subspace_cluster("../data/iris.csv", "../data/iris_y.csv", "MWK-Means", 3)
