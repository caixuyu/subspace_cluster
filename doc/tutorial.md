### 使用文档
#### 简介
本工具包括了k-means, ik-means, wk-means, mwk-means, imwk-means, proclus算法的实现。
#### 调用方法
test/test.py提供了调用的测试代码，src/main.py是程序的入口，导入main.py后调用main中的subspace_clustering方法即可，例如

    from main import subspace_cluster
    [u, centroids, weights, ite, dist_tmp], time_elapsed, acc = subspace_cluster("data/iris.csv", "data/iris_y.csv", "MWK-Means", k=3, beta=2)

#### 参数
- datapath: 输入数据的路径，txt格式，原始数据
- ypath: 输入y值的路径，txt格式，用来验证准确率的，若无y值，传入空文件即可
- algorithm: 算法名称，可选的值有K-Means，iK-Means，WK-Means，MWK-Means，iMWK-Means，proclus
- k: 类别个数
- sep: 读取文件的分隔符，默认为逗号
- preprocess_method: 数据预处理方法，默认为z-socre均值，可选的值有standard和maxmin。standard为z-score均值，maxmin为最大最小化的归一化方式
- replicates: 算法运算次数，会返回最好的结果
- max_ite: 最大迭代次数
- beta: wkmeans和mwkmeans中的beta值，权重的幂值
- init_centroids: 初始聚类中心，只在wkmeans和mwkmeans中使用，默认为随机产生
- init_weights: 初始权重，只在wkmeans和mwkmeans中使用，默认产生方式由init_weights_method决定
- init_weights_method: 初始权重的产生方式，只在wkmeans和mwkmeans中使用，默认为随机产生，可选值为"random"或"fixed"，取"fixed"时初始权重均为1/n，n是特征个数
- is_sparse: 只在wkmeans和mwkmeans中使用，该参数是为了避免原始数据中大量为0的特征的权重过大，当该参数值设为1的时候，会将超过threshold的数量为0的特征的权重设为0，该值默认为0
- threshold: 只有当is_sparse为1的时候启用，默认为0.9
- l: 只在proclus中启用，是每个类簇的平均维度数
- minDeviation: 只在proclus中启用，用来筛选bad medoid
- A: 只在proclus中启用，初始聚类中心候选集中点的数量
- B: 只在proclus中启用，通过greedy算法选出的聚类中心候选集中点的数量，小于A

#### 返回结果
第一个返回值为一个list，weights只在wk-means，mwk-means和imwk-means中返回，M, D, A只在proclus中返回，proclus的返回结果和k-means类算法的结果完全不一样。

- u: 每条数据属于的类别
- centroids: 聚类中心
- weights: 特征的权重
- ite: 使用的步数
- dist_tmp: 距离误差
- M: 类簇中心
- D: 子空间的特征
- A: 数据属于的类别
- time_elapsed: 消耗的时间
- acc: 准确率