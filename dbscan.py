import numpy as np
import matplotlib.pyplot as plt
import math
import random
from scipy.spatial import KDTree
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
# visitlist类用于记录访问列表
# unvisitedlist记录未访问过的点
# visitedlist记录已访问过的点
# unvisitednum记录访问过的点数量
class visitlist:
    def __init__(self, count=0):
        self.unvisitedlist=[i for i in range(count)]
        self.visitedlist=list()
        self.unvisitednum=count

    def visit(self, pointId):
        self.visitedlist.append(pointId)
        self.unvisitedlist.remove(pointId)
        self.unvisitednum -= 1

def  dist(a, b):
    # 计算a,b两个元组的欧几里得距离
    return math.sqrt(np.power(a-b, 2).sum())

def mydbscan(dataSet, eps, minPts):
    # numpy.ndarray的 shape属性表示矩阵的行数与列数
    # 行数即表小所有点的个数
    print(dataSet.shape[0])
    nPoints = dataSet.shape[0]
    # (1) 标记所有对象为unvisited
    # 在这里用一个类vPoints进行实现
    count=nPoints
    vPoints = visitlist(count)
    # 初始化簇标记列表C，簇标记为 k
    k = -1
    C = [-1 for i in range(nPoints)]
    # 构建KD-Tree，并生成所有距离<=eps的点集合
    kd = KDTree(dataSet)
    while(vPoints.unvisitednum>0):
        # (3) 随机选择一个unvisited对象p
        p = random.choice(vPoints.unvisitedlist)
        # (4) 标t己p为visited
        vPoints.visit(p)
        # (5) if p 的$\varepsilon$-邻域至少有MinPts个对象
        # N是p的$\varepsilon$-邻域点列表
        N = kd.query_ball_point(dataSet[p], eps)
        if len(N) >= minPts:
            # (6) 创建个一个新簇C，并把p添加到C
            # 这里的C是一个标记列表，直接对第p个结点进行赋值
            k += 1
            C[p] = k
            # (7) 令N为p的$\varepsilon$-邻域中的对象的集合
            # N是p的$\varepsilon$-邻域点集合
            # (8) for N中的每个点p'
            for p1 in N:
                # (9) if p'是unvisited
                if p1 in vPoints.unvisitedlist:
                    # (10) 标记p'为visited
                    vPoints.visit(p1)
                    # (11) if p'的$\varepsilon$-邻域至少有MinPts个点，把这些点添加到N
                    # 找出p'的$\varepsilon$-邻域点，并将这些点去重新添加到N
                    M = kd.query_ball_point(dataSet[p1], eps)
                    if len(M) >= minPts:
                        for i in M:
                            if i not in N:
                                N.append(i)
                    # (12) if p'还不是任何簇的成员，把p'添加到c
                    # C是标记列表，直接把p'分到对应的簇里即可
                    if C[p1] == -1:
                        C[p1] = k
        # (15) else标记p为噪声
                else:
                    C[p1] = -1
    # (16) until没有标记为unvisited的对象
    return C
#加载数据集
iris=load_iris()
iris.keys()

#数据的条数和维数
n_samples,n_features=iris.data.shape
print("Number of sample:",n_samples)  #Number of sample: 150
print("Number of feature",n_features)  #Number of feature 4
#第一个样例
print(iris.data[0])      #[ 5.1  3.5  1.4  0.2]
print(iris.data.shape)    #(150, 4)
print(iris.target.shape)  #(150,)
#print(iris.target)

#归一化
mm = MinMaxScaler()
iris.data=mm.fit_transform(iris.data)
C1=mydbscan(iris.data,0.2,3)
plt.scatter(iris.data[:, 0], iris.data[:, 1], c=C1, marker='.')
plt.show()
