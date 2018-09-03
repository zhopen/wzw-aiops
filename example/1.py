# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 14:19:36 2018

@author: zhan
"""

from sklearn.datasets import make_blobs 
import matplotlib.pyplot as plt
X, y = make_blobs( n_samples= 200,
                   n_features= 2,
                   centers= 4,
                   cluster_std= 1,
                   center_box=(-10.0, 10.0),
                   shuffle= True,
                   random_state= 1);

plt.figure( figsize=( 6, 4), dpi= 144)
plt.xticks(())
plt.yticks(()) 
plt.scatter( X[:, 0], X[:, 1], s= 20, marker='o');

from sklearn.cluster import KMeans 
n_clusters = 3 
kmean = KMeans( n_clusters= n_clusters) 
kmean.fit(X); 
print(" kmean: k={}, cost={}".format( n_clusters, int( kmean.score(X))))

labels = kmean.labels_
centers = kmean.cluster_centers_
markers = ['o', '^', '*']
colors = ['r', 'b', 'y'] 
plt.figure( figsize=( 6, 4), dpi= 144) 
plt.xticks(())
plt.yticks(()) # 画 样本 for c in range( n_clusters):
# 画 样本 
for c in range( n_clusters):
    cluster = X[ labels == c] 
    plt.scatter(cluster[:, 0], cluster[:, 1], marker= markers[c], s= 20, c=colors[c]) 
# 画出 中心点 
plt.scatter( centers[:, 0], centers[:, 1], marker='o', c="white", alpha= 0.9, s= 300) 
for i, c in enumerate( centers): 
    plt.scatter( c[ 0], c[ 1], marker='$% d$' % i, s= 50, c= colors[ i])

