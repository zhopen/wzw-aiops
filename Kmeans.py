#-*- coding: utf8 -*-
__author__ = 'Smith Duan[HPE]'

import json
import pandas as pd
import numpy as np
import Levenshtein
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


file = open('D:\\Python_example\\example_file\\CMCC\\dataSample\\test_2000.log', 'r')
log_df = pd.DataFrame(columns = ["log", "stream", "hostip", "hostname", "container_log_file"])
for line in file.readlines():
    log_info = json.loads(line)
    pd_data=pd.DataFrame.from_dict(log_info,orient='index').T
    log_df = log_df.append(pd_data, ignore_index=True)
file.close()
str_1 = log_df[u'log'].values[0]
row_num = log_df.iloc[:,0].size
i = 0
len_list = []
sd_list = []
while i < row_num:
    str_len = len(log_df[u'log'].values[i].encode('utf-8'))
    str_2 = log_df[u'log'].values[i]
    sd = Levenshtein.ratio(str_1,str_2)
    i += 1
    len_list.append(str_len)
    sd_list.append(sd)
log_df.insert(5,'message.len',len_list)
log_df.insert(6,'Levenshtein.ratio',sd_list)

r_l = []
a_zs = []
a = np.array(log_df[u'message.len'])
mua = np.average(a)
sigma = np.std(a)
for ia in a:
    i = (ia - mua) / sigma
    a_zs.append(i)
r_l.append(a_zs)
b_zs = []
b = np.array(log_df[u'Levenshtein.ratio'])
mub = np.average(b)
sigmb = np.std(b)
for ib in b:
    j = (ib - mub) / sigmb
    b_zs.append(j)
r_l.append(b_zs)
X = np.array(r_l).T

distance = []
k = []
sc_scores = []
for n_clusters in range(2,16):
    cls = KMeans(n_clusters).fit(X)
    def manhattan_distance(x,y):
        return np.sum(abs(x-y))
    distance_sum = 0
    for i in range(n_clusters):
        group = cls.labels_ == i
        members = X[group,:]
        for v in members:
            distance_sum += manhattan_distance(np.array(v), cls.cluster_centers_[i])
    distance.append(distance_sum)
    k.append(n_clusters)
    sc_score = silhouette_score(X,labels=cls.labels_, metric='euclidean')
    sc_scores.append(sc_score)
n = sc_scores.index(max(sc_scores))
n_cluster = k[n]
print("K values:", str(k[n]))
print("Distance = ", str(distance[n]))
plt.scatter(k, distance)
plt.plot(k, distance)
plt.xlabel("k")
plt.ylabel("distance")
plt.show()

df_t = pd.DataFrame(data= [log_df['Levenshtein.ratio']]).T
X = np.array(df_t)
km = KMeans(n_clusters = n_clusters,max_iter=30,n_init=3)
cl_mbk = km.fit(X)

cl_df = pd.Series(cl_mbk.labels_,name="Lable")  # 构建新的series（分类）
results = pd.concat([log_df, cl_df], axis = 1) # 横向连接（0是纵向），得到分类标签并追加到原始的数据框中

results.to_csv('D:\\Python_example\\example_file\\CMCC\\NLP_results\\results_2000_1.csv', sep='|', header=False, index=False)