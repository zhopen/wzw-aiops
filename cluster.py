#-*- coding: utf8 -*-
import pandas as pd
import numpy as np
import Levenshtein
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.externals import joblib
import sys,os,re,time
import argparse

def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sample', required=True, help='path to sample file')
    parser.add_argument('-c', '--csv',  help='path to csv file as a result')
    parser.add_argument('-p', '--pkl', help='path to pkl file')
    parser.add_argument('-m', '--maxclusters', help='max clusters')
    args = parser.parse_args()
    if args.csv == None: args.csv = args.sample + '.csv'
    if args.pkl == None: args.pkl = args.sample + '.pkl'
    if args.maxclusters == None: args.maxclusters = 15
    return args


def draw(x,y, xlabel='x', ylabel='y',line=True, scatter=True, grid=True):
    plt.figure(figsize=[8,8])
    ax = plt.subplot(111)
    if scatter == True: ax.scatter(x, y, marker=",",s=2)
    if line == True: ax.plot(x, y)
    if grid == True: ax.grid()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()
    
def draw3d(x,y,z, xlabel='x', ylabel='y',zlabel='z'):
    plt.figure(figsize=[8,8])
    ax = plt.subplot(111,projection='3d')
    ax.scatter(x, y, z, marker=".")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.show()
    
def inputnum(comment, ifnotnum='continue'):
    while True:
        str = ""
        try: 
            str = input(comment)
            return  int(str)
        except: 
            if ifnotnum == 'exit': 
                sys.exit()
            elif ifnotnum == 'break':
                print("bbbb")
                return None
            else: 
                continue

def zscore(a):
    return (a - a.mean()) / a.std()          #z-score标准化

def str_sum(str):
    b = bytes(str,encoding="utf-8")
    return sum(b)

def import_sample(sample_file):
    df = pd.read_json(sample_file, lines=True)
    df["log"] = df['log'].str.strip()
    return df

def extract_feature(df, txt_ref):   
    '''抽取特征和构建特征矩阵：ratio，len'''
    df['message.len'] = df[u'log'].apply(lambda x:len(x))
    df['Levenshtein.ratio'] = df[u'log'].apply(lambda x:Levenshtein.ratio(txt_ref,x))
    df['string.sum'] = df[u'log'].str.replace(re.compile('^\d{4}-\d{2}-\d{2} \d+:\d+:\d+.\d+'),"").apply(str_sum)
    return df

def make_X(df):
    '''抽取有用的特征，对特征值做标准化处理'''
    len_zs = zscore(df['message.len'])
    ratio_zs = zscore(df['Levenshtein.ratio'])
    strsum_zs = zscore(df['string.sum'])
    draw(list(len_zs),list(ratio_zs), xlabel='len_zs', ylabel='ratio_zs', line=False, grid=False)
    draw(list(len_zs),list(strsum_zs), xlabel='len_zs', ylabel='strsum_zs', line=False, grid=False)
    draw3d(list(len_zs),list(ratio_zs), z=list(strsum_zs), xlabel='len_zs', ylabel='ratio_zs', zlabel='strsum_zs')
    X = np.concatenate(([len_zs],[ratio_zs],[strsum_zs]), axis=0).T
    #X = np.concatenate(([ratio_zs]), axis=0).T
    df['len_zscore'], df['ratio_zscore'],df['strsum_zs'] = len_zs, ratio_zs, strsum_zs
    return X,df

def train_check(X, max_clusters):
    '''聚类分析（Kmean算法），然后通过silhouette_score算法评估出最佳的分类数'''
    k,distance, sc_scores = range(2,max_clusters+1),[], []
    for n_clusters in k:
        cls = KMeans(n_clusters).fit(X)
        distance_sum = 0
        for i in range(n_clusters):
            group = cls.labels_ == i
            members = X[group,:]     
            distance_sum += (abs(members - cls.cluster_centers_[i])).sum() #同一标签的所有样本与质心的manhattan distance的和
        distance.append(distance_sum)
        #k.append(n_clusters)
        sc_score = silhouette_score(X,labels=cls.labels_, metric='euclidean')
        sc_scores.append(sc_score)
    best_clusters_idx = sc_scores.index(max(sc_scores))
    print("K values（最佳分类数量）:", str(k[best_clusters_idx]))
    print("Distance = ", str(distance[best_clusters_idx]))
    draw(k, distance, xlabel='k', ylabel='distance')
    draw(k, sc_scores, xlabel='k', ylabel='silhouette_score')
    return k[best_clusters_idx]

def train(X, n_clusters, df):
    kmeans = KMeans(n_clusters,max_iter=30,n_init=3).fit(X)
    df["label"] = kmeans.labels_
    return df, kmeans

TXT_REF = '[YYYY][INFO] abcdefghjklmnopqrstuvwxyz0123456789'

if __name__ == '__main__':
    args = _get_args()
    print(args)
    
    print('\n----------------------------------------------------------------')
    print("读入样本文件数据："+args.sample)
    
    log_df = import_sample(args.sample)
    print('提取特征。参考文本：' + TXT_REF)
    
    log_df = extract_feature(log_df, TXT_REF)
    X, log_df = make_X(log_df)
    print('正在训练/评估性能最佳的分类数...')
    best_clusters = train_check(X,args.maxclusters)
    print('最佳分类数为：' + str(best_clusters))
    
    print('正在分类数据...')
    log_df, kmeans = train(X, best_clusters, log_df)
    print("完成！") 

    log_df.to_csv(args.csv, sep=',', header=True, index=False)
    print("结果存入文件："+args.csv)
    joblib.dump(kmeans, args.pkl)
    print("kmean存入"+args.pkl)