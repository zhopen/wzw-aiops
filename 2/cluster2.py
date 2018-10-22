#-*- coding: utf8 -*-
import pandas as pd
import numpy as np
import Levenshtein
from sklearn.cluster import KMeans
#from sklearn.metrics import silhouette_score
from sklearn.externals import joblib
from sklearn import preprocessing
import sys
import re
import os
import argparse

def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sample', required=True, help='path to sample file')
    parser.add_argument('-c', '--csv',  help='path to csv file as a result')
    parser.add_argument('-p', '--pkl', help='path to pkl file')
    parser.add_argument('-m', '--maxclusters', help='max clusters',type=int,default=20)
    parser.add_argument('-H', '--host', help='host ip for database', default='localhost')
    parser.add_argument('-P', '--port', help='port for database', default='8086')
    parser.add_argument('-u', '--user', help='user name for database', default='root')
    parser.add_argument('-w', '--password', help='password for database', default='root')
    parser.add_argument('-d', '--draw', help='enable drawing', default='false', action="store_true")
    args = parser.parse_args()
    if args.csv == None: args.csv = args.sample + '.csv'
    if args.pkl == None: args.pkl = args.sample + '.pkl'
    return args


def draw(x,y, xlabel='x', ylabel='y',line=True, scatter=True, grid=True):
    import matplotlib.pyplot as plt
    plt.figure(figsize=[8,8])
    ax = plt.subplot(111)
    if scatter == True: ax.scatter(x, y, marker=",",s=2)
    if line == True: ax.plot(x, y)
    if grid == True: ax.grid()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()
    
def draw3d(x,y,z, xlabel='x', ylabel='y',zlabel='z'):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=[8,8])
    #ax = plt.subplot(111,projection='3d')
    ax = Axes3D(fig)
    ax.scatter(x, y, z, marker=".")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.show()
    
def import_sample_json(sample_file):
    df = pd.read_json(sample_file, lines=True)
    df["log"] = df['log'].str.strip()
    return df

def extract_feature(df, txt_ref):   
    '''抽取特征和构建特征矩阵：ratio，len'''
    df[u'log'].replace(re.compile("\d+"), "x", inplace=True)
    df = pd.DataFrame(df['log'].drop_duplicates())
    df = df.reset_index(drop=True)
    df['message.len'] = df['log'].apply(lambda x:len(x))
    df['Levenshtein.ratio'] = df['log'].apply(lambda x:Levenshtein.ratio(txt_ref,x))
    #df['string.sum'] = log.apply(str_sum)
    return df

def make_X(df):
    X_train = np.array(df.loc[:, ['message.len', 'Levenshtein.ratio']])
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_scaled  = scaler.transform(X_train)   
    df = pd.concat([df, pd.DataFrame(X_scaled,columns=['len_zs', 'ratio_zs'])], axis=1)
    return df, scaler
    

def train_check(X, max_clusters, isdraw=False):
    '''聚类分析（Kmean算法），然后通过silhouette_score算法评估出最佳的分类数'''
    k,distance, scores = range(2,max_clusters+1),[], []
    for n_clusters in k:
        cls = KMeans(n_clusters).fit(X)
        distance_sum = 0
        for i in range(n_clusters):
            group = cls.labels_ == i
            members = X[group,:]     
            distance_sum += (abs(members - cls.cluster_centers_[i])).sum() #同一标签的所有样本与质心的manhattan distance的和
        distance.append(distance_sum)
        score = cls.score(X) #silhouette_score(X,labels=cls.labels_, metric='euclidean')
        scores.append(score)
    best_clusters_idx = scores.index(max(scores))
    print("K values（最佳分类数量）:", str(k[best_clusters_idx]))
    print("Distance = ", str(distance[best_clusters_idx]))
    if isdraw==True:
        draw(k, distance, xlabel='k', ylabel='distance')
        draw(k, scores, xlabel='k', ylabel='score')
    return k[best_clusters_idx]

def train(X, n_clusters, df):
    kmeans = KMeans(n_clusters,max_iter=30,n_init=3).fit(X)
    df["label"] = kmeans.labels_
    return df, kmeans
    
def save_to_db(log_df, user='root',password='root', host='localhost', port=8086, table_name="demo",batch_size=1000):
    from influxdb import InfluxDBClient
    client = InfluxDBClient(host, port, user, password, 'log_label')
    #client.create_database('example')     
    for n in range(len(log_df)//batch_size+1):
        df = log_df[n*batch_size:(n+1)*batch_size]
        json_bodys = []
        for i in range(len(df)):
            json_body = {
                    "measurement": table_name,
                    "tags": {
                        "id": 0,
                        "label": 0
                    },
                    #"time": "2009-11-10T23:01:00Z",
                    "fields": {
                        "log":""
                    }
                }
            pos = n*batch_size+i
            json_body["tags"]["id"] = pos
            json_body["tags"]["label"] = df['label'][pos]
            json_body["fields"]["log"] = df['log'][pos].replace('"', '\"')
            json_bodys.append(json_body)
        client.write_points(json_bodys)     

TXT_REF = '[YYYY][INFO] abcdefghjklmnopqrstuvwxyz0123456789'
zscore_len =  0
zscore_ratio = 0
if __name__ == '__main__':
    args = _get_args()
    print(args)
    
    print('\n----------------------------------------------------------------')
    print("读入样本文件数据："+args.sample)
    
    log_df = import_sample_json(args.sample)
    print('提取特征。参考文本：' + TXT_REF)
    
    log_df = extract_feature(log_df, TXT_REF)
    log_df,scaler = make_X(log_df)
    
    
    print('正在训练/评估性能最佳的分类数...')
    best_clusters = train_check(np.array(log_df.loc[:, ['len_zs', 'ratio_zs']]),args.maxclusters, isdraw=args.draw)
    print('最佳分类数为：' + str(best_clusters))
    
    print('正在分类数据...')
    log_df, kmeans = train(np.array(log_df.loc[:, ['len_zs', 'ratio_zs']]), best_clusters, log_df)
    print("完成！") 

    log_df.to_csv(args.csv, sep=',', header=True, index=False)
    print("结果存入文件："+args.csv)
    
    if args.host!=None:
        try:
            print("save to database")
            log_df["log"] = log_df["log"].str.replace('"', r'\"' )
            save_to_db(log_df, host=args.host, port=args.port, table_name=os.path.basename(args.sample),
                       user=args.user, password=args.password)    
        except Exception as e:
            print("Failed to save database. \n" + str(e))

    setattr(kmeans, "scaler", scaler)
    joblib.dump(kmeans, args.pkl)
    print("kmean存入"+args.pkl)
    
