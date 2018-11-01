# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 17:22:43 2018

@author: Zhang Han

do clustering by TF-IDF and Kmean of scikit learn 
"""
#-*- coding: utf8 -*-
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
import re,os

MAX_FEATURES = 500

def _get_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    parser_cluster = subparsers.add_parser('cluster', help='cluster sample')   
    parser_cluster.add_argument('-s', '--sample', required=True, help='path to load sample file from')
    parser_cluster.add_argument('-c', '--csv',  help='path to csv file as a result')
    parser_cluster.add_argument('-p', '--pkl', help='path to pkl file')
    parser_cluster.add_argument('-l', '--clusters', help='number of clusters',type=int,default=20)
    parser_cluster.add_argument('-H', '--dbhost', help='host ip for database', default='localhost')
    parser_cluster.add_argument('-P', '--dbport', help='port for database', default='8086')
    parser_cluster.add_argument('-u', '--dbuser', help='user name for database', default='root')
    parser_cluster.add_argument('-w', '--dbpassword', help='password for database', default='root')
    parser_cluster.add_argument('-d', '--draw', help='enable drawing', default='false', action="store_true")
    parser_cluster.set_defaults(func=cluster)
    
    parser_score = subparsers.add_parser('score', help='get score by diferrent k value')   
    parser_score.add_argument('-s', '--sample', required=True, help='path to load sample file from')
    parser_score.add_argument('-i', '--minclusters', help='min clusters you wan to compare, default 2', type=int, default=2)
    parser_score.add_argument('-a', '--maxclusters', help='max clusters you wan to compare, default 10', type=int, default=10)
    parser_score.add_argument('-d', '--draw', help='enable drawing', default='false', action="store_true")
    parser_score.set_defaults(func=score)
    
    parser_predict = subparsers.add_parser('predict', help='cluster sample')   
    parser_predict.add_argument('-p', '--pkl',  required=True, help='path to import pkl file')  
    parser_predict.add_argument('-H', '--dbhost', help='host ip for database')
    parser_predict.add_argument('-P', '--dbport', help='port for database', default='8086')
    parser_predict.add_argument('-u', '--dbuser', help='user name for database', default='root')
    parser_predict.add_argument('-w', '--dbpassword', help='password for database', default='root')    
    parser_predict.add_argument('-j', '--json', help='json string')
    parser_predict.add_argument('-c', '--csv',  help='path to csv file as a result')
    parser_predict.set_defaults(func=predict)
    
    parser_predict = subparsers.add_parser('predictserver', help='cluster sample')   
    parser_predict.add_argument('-p', '--pkl',  required=True, help='path to import pkl file')  
    parser_predict.add_argument('-H', '--dbhost', help='host ip for database')
    parser_predict.add_argument('-P', '--dbport', help='port for database', default='8086')
    parser_predict.add_argument('-u', '--dbuser', help='user name for database', default='root')
    parser_predict.add_argument('-w', '--dbpassword', help='password for database', default='root')    
    parser_predict.add_argument('-s', '--source', help='broker that data is from,like kafka', default='localhost:9092')    
    parser_predict.add_argument('-t', '--topic', help='topic from kafka')
    parser_predict.add_argument('-j', '--json', help='json string')
    parser_predict.set_defaults(func=predict_server)
    
    args = parser.parse_args()
    return args

def cluster(args):
    if args.csv == None: args.csv = args.sample + '.csv'
    if args.pkl == None: args.pkl = args.sample + '.pkl'
    print(args)
    #load and preprocess dataset
    print("Loading sample file")
    df = pd.read_json(args.sample, lines=True)
    data = cleaning(df['log'])
    # transform
    X,vectorizer = transform(data,max_features=MAX_FEATURES)
    #train
    print("Training")
    kmeans = train(X, k=args.clusters)
    print("Training finished")
    #save to file
    df = pd.DataFrame(data)
    df['label'] = kmeans.labels_
    try:
        df.to_csv(args.csv, sep=',', header=True, index=False)
        print("Saved cluster result to file {}".format(args.csv))
    except Exception as e:
        print("failed to save csv file! Error:",str(e))
    #save to database
    if args.dbhost!=None:
        try:
            print("Saving result to database")
            df["log"] = df["log"].str.replace('"', r'\"' )
            save_to_db(df, host=args.dbhost, port=args.dbport, table_name=os.path.basename(args.sample),
                       user=args.dbuser, password=args.dbpassword)
            print("Finished saving to database")
        except Exception as e:
            print("Failed to save database. Error:" + str(e))
    setattr(kmeans, "vectorizer", vectorizer)
    joblib.dump(kmeans, args.pkl)
    print("kmean存入"+args.pkl)

def score(args):
    '''测试选择最优参数'''
    df = pd.read_json(args.sample, lines=True)
    data = cleaning(df['log'])
    X,_ = transform(data,max_features=500)
    ks = []
    scores = []
    for i in range(args.minclusters, args.maxclusters):        
        km= train(X,k=i)
        print(i,km.inertia_)
        ks.append(i)
        scores.append(km.inertia_)
    if args.draw == True:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,4))
        plt.plot(ks,scores,label="inertia",color="red",linewidth=1)
        plt.xlabel("Feature")
        plt.ylabel("Error")
        plt.legend()
        plt.show()


def predict(args):    
    model = joblib.load(args.pkl)
    json = args.json
    df = pd.read_json(json, lines=True)
    data = cleaning(df['log'].copy(), drop_duplicates=False)
    X,_ = transform(data, max_features=MAX_FEATURES, vectorizer=model.vectorizer)
    labels = model.predict(X)
    label_log =  pd.DataFrame()
    label_log['label'] = labels
    label_log['log'] = df['log']
    for i in range(len(labels)):
        print("{} --- {}".format(labels[i],df['log'][i]))
    if args.csv!=None:
        label_log.to_csv(args.csv, sep=',', header=True, index=False)
    
def predict_server(args):
    from kafka import KafkaConsumer
    
    model = joblib.load(args.pkl)
    print("start")
    consumer = KafkaConsumer(args.topic, bootstrap_servers=[args.source])
#    consumer = [
#            '{"log":"bird: BGP: Unexpected connect from unknown address 10.252.21.153 (port 27467)\n","stream":"stdout","hostname":"core-cmhadoop5-2","container_log_file":"calico-node-lmcsz_kube-system_calico-node-d3fcbf92d8c09506a8493dfffeedd730543ec50b4e31564921444ef65ebd0a71"}'
#            ]
    print("receiving")
    for msg in consumer:
        json_str = msg.value.decode()
        print (json_str)
        df = pd.read_json(json_str, lines=True)
        data = cleaning(df['log'].copy(), drop_duplicates=False)
        X, _ = transform(data, max_features=MAX_FEATURES, vectorizer=model.vectorizer)
        print("***********")
        labels = model.predict(X)
        df["label"] = labels
        try:
            print(df)
            if args.dbhost != None:
                print("save to database")
                df["log"] = df["log"].str.replace('"', r'\"' )
                save_to_db(df, host=args.dbhost, port=args.dbport, table_name="label",
                                   user=args.dbuser, password=args.dbpassword)    
        except Exception as e:
            print(str(e))    

def cleaning(data, drop_duplicates=True):
    '''
    Param:
        data - pandas.core.series.Series
    Return:
        pandas.DataFrame type
    '''
    data.replace(re.compile("^\s+|\s+$"), "", inplace=True)
    data.replace(re.compile("\d+"), "", inplace=True)
    if drop_duplicates==True: 
        data = data.drop_duplicates()
        data.reset_index(inplace=True, drop=True)
    return data

def transform(data, max_features=500, vectorizer=None):
    if vectorizer==None:
        vectorizer = TfidfVectorizer(max_features=max_features, use_idf=True)
        X = vectorizer.fit_transform(data)
    else:
        X = vectorizer.transform(data)
    return X,vectorizer

def train(X,k=10):
    km = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=1, verbose=False)
    km.fit(X)
    return km

def save_to_db(log_df, user='root',password='root', host='localhost', port=8086, table_name="demo",batch_size=1000):
    from influxdb import InfluxDBClient
    client = InfluxDBClient(host, port, user, password, 'log_predict')
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

if __name__ == '__main__':
    args = _get_args()
    args.func(args)    

    
    