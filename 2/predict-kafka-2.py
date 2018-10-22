# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 13:25:57 2018

@author: zhan

需要第三方包：python-kafka， scikit learn
"""

import cluster2 as cluster
from sklearn.externals import joblib
import argparse
from kafka import KafkaConsumer


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pkl',  required=True, help='path to pkl file')  
    parser.add_argument('-H', '--host', help='host ip for database')
    parser.add_argument('-P', '--port', help='port for database', default='8086')
    parser.add_argument('-u', '--user', help='user name for database', default='root')
    parser.add_argument('-w', '--password', help='password for database', default='root')    
    parser.add_argument('-s', '--source', help='broker that data is from,like kafka', default='localhost:9092')    
    parser.add_argument('-t', '--topic', required=True, help='topic from kafka')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _get_args()  
    model = joblib.load(args.pkl)
    scaler = model.scaler
    
 
    
    print("start")
    consumer = KafkaConsumer(args.topic, bootstrap_servers=[args.source])
    consumer = [
            '{"log":"bird: BGP: Unexpected connect from unknown address 10.252.21.153 (port 27467)\n","stream":"stdout","hostname":"core-cmhadoop5-2","container_log_file":"calico-node-lmcsz_kube-system_calico-node-d3fcbf92d8c09506a8493dfffeedd730543ec50b4e31564921444ef65ebd0a71"}'
            ]
    print("receiving")
    for msg in consumer:
        print (msg.value.decode())
        df = cluster.import_sample_json(msg.value.decode())
        df = cluster.extract_feature(df, cluster.TXT_REF)
        X = scaler.transform(df.loc[:, ['message.len', 'Levenshtein.ratio']])
        print("***********")
        labels = model.predict(X)
        df["label"] = labels
        
        try:
            print(df)
            if args.host != None:
                print("save to database")
                df["log"] = df["log"].str.replace('"', r'\"' )
                cluster.save_to_db(df, host=args.host, port=args.port, table_name="log_cluster",
                                   user=args.user, password=args.password)    
        except Exception as e:
            print(str(e))