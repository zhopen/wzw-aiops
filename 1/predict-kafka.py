# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 13:25:57 2018

@author: zhan

需要第三方包：python-kafka， scikit learn
"""

import cluster  
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


#model = joblib.load(r"C:/zhanghan/wzw/aiops/data/a.log.pkl")
#print("len_std=", model.len_std)
#print("len_mean=", model.len_mean)
#print("ratio_std=", model.ratio_std)    
#print("ratio_mean=", model.ratio_mean)

#
#log='{"log":"bird: BGP: Unexpected connect from unknown address 10.252.21.153 (port 27467)\n","stream":"stdout","hostname":"core-cmhadoop5-2","container_log_file":"calico-node-lmcsz_kube-system_calico-node-d3fcbf92d8c09506a8493dfffeedd730543ec50b4e31564921444ef65ebd0a71"}'
#log='{"log":"2018-06-01 01:00:05.229 [INFO][176] health.go 150: Overall health summary=&health.HealthReport{Live:true, Ready:true}\n","stream":"stdout","hostname":"core-cmhadoop5-2","container_log_file":"calico-node-lmcsz_kube-system_calico-node-d3fcbf92d8c09506a8493dfffeedd730543ec50b4e31564921444ef65ebd0a71"}'
#log='{"log":"2018-06-01 01:00:07.282 [INFO][176] int_dataplane.go 690: Applying dataplane updates\n","stream":"stdout","hostname":"core-cmhadoop5-2","container_log_file":"calico-node-lmcsz_kube-system_calico-node-d3fcbf92d8c09506a8493dfffeedd730543ec50b4e31564921444ef65ebd0a71"}'
#df = cluster.import_sample_json(log)
#print(df)
#df = cluster.extract_feature(df, cluster.TXT_REF)
#X,df = cluster.make_X(df, 
#                      len_mean=model.len_mean, 
#                      len_std=model.len_std,
#                      ratio_mean=model.ratio_mean, 
#                      ratio_std=model.ratio_std)
#print("***********")
#print(X)
#labels = model.predict(X)
#df["label"] = labels
#print(df)
    
#'123.206.41.161:9092'

if __name__ == '__main__':
    args = _get_args()  
    model = joblib.load(args.pkl)
    
    
    print("len_std=", model.len_std)
    print("len_mean=", model.len_mean)
    print("ratio_std=", model.ratio_std)    
    print("ratio_mean=", model.ratio_mean)    
    
    print("start")
    consumer = KafkaConsumer(args.topic, bootstrap_servers=[args.source])
    print("receiving")
    for msg in consumer:
        print (msg.value.decode())
        df = cluster.import_sample_json(msg.value.decode())
        df = cluster.extract_feature(df, cluster.TXT_REF)
        X,df = cluster.make_X(df, 
                              len_mean=model.len_mean, 
                              len_std=model.len_std,
                              ratio_mean=model.ratio_mean, 
                              ratio_std=model.ratio_std)
        print("***********")
        print(X)
        labels = model.predict(X)
        df["label"] = labels
        
        try:
            if args.host != None:
                print("save to database")
                df["log"] = df["log"].str.replace('"', r'\"' )
                cluster.save_to_db(df, host=args.host, port=args.port, table_name="log_cluster",
                                   user=args.user, password=args.password)    
            print(df)
        except Exception as e:
            print(str(e))