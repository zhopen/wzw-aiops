# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 15:07:42 2018

@author: zhang han
"""
import cluster
from sklearn.externals import joblib
import argparse

def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sample', required=True, help='path to sample file')
    parser.add_argument('-p', '--pkl',  required=True, help='path to pkl file')    
    parser.add_argument('-c', '--csv',  help='path to csv file as a result')
    args = parser.parse_args()
    if args.csv == None: args.csv = args.sample + '.csv'
    return args

if __name__ == '__main__':
    args = _get_args()
    
    km = joblib.load(args.pkl)
    df = cluster.import_sample(args.sample)
    df = cluster.extract_feature(df, cluster.TXT_REF)
    X,df = cluster.make_X(df)
    labels = km.predict(X)
    df["label"] = labels
    df.to_csv(args.csv)
