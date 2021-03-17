import os
from os import listdir
from os.path import isfile, join
import sys
import pandas as pd
import random
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

from sklearn.linear_model import LinearRegression

def prepare_data(project, metric):
    data_path = '../data/700/merged_data_original/' + project + '.csv'
    data_df = pd.read_csv(data_path)
    data_df.rename(columns = {'Unnamed: 0':'id'},inplace = True)

    for col in ['id', 'commit_hash', 'release']:
        if col in data_df.columns:
            data_df = data_df.drop([col], axis = 1)
    data_df = data_df.dropna()
    y = data_df.Bugs
    X = data_df.drop(['Bugs'],axis = 1)
    if metric == 'process':
        X = X[['file_la', 'file_ld', 'file_lt', 'file_age', 'file_ddev',
    'file_nuc', 'own', 'minor', 'file_ndev', 'file_ncomm', 'file_adev',
    'file_nadev', 'file_avg_nddev', 'file_avg_nadev', 'file_avg_ncomm',
    'file_ns', 'file_exp', 'file_sexp', 'file_rexp', 'file_nd', 'file_sctr']]
    elif metric == 'product':
        X = X.drop(['file_la', 'file_ld', 'file_lt', 'file_age', 'file_ddev',
    'file_nuc', 'own', 'minor', 'file_ndev', 'file_ncomm', 'file_adev',
    'file_nadev', 'file_avg_nddev', 'file_avg_nadev', 'file_avg_ncomm',
    'file_ns', 'file_exp', 'file_sexp', 'file_rexp', 'file_nd', 'file_sctr'],axis = 1)
    else:
        X = X
    df = X
    df['Bugs'] = y
    return df



def create_models(metric):
    meta_data = pd.read_pickle('results/TCA/tca.pkl')
    source_projects = meta_data.Source.unique()
    pseudo_source_projects = meta_data.target.unique()

    # print(source_projects)

    src_project_map = {}
    train_X = []
    for sp in source_projects:
        src_data = prepare_data(sp, metric)
        src_data_vector = src_data.median().values.tolist()
        src_project_map[sp] = src_project_map
        for tp in pseudo_source_projects:
            trg_data = prepare_data(tp, metric)
            trg_data_vector = trg_data.median().values.tolist()
            train_X.append(src_data_vector+trg_data_vector)

    train_y_f = meta_data.f.values.tolist()
    train_y_p = meta_data.pci_20.values.tolist()

    clf_f = LinearRegression()
    clf_p = LinearRegression()

    clf_f.fit(train_X,train_y_f)
    clf_p.fit(train_X,train_y_p)

    cluster_data_loc = 'results/mixed_data/level_2/fold_0'
    test_data = pd.read_pickle(cluster_data_loc + '/test_data.pkl')
    target_projects = test_data.index.values.tolist()[0:20]

    test_X = []
    test_map = []
    for sp in source_projects:
        src_data = prepare_data(sp, metric)
        src_data_vector = src_data.median().values.tolist()
        src_project_map[sp] = src_project_map
        for tp in target_projects:
            trg_data = prepare_data(tp, metric)
            trg_data_vector = trg_data.median().values.tolist()
            test_X.append(src_data_vector+trg_data_vector)
            test_map.append([sp, tp])
    
    predicted_f = clf_f.predict(test_X)
    predicted_p = clf_p.predict(test_X)

    for i in range(len(predicted_f)):
        test_map[i].append(predicted_f[i])
        test_map[i].append(predicted_p[i])

    final_result = test_map
    final_result_df = pd.DataFrame(final_result, columns=['src','trg', 'f', 'pci_20'])
    final_result_df.to_csv('results/TCA/tptl.csv')






if __name__ == "__main__":
    metric = 'all'
    create_models(metric = metric)