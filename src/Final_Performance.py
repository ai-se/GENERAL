import pandas as pd
import numpy as np
import math
import pickle
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

import random

import platform
from os import listdir
from os.path import isfile, join
from glob import glob
from pathlib import Path
import sys
import os
import copy
import traceback
import timeit



import matplotlib.pyplot as plt

import SMOTE
import CFS
import birch


from multiprocessing import Pool, cpu_count
from threading import Thread
from multiprocessing import Queue

import metrics

import sys
import traceback
import warnings
warnings.filterwarnings("ignore")

def prepare_data(project, metric):
    data_path = '../data/700/merged_data/' + project + '.csv'
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

def apply_cfs(df):
    y = df.Bugs.values
    X = df.drop(labels = ['Bugs'],axis = 1)
    X = X.values
    selected_cols = CFS.cfs(X,y)
    cols = df.columns[[selected_cols]].tolist()
    cols.append('Bugs')
    return df[cols],cols
    
def apply_smote(df):
    cols = df.columns
    smt = SMOTE.smote(df)
    df = smt.run()
    df.columns = cols
    return df

def cluster_driver(df,print_tree = True):
    X = df.apply(pd.to_numeric)
    cluster = birch.birch(branching_factor=20)
    #X.set_index('Project Name',inplace=True)
    cluster.fit(X)
    cluster_tree,max_depth = cluster.get_cluster_tree()
    #cluster_tree = cluster.model_adder(cluster_tree)
    if print_tree:
        cluster.show_clutser_tree()
    return cluster,cluster_tree,max_depth


def get_predicted_1(cluster_data_loc,fold,data_location,default_bellwether_loc):
    with open('results/attributes/projects_attributes_all.pkl', 'rb') as handle:
        cfs_data = pickle.load(handle)
    cfs_data_df = pd.DataFrame.from_dict(cfs_data, orient= 'index')
    train_data = pd.read_pickle(cluster_data_loc + '/train_data.pkl')
    cluster,cluster_tree,max_depth = cluster_driver(train_data)
    test_data = pd.read_pickle(cluster_data_loc + '/test_data.pkl')
    test_projects = test_data.index.values.tolist()
    goals = ['recall','precision','pf','pci_20','ifa']
    results = {}
    bellwether_models = {}
    bellwether0_models = {}
    bellwether0_s_cols = {}
    bellwether_s_cols = {}
    self_model = {}
    self_model_test = {}
    test_data = test_data
    predicted_cluster = cluster.predict(test_data,1)
    s_project_df = pd.read_csv(cluster_data_loc + '/bellwether_cdom_' + str(2) + '.csv')
    # s_project_df.rename(columns = {'Unnamed: 0':'id'},inplace = True)
    for i in range(len(predicted_cluster)):
        F = {}
        F_bell = {}
        c_id = predicted_cluster[i]
        cluster_bellwether = s_project_df[s_project_df['id'] == predicted_cluster[i]].bellwether.values[0]
        d_project = test_projects[i]
        kf = StratifiedKFold(n_splits = 2)
        # d_path = '/Users/suvodeepmajumder/Documents/AI4SE/bellwether_comminity/data/700/merged_data/' + d_project
        test_df = prepare_data(d_project, 'all')
        test_df.reset_index(drop=True,inplace=True)
        test_y = test_df.Bugs
        test_X = test_df.drop(labels = ['Bugs'],axis = 1)
        _df_test_loc = test_df['file_la'] + test_df['file_lt']
        s_projects = s_project_df.bellwether.values.tolist()
        s_projects.remove(cluster_bellwether)
        s_project_lists = [s_projects[random.randint(0,len(s_projects)-1)]]
        s_project_lists.append(cluster_bellwether)
        print(s_project_lists)
        for s_project in s_project_lists:
            if s_project not in bellwether_models.keys():
                # s_path = '/Users/suvodeepmajumder/Documents/AI4SE/bellwether_comminity/data/700/merged_data/' + s_project
                df = prepare_data(s_project, 'all')
                cols = df.columns
                df.reset_index(drop=True,inplace=True)
                # df, s_cols = apply_cfs(df)
                s_cols = []
                for i  in range(cfs_data_df.loc[s_project].shape[0]):
                    col = cfs_data_df.loc[s_project].values.tolist()[i]
                    if col == 1:
                        s_cols.append(cols[i])
                df = df[s_cols]
                bellwether_s_cols[s_project] = s_cols
                df = apply_smote(df)
                y = df.Bugs
                X = df.drop(labels = ['Bugs'],axis = 1)
                clf_bellwether = RandomForestClassifier()
                clf_bellwether.fit(X,y)
                bellwether_models[s_project] = clf_bellwether
            else:
                clf_bellwether = bellwether_models[s_project]
                s_cols = bellwether_s_cols[s_project]

            for train_index, test_index in kf.split(test_X,test_y):
                X_train, X_test = test_X.iloc[train_index], test_X.iloc[test_index]
                y_train, y_test = test_y[train_index], test_y[test_index]
                if s_project !=  cluster_bellwether:
                    try:
                        df_test_bell = copy.deepcopy(test_df[s_cols])
                        y_test = df_test_bell.Bugs
                        X_test = df_test_bell.drop(labels = ['Bugs'],axis = 1)
                        predicted_self = clf_bellwether.predict(X_test) 
                        abcd = metrics.measures(y_test,predicted_self,_df_test_loc)
                        if 'f1' not in F.keys():
                            F['f1'] = []
                            F['precision'] = []
                            F['recall'] = []
                            F['g-score'] = []
                            F['d2h'] = []
                            F['pci_20'] = []
                            F['ifa'] = []
                            F['pd'] = []
                            F['pf'] = []
                        F['f1'].append(abcd.calculate_f1_score())
                        F['precision'].append(abcd.calculate_precision())
                        F['recall'].append(abcd.calculate_recall())
                        F['g-score'].append(abcd.get_g_score())
                        F['d2h'].append(abcd.calculate_d2h())
                        F['pci_20'].append(abcd.get_pci_20())
                        try:
                            F['ifa'].append(abcd.get_ifa_roc())
                        except:
                            F['ifa'].append(0)
                        F['pd'].append(abcd.get_pd())
                        F['pf'].append(abcd.get_pf())
                    except:
                        F['f1'].append(0)
                        F['precision'].append(0)
                        F['recall'].append(0)
                        F['g-score'].append(0)
                        F['d2h'].append(0)
                        F['pci_20'].append(0)
                        F['ifa'].append(0)
                        F['pd'].append(0)
                        F['pf'].append(0)
                else:
                    try:
                        df_test_bell = copy.deepcopy(test_df[s_cols])
                        y_test = df_test_bell.Bugs
                        X_test = df_test_bell.drop(labels = ['Bugs'],axis = 1)
                        predicted_self = clf_bellwether.predict(X_test) 
                        abcd = metrics.measures(y_test,predicted_self,_df_test_loc)
                        if 'f1' not in F_bell.keys():
                            F_bell['f1'] = []
                            F_bell['precision'] = []
                            F_bell['recall'] = []
                            F_bell['g-score'] = []
                            F_bell['d2h'] = []
                            F_bell['pci_20'] = []
                            F_bell['ifa'] = []
                            F_bell['pd'] = []
                            F_bell['pf'] = []
                        F_bell['f1'].append(abcd.calculate_f1_score())
                        F_bell['precision'].append(abcd.calculate_precision())
                        F_bell['recall'].append(abcd.calculate_recall())
                        F_bell['g-score'].append(abcd.get_g_score())
                        F_bell['d2h'].append(abcd.calculate_d2h())
                        F_bell['pci_20'].append(abcd.get_pci_20())
                        try:
                            F_bell['ifa'].append(abcd.get_ifa_roc())
                        except:
                            F_bell['ifa'].append(0)
                        F_bell['pd'].append(abcd.get_pd())
                        F_bell['pf'].append(abcd.get_pf())
                    except Exception as e:
                        print(e)
                        F_bell['f1'].append(0)
                        F_bell['precision'].append(0)
                        F_bell['recall'].append(0)
                        F_bell['g-score'].append(0)
                        F_bell['d2h'].append(0)
                        F_bell['pci_20'].append(0)
                        F_bell['ifa'].append(0)
                        F_bell['pd'].append(0)
                        F_bell['pf'].append(0)
        for goal in goals:
            if goal == 'g':
                _goal = 'g-score'
            else:
                _goal = goal
            if _goal not in results.keys():
                results[_goal] = {}
            if d_project not in results[_goal].keys():
                results[_goal][d_project] = []
            results[_goal][d_project].append(np.median(F[_goal]))
            results[_goal][d_project].append(np.median(F_bell[_goal]))
    return results
            
        
    
    
    



if __name__ == "__main__":
    for i in range(1):
        fold = str(i)
        data_location = 'results/mixed_data/level_2/fold_' + fold
        cluster_data_loc = 'results/mixed_data/level_2/fold_' + fold
        default_bellwether_loc = 'results/mixed_data/default_bell/fold_' + fold
        results = get_predicted_1(cluster_data_loc,fold,data_location,default_bellwether_loc)
        with open(data_location + '/level_1.pkl', 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)