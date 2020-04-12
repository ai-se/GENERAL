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

import platform
from os import listdir
from os.path import isfile, join
from glob import glob
from pathlib import Path
import sys
import os
import copy
import traceback
from pathlib import Path


import matplotlib.pyplot as plt

import SMOTE
import CFS
import birch
import metrics.abcd

import metrices
import measures

import sys
import traceback
import warnings
warnings.filterwarnings("ignore")


def prepare_data(path):
    df = pd.read_csv(path)
    df = df.drop(labels = ['Host','Vcs','Project','File','PL','IssueTracking'],axis=1)
    df = df.dropna()
    df = df[['TLOC', 'TNF', 'TNC', 'TND', 'LOC', 'CL', 'NStmt', 'NFunc',
       'RCC', 'MNL', 'avg_WMC', 'max_WMC', 'total_WMC', 'avg_DIT', 'max_DIT',
       'total_DIT', 'avg_RFC', 'max_RFC', 'total_RFC', 'avg_NOC', 'max_NOC',
       'total_NOC', 'avg_CBO', 'max_CBO', 'total_CBO', 'avg_DIT.1',
       'max_DIT.1', 'total_DIT.1', 'avg_NIV', 'max_NIV', 'total_NIV',
       'avg_NIM', 'max_NIM', 'total_NIM', 'avg_NOM', 'max_NOM', 'total_NOM',
       'avg_NPBM', 'max_NPBM', 'total_NPBM', 'avg_NPM', 'max_NPM', 'total_NPM',
       'avg_NPRM', 'max_NPRM', 'total_NPRM', 'avg_CC', 'max_CC', 'total_CC',
       'avg_FANIN', 'max_FANIN', 'total_FANIN', 'avg_FANOUT', 'max_FANOUT',
       'total_FANOUT', 'NRev', 'NFix', 'avg_AddedLOC', 'max_AddedLOC',
       'total_AddedLOC', 'avg_DeletedLOC', 'max_DeletedLOC',
       'total_DeletedLOC', 'avg_ModifiedLOC', 'max_ModifiedLOC',
       'total_ModifiedLOC','Buggy']]
    return df

def get_features(df):
    fs = feature_selector.featureSelector()
    df,_feature_nums,features = fs.cfs_bfs(df)
    return df,features

def apply_cfs(df):
    y = df.Buggy.values
    X = df.drop(labels = ['Buggy'],axis = 1)
    X = X.values
    selected_cols = CFS.cfs(X,y)
    cols = df.columns[[selected_cols]].tolist()
    cols.append('Buggy')
    return df[cols],cols
    
def apply_smote(df):
    cols = df.columns
    smt = SMOTE.smote(df)
    df = smt.run()
    df.columns = cols
    return df

def load_data(path,target):
    df = pd.read_csv(path)
    if path == 'data/jm1.csv':
        df = df[~df.uniq_Op.str.contains("\?")]
    y = df[target]
    X = df.drop(labels = target, axis = 1)
    X = X.apply(pd.to_numeric)
    return X,y

# Cluster Driver
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

def get_predicted(cluster_data_loc,metrices_loc,fold,data_location,default_bellwether_loc):
    train_data = pd.read_pickle(cluster_data_loc + '/train_data.pkl')
    cluster,cluster_tree,max_depth = cluster_driver(train_data)
    t_df = pd.DataFrame()
    for project in train_data.index.values.tolist():
        _s_path = '/Users/suvodeepmajumder/Documents/AI4SE/bellwether_comminity/data/1385/converted/' + project
        s_df = prepare_data(_s_path)
        t_df = pd.concat([t_df,s_df])
        break
    t_df.reset_index(drop=True,inplace=True)
    d = {'buggy': True, 'clean': False}
    t_df['Buggy'] = t_df['Buggy'].map(d)
    t_df, g_s_cols = apply_cfs(t_df)
    t_df = apply_smote(t_df)
    train_y = t_df.Buggy
    train_X = t_df.drop(labels = ['Buggy'],axis = 1)
    clf_global = LogisticRegression()
    clf_global.fit(train_X,train_y)
    test_data = pd.read_pickle(cluster_data_loc + '/test_data.pkl')
    #print(test_data)
    test_projects = test_data.index.values.tolist()
    goals = ['recall','precision','pf','pci_20','ifa']
    levels = [2,1,0]
    results = {}
    bellwether_models = {}
    bellwether0_models = {}
    bellwether0_s_cols = {}
    bellwether_s_cols = {}
    self_model = {}
    self_model_test = {} 
    for level in levels:
        test_data = test_data
        predicted_cluster = cluster.predict(test_data,level)
        #print(level,predicted_cluster)
        for i in range(len(predicted_cluster)):
            try:
                F = {}
                _F = {}
                b_F = {}
                g_F = {}
                r_F = {}
                c_id = predicted_cluster[i]
                s_project_df = pd.read_csv(cluster_data_loc + '/bellwether_cdom_' + str(level) + '.csv')
                if level == 1:
                    s_project_df.rename(columns = {'Unnamed: 0':'id'},inplace = True)
                if level == 0:
                    s_project = s_project_df.bellwether.values[0]
                else:
                    s_project = s_project_df[s_project_df['id'] == predicted_cluster[i]].bellwether.values[0]
                if s_project not in bellwether_models.keys():
                    s_path = '/Users/suvodeepmajumder/Documents/AI4SE/bellwether_comminity/data/1385/converted/' + s_project
                    df = prepare_data(s_path)
                    df.reset_index(drop=True,inplace=True)
                    d = {'buggy': True, 'clean': False}
                    df['Buggy'] = df['Buggy'].map(d)
                    df, s_cols = apply_cfs(df)
                    bellwether_s_cols[s_project] = s_cols
                    df = apply_smote(df)
                    y = df.Buggy
                    X = df.drop(labels = ['Buggy'],axis = 1)
                    clf_bellwether = LogisticRegression()
                    clf_bellwether.fit(X,y)
                    bellwether_models[s_project] = clf_bellwether
                else:
                    clf_bellwether = bellwether_models[s_project]
                    s_cols = bellwether_s_cols[s_project]
                    
                b_s_project_df = pd.read_csv(default_bellwether_loc + '/cdom_bellwether.csv')
                b_s_project_df.columns = ['bellwether', 'recall', 'precision', 'pf', 'wins']
                b_s_project = b_s_project_df.bellwether.values[b_s_project_df.wins.idxmax()]
                if b_s_project not in bellwether0_models.keys():
                    b_s_path = '/Users/suvodeepmajumder/Documents/AI4SE/bellwether_comminity/data/1385/converted/' + b_s_project
                    b_df = prepare_data(b_s_path)
                    b_df.reset_index(drop=True,inplace=True)
                    d = {'buggy': True, 'clean': False}
                    b_df['Buggy'] = b_df['Buggy'].map(d)
                    b_df, b_s_cols = apply_cfs(b_df)
                    bellwether0_s_cols[b_s_project] = b_s_cols
                    b_df = apply_smote(b_df)
                    b_y = b_df.Buggy
                    b_X = b_df.drop(labels = ['Buggy'],axis = 1)
                    b_clf_bellwether = LogisticRegression()
                    b_clf_bellwether.fit(b_X,b_y)
                    bellwether0_models[b_s_project] = b_clf_bellwether
                else:
                    b_clf_bellwether = bellwether0_models[b_s_project]
                    b_s_cols = bellwether0_s_cols[b_s_project]

                d_project = test_projects[i]
                kf = StratifiedKFold(n_splits = 2)
                d_path = '/Users/suvodeepmajumder/Documents/AI4SE/bellwether_comminity/data/1385/converted/' + d_project
                test_df = prepare_data(d_path)
                test_df.reset_index(drop=True,inplace=True)
                d = {'buggy': True, 'clean': False}
                test_df['Buggy'] = test_df['Buggy'].map(d)
                #test_df, x_s_cols = apply_cfs(test_df)
                test_y = test_df.Buggy
                test_X = test_df.drop(labels = ['Buggy'],axis = 1)
                for train_index, test_index in kf.split(test_X,test_y):
                    X_train, X_test = test_X.iloc[train_index], test_X.iloc[test_index]
                    y_train, y_test = test_y[train_index], test_y[test_index]
                    x_test_df = pd.concat([X_train,y_train], axis = 1)
                    x_test_df = apply_smote(x_test_df)
                    y_train = x_test_df.Buggy
                    X_train = x_test_df.drop(labels = ['Buggy'],axis = 1)
                    clf_self = LogisticRegression()
                    clf_self.fit(X_train,y_train)
                    self_model[d_project] = clf_self
                    self_model_test[d_project] = [X_test,y_test]

                    _test_df = pd.concat(self_model_test[d_project], axis = 1)
                    _df_test_loc = _test_df.LOC
                    _test_df_1 = copy.deepcopy(_test_df[s_cols])
                    _test_df_2 = copy.deepcopy(_test_df)
                    _test_df_3 = copy.deepcopy(_test_df[b_s_cols])
                    _test_df_4 = copy.deepcopy(_test_df[g_s_cols])
                    _test_df_5 = copy.deepcopy(_test_df)

                    y_test = _test_df_1.Buggy
                    X_test = _test_df_1.drop(labels = ['Buggy'],axis = 1)
                    predicted_bellwether = clf_bellwether.predict(X_test)
                    abcd = metrices.measures(y_test,predicted_bellwether,_df_test_loc)
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

                    try:
                        y_test = _test_df_2.Buggy
                        X_test = _test_df_2.drop(labels = ['Buggy'],axis = 1)
                        predicted_self = clf_self.predict(X_test) 
                        abcd = metrices.measures(y_test,predicted_self,_df_test_loc)
                        if 'f1' not in _F.keys():
                            _F['f1'] = []
                            _F['precision'] = []
                            _F['recall'] = []
                            _F['g-score'] = []
                            _F['d2h'] = []
                            _F['pci_20'] = []
                            _F['ifa'] = []
                            _F['pd'] = []
                            _F['pf'] = []
                        _F['f1'].append(abcd.calculate_f1_score())
                        _F['precision'].append(abcd.calculate_precision())
                        _F['recall'].append(abcd.calculate_recall())
                        _F['g-score'].append(abcd.get_g_score())
                        _F['d2h'].append(abcd.calculate_d2h())
                        _F['pci_20'].append(abcd.get_pci_20())
                        try:
                            _F['ifa'].append(abcd.get_ifa_roc())
                        except:
                            _F['ifa'].append(0)
                        _F['pd'].append(abcd.get_pd())
                        _F['pf'].append(abcd.get_pf())
                    except:
                        _F['f1'].append(0)
                        _F['precision'].append(0)
                        _F['recall'].append(0)
                        _F['g-score'].append(0)
                        _F['d2h'].append(0)
                        _F['pci_20'].append(0)
                        _F['ifa'].append(0)
                        _F['pd'].append(0)
                        _F['pf'].append(0)

                    b_y_test = _test_df_3.Buggy
                    b_X_test = _test_df_3.drop(labels = ['Buggy'],axis = 1)
                    predicted_bell0 = b_clf_bellwether.predict(b_X_test) 
                    abcd = metrices.measures(b_y_test,predicted_bell0,_df_test_loc)
                    if 'f1' not in b_F.keys():
                        b_F['f1'] = []
                        b_F['precision'] = []
                        b_F['recall'] = []
                        b_F['g-score'] = []
                        b_F['d2h'] = []
                        b_F['pci_20'] = []
                        b_F['ifa'] = []
                        b_F['pd'] = []
                        b_F['pf'] = []
                    b_F['f1'].append(abcd.calculate_f1_score())
                    b_F['precision'].append(abcd.calculate_precision())
                    b_F['recall'].append(abcd.calculate_recall())
                    b_F['g-score'].append(abcd.get_g_score())
                    b_F['d2h'].append(abcd.calculate_d2h())
                    b_F['pci_20'].append(abcd.get_pci_20())
                    try:
                        b_F['ifa'].append(abcd.get_ifa_roc())
                    except:
                        b_F['ifa'].append(0)
                    b_F['pd'].append(abcd.get_pd())
                    b_F['pf'].append(abcd.get_pf())
                    
                    
                    g_y_test = _test_df_4.Buggy
                    g_X_test = _test_df_4.drop(labels = ['Buggy'],axis = 1)
                    predicted_global = clf_global.predict(g_X_test) 
                    abcd = metrices.measures(g_y_test,predicted_global,_df_test_loc)
                    if 'f1' not in g_F.keys():
                        g_F['f1'] = []
                        g_F['precision'] = []
                        g_F['recall'] = []
                        g_F['g-score'] = []
                        g_F['d2h'] = []
                        g_F['pci_20'] = []
                        g_F['ifa'] = []
                        g_F['pd'] = []
                        g_F['pf'] = []
                    g_F['f1'].append(abcd.calculate_f1_score())
                    g_F['precision'].append(abcd.calculate_precision())
                    g_F['recall'].append(abcd.calculate_recall())
                    g_F['g-score'].append(abcd.get_g_score())
                    g_F['d2h'].append(abcd.calculate_d2h())
                    g_F['pci_20'].append(abcd.get_pci_20())
                    try:
                        g_F['ifa'].append(abcd.get_ifa_roc())
                    except:
                        g_F['ifa'].append(0)
                    g_F['pd'].append(abcd.get_pd())
                    g_F['pf'].append(abcd.get_pf())
                    
                    
                    
                    r_y_test = _test_df_5.Buggy
                    r_X_test = _test_df_5.drop(labels = ['Buggy'],axis = 1)
                    _count_major = Counter(y_train)
                    _count_major = _count_major.most_common(1)[0][0]
                    predicted_random = [_count_major]*r_X_test.shape[0]
                    abcd = metrices.measures(r_y_test,predicted_random,_df_test_loc)
                    if 'f1' not in r_F.keys():
                        r_F['f1'] = []
                        r_F['precision'] = []
                        r_F['recall'] = []
                        r_F['g-score'] = []
                        r_F['d2h'] = []
                        r_F['pci_20'] = []
                        r_F['ifa'] = []
                        r_F['pd'] = []
                        r_F['pf'] = []
                    r_F['f1'].append(abcd.calculate_f1_score())
                    r_F['precision'].append(abcd.calculate_precision())
                    r_F['recall'].append(abcd.calculate_recall())
                    r_F['g-score'].append(abcd.get_g_score())
                    r_F['d2h'].append(abcd.calculate_d2h())
                    r_F['pci_20'].append(abcd.get_pci_20())
                    try:
                        r_F['ifa'].append(abcd.get_ifa_roc())
                    except:
                        r_F['ifa'].append(0)
                    r_F['pd'].append(abcd.get_pd())
                    r_F['pf'].append(abcd.get_pf())

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
                    results[_goal][d_project].append(np.median(b_F[_goal]))
                    results[_goal][d_project].append(np.median(_F[_goal]))
                    results[_goal][d_project].append(np.median(g_F[_goal])) 
                    results[_goal][d_project].append(np.median(r_F[_goal])) 
            except Exception as e:
                print(e)
                continue
    _cols = ['level2_bellwether','default_bellwether','self','global',
                'random','level1_bellwether','default_bellwether','self',
                'global','random','level0_bellwether','default_bellwether',
                'self','global','random']
    print(results)
    for key in results:
        df = pd.DataFrame.from_dict(results[key],orient='index',columns = _cols)
        if not Path(data_location).is_dir():
            os.makedirs(Path(data_location))
            df.to_csv(data_location + '/bellwether_' + key + '.csv')
    return results



if __name__ == "__main__":
    for i in range(1):
        fold = str(i)
        data_location = 'data/1385/Cluster_data/final_results/fold_' + fold
        cluster_data_loc = 'data/1385/Cluster_data/fold_' + fold
        metrices_loc = '/Users/suvodeepmajumder/Documents/AI4SE/bell/data/1385/converted'
        default_bellwether_loc = 'data/1385/Cluster_data/fold_' + str(fold) + '/0'
        results = get_predicted(cluster_data_loc,metrices_loc,fold,data_location,default_bellwether_loc)