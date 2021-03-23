import pandas as pd
import numpy as np
import math
import pickle
import collections

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
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
import Birch_Bellwether as birch_bellwether

import metrics

import sys
import traceback
import warnings
warnings.filterwarnings("ignore")

# Birch Cluster Creator
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

def build_BIRCH(attr_df):
    cluster,cluster_tree,_ = cluster_driver(attr_df)
    return cluster,cluster_tree

def get_clusters(data_source):
    if platform.system() == 'Darwin' or platform.system() == 'Linux':
        _dir = data_source + '/'
    else:
        _dir = data_source + '\\'

    clusters = [(join(_dir, f)) for f in listdir(_dir) if Path(join(_dir, f)).is_dir()]
    return clusters

def norm(x,df):
    lo = df.min()
    hi = df.max()
    return (x - lo) / (hi - lo +0.00000001)

def dominate(_df,t,row_project_name,goals):
    wins = 0
    for i in range(_df.shape[0]):
        project_name = _df.iloc[i].name
        row = _df.iloc[i].tolist()
        if project_name != row_project_name:
            if dominationCompare(row, t,goals,_df):
                wins += 1
    return wins

def dominationCompare(other_row, t,goals,df):
    n = len(goals)
    weight = {'recall':1,'precision':1,'pf':-3.5}
    sum1, sum2 = 0,0
    for i in range(len(goals)):
        _df = df[goals[i]]
        w = weight[goals[i]]
        x = t[i]
        y = other_row[i]
        x = norm(x,_df)
        y = norm(y,_df)
        sum1 = sum1 - math.e**(w * (x-y)/n)
        sum2 = sum2 - math.e**(w * (y-x)/n)
    return sum1/n < sum2/n

def find_bellwether_level2(data_source,fold):
    goals = ['recall','precision','pf']
    clusters = get_clusters(data_source)
    for cluster in clusters:
        if cluster.rsplit('/',1)[1] == 'results' or cluster.rsplit('/',1)[1] == 'cdom_level1':
            continue
        projects_performance = {}
        for goal in goals:
            df = pd.read_csv(cluster + '/700_RF_bellwether_' + goal + '.csv')
            for row in range(df.shape[0]):
                j = df.iloc[row].values[1:]
                j_med = np.median(j)
                project_name = df.iloc[row].values[0]
                if project_name not in projects_performance.keys():
                    projects_performance[project_name] = {}
                projects_performance[project_name][goal] = j_med
        _df = pd.DataFrame.from_dict(projects_performance, orient = 'index')
        dom_score = []
        for row_id in range(_df.shape[0]):
            project_name = _df.iloc[row_id].name
            row = _df.iloc[row_id].tolist()
            wins = dominate(_df,row,project_name,goals)
            dom_score.append(wins)
        _df['wins'] = dom_score
        _df.to_csv(cluster + '/cdom_latest.csv')
    return projects_performance

def calculate_level_1_performance(data_source,clusters,path,fold):
    df_train = pd.read_pickle(data_source + '/train_data.pkl')
    cluster,cluster_tree = build_BIRCH(df_train)
    cluster_ids = []
    cluster_structure = {}
    size = {}
    for key in cluster_tree:
        if cluster_tree[key].depth != None:
            cluster_ids.append(key)
            if cluster_tree[key].depth not in cluster_structure.keys():
                cluster_structure[cluster_tree[key].depth] = {}
            cluster_structure[cluster_tree[key].depth][key] = cluster_tree[key].parent_id
            size[key] = cluster_tree[key].size
    goals = ['recall','precision','pf','pci_20','ifa']
    count = 0
    score = []
    score_med = []
    cluster_info = {}
    for cluster in clusters:
        if cluster.rsplit('/',1)[1] in ['results','cdom_level1']:
            continue
        df = pd.read_csv(cluster + '/cdom_latest.csv')
        counts = {}
        med_count = []
        c_dom = df.wins.values.tolist()
        best_project = df.iloc[c_dom.index(max(c_dom)),0]
        for goal in goals:
            goal_df = pd.read_csv(cluster + '/700_RF_bellwether_' + goal + '.csv')
            goal_df.rename(columns={'Unnamed: 0':'projects'},inplace=True)
            j = goal_df[goal_df['projects'] == best_project].values[0][1:]
            if goal == 'pci_20': # check number of projects >= 0.4 when goal is pci_20
                value = sum(i >= 0.40 for i in j)
            elif goal != 'pf': # check number of projects >= 0.67 when goal is other then pci_20 and pf
                value = sum(i >= 0.67 for i in j)
            else: # check number of projects <= 0.33 when goal is pf
                value = sum(i <= 0.33 for i in j)
            counts[goal] = value
        score_med.append([int(cluster.rsplit('/',1)[1]),goal_df.shape[0],
                          counts['recall'],
                          counts['precision'],
                          counts['pf'],
                          counts['pci_20'],
                          max(c_dom),
                          best_project])
    score_df = pd.DataFrame(score_med, columns = ['id','Total_projects','count_recall',
                                                  'count_precision','count_pf','count_pci_20',
                                                  'cdom_score','bellwether'])
    score_df = score_df.sort_values('id')
    score_df.to_csv(data_source + '/bellwether_cdom_2.csv')
    level_1_bellwethers = {}
    for cluster in cluster_structure[2].keys():
        print(cluster)
        # if cluster in [18,28]:
        #     continue
        if cluster_structure[2][cluster] not in level_1_bellwethers.keys():
            level_1_bellwethers[cluster_structure[2][cluster]] = []
        level_1_bellwethers[cluster_structure[2][cluster]].append(score_df[score_df['id'] == cluster].bellwether.values[0])
    score_med = []
    for key in  level_1_bellwethers.keys():
        sub_cluster_bellwethers = level_1_bellwethers[key]
        bell = birch_bellwether.Bellwether_Method(path,df_train)
        final_score = bell.bellwether(sub_cluster_bellwethers,sub_cluster_bellwethers,metric='all')

        if not os.path.exists(data_source + '/cdom_level1/'):
            os.makedirs(data_source + '/cdom_level1/')
        
        with open(data_source + '/cdom_level1/cluster_'  + str(key) + '_performance.pkl', 'wb') as handle:
            pickle.dump(final_score, handle, protocol=pickle.HIGHEST_PROTOCOL) 

def find_bellwether_level1(data_source,clusters,path,fold):
    df_train = pd.read_pickle(data_source + '/train_data.pkl')
    cluster,cluster_tree = build_BIRCH(df_train)
    cluster_ids = []
    cluster_structure = {}
    size = {}
    for key in cluster_tree:
        if cluster_tree[key].depth != None:
            cluster_ids.append(key)
            if cluster_tree[key].depth not in cluster_structure.keys():
                cluster_structure[cluster_tree[key].depth] = {}
            cluster_structure[cluster_tree[key].depth][key] = cluster_tree[key].parent_id
            size[key] = cluster_tree[key].size
    goals = ['recall','precision','pf']
    score_df = pd.read_csv(data_source + '/bellwether_cdom_2.csv')
    score_df.drop(labels = ['Unnamed: 0'], axis = 1 ,inplace = True)
    level_1_bellwethers = {}
    for cluster in cluster_structure[2].keys():
        if cluster_structure[2][cluster] not in level_1_bellwethers.keys():
            level_1_bellwethers[cluster_structure[2][cluster]] = []
        level_1_bellwethers[cluster_structure[2][cluster]].append(score_df[score_df['id'] == cluster].bellwether.values[0])
    for cluster in cluster_structure[1].keys():
        if cluster not in level_1_bellwethers.keys():
            level_1_bellwethers[cluster] = list(df_train.loc[cluster_tree[cluster].data_points].index)
    bell_df = {}
    for key in  level_1_bellwethers.keys():
        sub_cluster_bellwethers = level_1_bellwethers[key]
        try:
            final_score = pd.read_pickle(data_source + '/cdom_level1/cluster_'  + str(key) + '_performance.pkl')
        except:
            continue
        _results = {}
        for goal in goals:    
            for s_project in final_score.keys():
                if s_project not in _results.keys():
                    _results[s_project] = {}
                    _temp = []
                for d_projects in final_score[s_project].keys():
                    if goal == 'g':
                        _goal = 'g-score'
                    else:
                        _goal = goal
                    _score = np.median(final_score[s_project][d_projects][_goal])
                    _temp.append(np.median(final_score[s_project][d_projects][_goal]))
                if goal not in _results[s_project].keys():
                    _results[s_project][goal] = []
                _results[s_project][goal] = np.median(_temp)
        _df = pd.DataFrame.from_dict(_results, orient = 'index')
        dom_score = []
        for row_id in range(_df.shape[0]):
            project_name = _df.iloc[row_id].name
            row = _df.iloc[row_id].tolist()
            wins = dominate(_df,row,project_name,goals)
            dom_score.append(wins)
        _df['wins'] = dom_score
        c_dom = _df.wins.values.tolist()
        best_project = _df.index[c_dom.index(max(c_dom))]
        best_project_perf = _df.loc[best_project].values.tolist()
        best_project_perf.append(best_project)
        bell_df[key] = best_project_perf
    perf_df = pd.DataFrame.from_dict(bell_df, orient = 'index', columns = ['recall','precision','pf','cdom','bellwether'])    
    perf_df.to_csv(data_source + '/bellwether_cdom_1.csv')

def find_bellwether_level0(data_source,path,fold):
    df_train = pd.read_pickle(data_source + '/train_data.pkl')
    cluster,cluster_tree = build_BIRCH(df_train)
    cluster_ids = []
    cluster_structure = {}
    size = {}
    for key in cluster_tree:
        if cluster_tree[key].depth != None:
            cluster_ids.append(key)
            if cluster_tree[key].depth not in cluster_structure.keys():
                cluster_structure[cluster_tree[key].depth] = {}
            cluster_structure[cluster_tree[key].depth][key] = cluster_tree[key].parent_id
            size[key] = cluster_tree[key].size
    goals = ['recall','precision','pf']
    bell_df = {}
    score_df = pd.read_csv(data_source + '/bellwether_cdom_1.csv')
    score_df = score_df.rename(columns = {'Unnamed: 0':'id'})
    _cluster_bellwethers = score_df.bellwether.values.tolist()
    bell = birch_bellwether.Bellwether_Method(path,score_df)
    final_score = bell.bellwether(_cluster_bellwethers,_cluster_bellwethers,metric='all')
    _results = {}
    for goal in goals:    
        for s_project in final_score.keys():
            if s_project not in _results.keys():
                _results[s_project] = {}
                _temp = []
            for d_projects in final_score[s_project].keys():
                if goal == 'g':
                    _goal = 'g-score'
                else:
                    _goal = goal
                _score = np.median(final_score[s_project][d_projects][_goal])
                _temp.append(np.median(final_score[s_project][d_projects][_goal]))
            if goal not in _results[s_project].keys():
                _results[s_project][goal] = []
            _results[s_project][goal] = np.median(_temp)
    _df = pd.DataFrame.from_dict(_results, orient = 'index')
    dom_score = []
    for row_id in range(_df.shape[0]):
        project_name = _df.iloc[row_id].name
        row = _df.iloc[row_id].tolist()
        wins = dominate(_df,row,project_name,goals)
        dom_score.append(wins)
    _df['wins'] = dom_score
    print(_df)
    c_dom = _df.wins.values.tolist()
    best_project = _df.index[c_dom.index(max(c_dom))]
    best_project_perf = _df.loc[best_project].values.tolist()
    best_project_perf.append(best_project)
    bell_df[key] = best_project_perf
    perf_df = pd.DataFrame.from_dict(bell_df, orient = 'index', columns = ['recall','precision','pf','cdom','bellwether'])    
    perf_df.to_csv(data_source + '/bellwether_cdom_0.csv')


if __name__ == "__main__":
    for i in range(1):
        fold = str(i)
        path = '../data/700/merged_data'
        data_source = 'results/mixed_data/level_2/fold_' + fold
        if platform.system() == 'Darwin' or platform.system() == 'Linux':
            _dir = data_source + '/'
        else:
            _dir = data_source + '\\'

        clusters = [(join(_dir, f)) for f in listdir(_dir) if Path(join(_dir, f)).is_dir()]
        find_bellwether_level2(data_source,i)
        calculate_level_1_performance(data_source,clusters,path,i)
        find_bellwether_level1(data_source,clusters,path,i)
        find_bellwether_level0(data_source,path,i)