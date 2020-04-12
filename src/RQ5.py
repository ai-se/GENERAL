import pandas as pd
import CFS
import numpy as np
import math

import sys
import traceback
import warnings
import os
import copy
import platform
from os import listdir
from os.path import isfile, join
from glob import glob
from pathlib import Path

from birch_bellwether_v2 import bellwether
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import SMOTE
from sklearn.calibration import CalibratedClassifierCV

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


def apply_cfs(df):
    _cols = df.columns
    y = df.Buggy.values
    X = df.drop(labels = ['Buggy'],axis = 1)
    X = X.values
    selected_cols = CFS.cfs(X,y)
    fss = []
    cols = df.columns[[selected_cols]].tolist()
    cols.append('Buggy')
    for col in _cols:
        if col in cols:
            fss.append(1)
        else:
            fss.append(0)
    return df[cols],cols,fss
    
def apply_smote(df):
    cols = df.columns
    smt = SMOTE.smote(df)
    df = smt.run()
    df.columns = cols
    return df

def run_birch():
    i = 0
    all_attrs_df = pd.DataFrame()
    selected_attrs = pd.read_pickle('data/1385/projects/selected_attr.pkl')
    selected_attrs = pd.DataFrame.from_dict(selected_attrs,orient='index')
    path = '/Users/suvodeepmajumder/Documents/AI4SE/bell/data/1385/converted'
    meta_path = 'data/1385/projects/selected_attr.pkl'
    _data_store_path = 'data/1385/Exp_results/'
    attr_dict = pd.read_pickle(meta_path)
    attr_df = pd.DataFrame.from_dict(attr_dict,orient='index')
    attr_df_index = list(attr_df.index)
    kf = KFold(n_splits=10,random_state=24)
    data_store_path = _data_store_path
    data_store_path = data_store_path + 'fold_' + str(i) + '/'
    _attr_df_train = pd.read_pickle(data_store_path+'train_data.pkl')
    data_path = Path(data_store_path)
    if not data_path.is_dir():
        os.makedirs(data_path)
    bell = bellwether(path,_attr_df_train)
    cluster,cluster_tree = bell.build_BIRCH()
    cluster_ids = []
    for key in cluster_tree:
        if cluster_tree[key].depth == 2:
            #print(cluster_tree[key].parent_id)
            if cluster_tree[key].parent_id not in cluster_ids:
                cluster_ids.append(cluster_tree[key].parent_id)

    df_bells = pd.read_csv(data_store_path + 'bellwether_cdom_1.csv')
    bells = []
    print(cluster_ids)
    for bell_w in df_bells.bellwether.values.tolist():
        for key in cluster_ids:
            if cluster_tree[key].depth == 1:
                #print(cluster_tree[key].parent_id)
                if bell_w in cluster_tree[key].data_points:
                    bells.append(bell_w)
    df_bells.columns = ['cluster_ids','recall',  'precision', 'pf',  'cdom','bellwether']
    df_bells = df_bells[df_bells.bellwether.isin(bells)]
#     print(df_bells)
    sub_selected_attrs = selected_attrs[selected_attrs.index.isin(df_bells.bellwether.values.tolist())]
    all_attrs_df = pd.concat([all_attrs_df,sub_selected_attrs],axis = 0)
    return all_attrs_df,cluster_ids,df_bells

def check_metrics(all_attrs_df):
    path = '/Users/suvodeepmajumder/Documents/AI4SE/bell/data/1385/converted/'
    all_attrs_df.columns = ['TLOC', 'TNF', 'TNC', 'TND', 'LOC', 'CL', 'NStmt', 'NFunc',
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
        'total_ModifiedLOC','Buggy']
    coefs = {}
    all_coefs = {}
    actual_odds = {}
    for x in range(all_attrs_df.shape[0]):
        proj = all_attrs_df.iloc[x]
        s_cols = proj.where(proj == 1).dropna().index.values.tolist()
        data_path = path + all_attrs_df.index[x]
        df = prepare_data(data_path)
        df.reset_index(drop=True,inplace=True)
        d = {'buggy': True, 'clean': False}
        df['Buggy'] = df['Buggy'].map(d)
        df = df[s_cols]
        train_y = df.Buggy
        train_X = df.drop(labels = ['Buggy'],axis = 1)

        clf = LogisticRegression()
        clf.fit(train_X,train_y)
        columns = train_X.columns.values
        coef = clf.coef_[0]
        if all_attrs_df.index[x] not in coefs.keys():
            coefs[all_attrs_df.index[x]] = {}
        for i in range(len(columns)):
            coefs[all_attrs_df.index[x]][columns[i]] = math.exp(round(coef[i],2))
        for i in range(len(columns)):
            if columns[i] not in all_coefs.keys():
                all_coefs[columns[i]] = []
            all_coefs[columns[i]].append(round(coef[i],2))
        if all_attrs_df.index[x] not in actual_odds.keys():
            actual_odds[all_attrs_df.index[x]] = {}
        actual_odds[all_attrs_df.index[x]] = coefs
    return all_coefs,coefs

def get_predicted(cluster_data_loc,metrices_loc,fold,data_location):
    train_data = pd.read_pickle(cluster_data_loc + '/train_data.pkl')
    project = train_data.index.values.tolist()
    _s_path = '/Users/suvodeepmajumder/Documents/AI4SE/bell/data/1385/converted/' + project[0]
    s_df = prepare_data(_s_path)
    s_cols = s_df.columns
    train_data.columns = s_cols
    result = train_data.sum(axis = 0)
    return result.values.tolist(),s_cols

num_cluster = []
_max = {}
for i in range(1):
    fold = str(i)
    all_attrs_df,cluster_ids,df_bells = run_birch()
    num_cluster.append(df_bells.shape[0])
    all_coefs,odds = check_metrics(all_attrs_df)
    print(all_coefs)
    unq_key = {}
    for key in all_coefs.keys():
        if '_' in key:
            if key.split('_')[1] not in unq_key:
                if key.split('_')[1] not in unq_key.keys():
                    unq_key[key.split('_')[1]] = []
                unq_key[key.split('_')[1]].append(key.split('_')[1])
        else:
            if key not in unq_key.keys():
                unq_key[key] = []
            unq_key[key].append(key)
    print(len(unq_key))
    print(unq_key)
    break

final_list = []
for i in range(1):
    fold = str(i)
    data_location = '/Users/suvodeepmajumder/Documents/AI4SE/bell/src/data/1385/Exp_results/fold_' + fold
    cluster_data_loc = '/Users/suvodeepmajumder/Documents/AI4SE/bell/src/data/1385/Exp_results/fold_' + fold
    metrices_loc = '/Users/suvodeepmajumder/Documents/AI4SE/bell/data/1385/converted'
    results,s_cols = get_predicted(cluster_data_loc,metrices_loc,fold,data_location)
    final_list.append(results)
final_df = pd.DataFrame(final_list,columns=s_cols)


fig, ax = plt.subplots(figsize=(30, 10)) 
ax.set_ylim(0, 400)
x_pos = range(len(final_df.iloc[0].values.tolist())-1)
y = final_df.iloc[0].sort_values(ascending=False)
barlist = plt.bar(y.index.tolist()[1:], y.values.tolist()[1:],color = 'darkgray')
ax.set_xticks(x_pos, y.index.tolist()[1:])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for i in range(len(y.values.tolist()[1:])):
    v = y.values.tolist()[1:][i]
    z = y.values.tolist()[1:][i]
    ax.text(i-0.25,v+25 , str(int(round((v/627)*100,0))) + '%', color='blue', fontweight='bold',fontsize=16,rotation='vertical')
all_attrs_df = all_attrs_df[y.index]
all_attrs_df = all_attrs_df.drop('Buggy',axis = 1)
print(all_attrs_df)
for i in range(all_attrs_df.shape[0]):
    proj = all_attrs_df.iloc[i]
    print(all_attrs_df.index[i])
    for j in range(proj.shape[0]):
        if proj.index[j] in odds[all_attrs_df.index[i]].keys():
            if (odds[all_attrs_df.index[i]][proj.index[j]] < 0.79 or odds[all_attrs_df.index[i]][proj.index[j]] > 1.21):
                print(proj.index[j],odds[all_attrs_df.index[i]][proj.index[j]])
                barlist[j].set_color('red')
plt.xticks(fontsize=18,rotation=90)
plt.yticks(fontsize=18)
#plt.show()
plt.savefig('fss.pdf',dpi=600,bbox_inches='tight', pad_inches=0.3)