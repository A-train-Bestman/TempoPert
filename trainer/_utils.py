# -*- coding: utf-8 -*-
# @Author: Xiaoning Qi
# @Date:   2022-06-23 03:36:39
# @Last Modified by:   Xiaoning Qi
# @Last Modified time: 2024-03-21 21:28:04

from cgi import test
from random import shuffle
import sys
import anndata
import numpy as np
from scipy import sparse
from sklearn import preprocessing

from anndata import AnnData
from collections import defaultdict

def shuffle_adata(adata):
    """
    Shuffles the `adata`.
    """
    if sparse.issparse(adata.X):
        #adata.X: sparse matrix to array
        adata.X = adata.X.A

    ind_list = [i for i in range(adata.shape[0])]
    shuffle(ind_list)
    new_adata = adata[ind_list, :]
    return new_adata

def train_valid_test_analysi(adata: AnnData, split_key = 'cov_drug_dose_name_split',drug="Vorinostat"):
    '''
    Get train_valid_test dataset
    '''

    shuffled = shuffle_adata(adata)
    #train_index = adata.obs[adata.obs[split_key] == 'train'].index.tolist()
    #valid_index = adata.obs[adata.obs[split_key] == 'valid'].index.tolist()
    test_index=adata.obs[adata.obs.astype(str).apply(lambda x: x.str.contains(drug, case=False, na=False)).any(axis=1)]
    #test_index=adata.obs[adata.obs[split_key] == 'test']
    test_index = test_index[test_index[split_key] == 'test'].index.tolist()
    control_index = adata.obs[adata.obs['dose'] == 0.0].index.tolist()

    # if len(train_index) > 0:
    #     train_index = train_index + control_index
    #     train_adata = shuffled[train_index, :]
    # else:
    #     train_adata = None
    # if len(valid_index) > 0:
    #     valid_index = valid_index + control_index
    #     valid_adata = shuffled[valid_index, :]
    # else:
    #     valid_adata = None
    if len(test_index) > 0:
        test_index = test_index + control_index
        test_adata = shuffled[test_index, :]
    else:
        test_adata = None

    return None, None, test_adata

def train_valid_test_TranSiGen(adata: AnnData, split_key='cov_drug_dose_name_split'):
    '''
    Get train_valid_test dataset
    '''

    shuffled = shuffle_adata(adata)

    train_index=adata[adata.obs['dose'] == 10.0]
    train_index=train_index[train_index.obs['pert_time'] == 24.0]
    train_index = train_index.obs[train_index.obs[split_key] == 'train'].index.tolist()

    valid_index = adata[adata.obs['dose'] == 10.0]
    valid_index = valid_index[valid_index.obs['pert_time'] == 24.0]
    valid_index = valid_index.obs[valid_index.obs[split_key] == 'valid'].index.tolist()

    test_index = adata[adata.obs['dose'] == 10.0]
    test_index = test_index[test_index.obs['pert_time'] == 24.0]
    test_index = test_index.obs[test_index.obs[split_key] == 'test'].index.tolist()

    control_index = adata.obs[adata.obs['dose'] == 0.0].index.tolist()

    if len(train_index) > 0:
        train_index = train_index + control_index
        train_adata = shuffled[train_index, :]
    else:
        train_adata = None
    if len(valid_index) > 0:
        valid_index = valid_index + control_index
        valid_adata = shuffled[valid_index, :]
    else:
        valid_adata = None
    if len(test_index) > 0:
        test_index = test_index + control_index
        test_adata = shuffled[test_index, :]
    else:
        test_adata = None

    return train_adata, valid_adata, test_adata


def train_valid_test_PRnet(adata: AnnData, split_key='cov_drug_dose_name_split'):
    '''
    Get train_valid_test dataset
    '''

    shuffled = shuffle_adata(adata)

    #train_index=adata[adata.obs['dose'] == 10.0]
    train_index=adata[adata.obs['pert_time'].isin([24.0,6.0])]
    train_index = train_index.obs[train_index.obs[split_key] == 'train'].index.tolist()

    #valid_index = adata[adata.obs['dose'] == 10.0]
    valid_index = adata[adata.obs['pert_time'].isin([24.0,6.0])]
    valid_index = valid_index.obs[valid_index.obs[split_key] == 'valid'].index.tolist()

    #test_index = adata[adata.obs['dose'] == 10.0]
    test_index = adata[adata.obs['pert_time'].isin([24.0,6.0])]
    test_index = test_index.obs[test_index.obs[split_key] == 'test'].index.tolist()

    control_index = adata.obs[adata.obs['dose'] == 0.0].index.tolist()

    if len(train_index) > 0:
        train_index = train_index + control_index
        train_adata = shuffled[train_index, :]
    else:
        train_adata = None
    if len(valid_index) > 0:
        valid_index = valid_index + control_index
        valid_adata = shuffled[valid_index, :]
    else:
        valid_adata = None
    if len(test_index) > 0:
        test_index = test_index + control_index
        test_adata = shuffled[test_index, :]
    else:
        test_adata = None

    return train_adata, valid_adata, test_adata
def train_valid_test(adata: AnnData, split_key='cov_drug_dose_name_split'):
    '''
    Get train_valid_test dataset
    '''

    shuffled = shuffle_adata(adata)
    train_index = adata.obs[adata.obs[split_key] == 'train'].index.tolist()
    valid_index = adata.obs[adata.obs[split_key] == 'valid'].index.tolist()
    test_index = adata.obs[adata.obs[split_key] == 'test'].index.tolist()
    control_index = adata.obs[adata.obs['dose'] == 0.0].index.tolist()

    if len(train_index) > 0:
        train_index = train_index + control_index
        train_adata = shuffled[train_index, :]
    else:
        train_adata = None
    if len(valid_index) > 0:
        valid_index = valid_index + control_index
        valid_adata = shuffled[valid_index, :]
    else:
        valid_adata = None
    if len(test_index) > 0:
        test_index = test_index + control_index
        test_adata = shuffled[test_index, :]
    else:
        test_adata = None

    return train_adata, valid_adata, test_adata