#!/usr/bin/python
# coding: utf-8

import logging

import os,sys

import inspect

import datetime as dt

import numpy as np
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (15, 5)

import seaborn as sns

import networkx as nx

from tqdm import tqdm


#machine learning libraries
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import KFold,train_test_split
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor

import xgboost as xgb
from xgboost import XGBRegressor

import lightgbm as lgb

from keras.wrappers.scikit_learn import KerasRegressor

from artgor_utils import train_model_regression,reduce_mem_usage

#cheminformatics library
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import rdmolops

from rdkit.Chem.AtomPairs import Pairs

import openbabel as ob
obConversion = ob.OBConversion()

from nmr_utils import plot_types
from qsprLib import buildXvalModel,removeLowVar,showAVGCorrelations

data_dir = r'./data/'
struct_dir = r'./data/structures/'

j_list = ['1JHC','1JHN','2JHC','2JHH','2JHN','3JHC','3JHH','3JHN']

descriptorList = ['MolWt', 'NumValenceElectrons', 'NumRotatableBonds', 'nrings', 'MaxEStateIndex', 'MinEStateIndex',
                  'MaxAbsEStateIndex', 'MinAbsEStateIndex', 'qed', 'MolWt.1', 'HeavyAtomMolWt', 'ExactMolWt'
                  , 'NumRadicalElectrons', 'MaxPartialCharge', 'MinPartialCharge',
                  'MaxAbsPartialCharge', 'MinAbsPartialCharge', 'FpDensityMorgan1', 'FpDensityMorgan2',
                  'FpDensityMorgan3', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v',
                  'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'HallKierAlpha', 'Ipc', 'Kappa1', 'Kappa2',
                  'Kappa3', 'LabuteASA', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13',
                  'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7',
                  'PEOE_VSA8', 'PEOE_VSA9', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5',
                  'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11',
                  'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7',
                  'SlogP_VSA8', 'SlogP_VSA9', 'TPSA', 'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2',
                  'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8',
                  'EState_VSA9', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4',
                  'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'FractionCSP3',
                  'HeavyAtomCount', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles',
                  'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings',
                  'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms',  'NumSaturatedCarbocycles',
                  'NumSaturatedHeterocycles', 'NumSaturatedRings', 'RingCount', 'MolLogP', 'MolMR']

descImportant = ['VSA_EState7', 'PEOE_VSA8', 'qed', 'BalabanJ', 'PEOE_VSA9', 'VSA_EState8', 'MinAbsEStateIndex', 'MinEStateIndex', 'VSA_EState5', 'VSA_EState4', 'MaxEStateIndex', 'Kappa3', 'Chi4n', 'SMR_VSA5', 'BertzCT', 'Ipc', 'MinPartialCharge', 'PEOE_VSA7', 'Chi3n', 'FpDensityMorgan3', 'PEOE_VSA10', 'MaxPartialCharge', 'SlogP_VSA2', 'MolLogP', 'VSA_EState6', 'Kappa2', 'Chi2n', 'MolMR', 'VSA_EState3', 'SlogP_VSA5', 'VSA_EState2', 'SMR_VSA1', 'FpDensityMorgan2', 'EState_VSA1', 'TPSA', 'VSA_EState1', 'FpDensityMorgan1', 'Kappa1',  'EState_VSA9', 'PEOE_VSA6', 'SlogP_VSA4', 'MaxAbsPartialCharge', 'SMR_VSA6', 'Chi1v', 'PEOE_VSA1', 'PEOE_VSA11', 'NumRotatableBonds', 'SMR_VSA10', 'SMR_VSA4', 'fr_bicyclic', 'Chi1', 'LabuteASA', 'HallKierAlpha', 'EState_VSA8', 'MolWt', 'MinAbsPartialCharge', 'Chi1n', 'PEOE_VSA2', 'Chi0', 'SlogP_VSA3', 'pairFP_5', 'SMR_VSA7', 'fr_aldehyde', 'SlogP_VSA6', 'PEOE_VSA12',  'FractionCSP3', 'VSA_EState9', 'NumSaturatedCarbocycles', 'NumAliphaticHeterocycles', 'EState_VSA2', 'NumSaturatedRings', 'NumSaturatedHeterocycles',  'SMR_VSA3', 'Chi0v',  'NumAliphaticCarbocycles', 'fr_epoxide', 'nrings', 'fr_ether', 'HeavyAtomMolWt', 'SlogP_VSA1', 'Chi0n']
descRDkitTop100 = ['y_0', 'dist_to_type_mean', 'dist_to_type_0_mean', 'x_0', 'dist_y', 'y_1', 'z_0', 'dist_z', 'dist_x', 'x_1', 'z_1', 'molecule_type_dist_mean', 'dist', 'VSA_EState7', 'qed', 'BalabanJ', 'PEOE_VSA8', 'PEOE_VSA9', 'dist_to_type_1_mean', 'VSA_EState8', 'MinEStateIndex', 'MinAbsEStateIndex', 'VSA_EState5', 'MaxEStateIndex', 'VSA_EState4', 'Chi4n', 'Kappa3', 'SMR_VSA5', 'Ipc', 'MinPartialCharge', 'BertzCT', 'PEOE_VSA10', 'MaxPartialCharge', 'PEOE_VSA7', 'SlogP_VSA2', 'Chi3n', 'FpDensityMorgan3', 'VSA_EState6', 'MolLogP', 'Kappa2', 'MolMR', 'Chi2n', 'VSA_EState2', 'VSA_EState3', 'SlogP_VSA5', 'type', 'SMR_VSA1', 'FpDensityMorgan2', 'TPSA', 'EState_VSA1', 'MaxAbsPartialCharge', 'VSA_EState1', 'FpDensityMorgan1', 'SMR_VSA10', 'Kappa1', 'EState_VSA9', 'PEOE_VSA6', 'SMR_VSA6', 'SlogP_VSA4', 'PEOE_VSA11', 'Chi1v', 'PEOE_VSA1', 'NumRotatableBonds', 'MinAbsPartialCharge', 'SMR_VSA4', 'MolWt', 'Chi1', 'HallKierAlpha', 'EState_VSA8', 'LabuteASA', 'PEOE_VSA2', 'SlogP_VSA3', 'SMR_VSA7', 'SlogP_VSA6', 'Chi1n', 'PEOE_VSA12', 'Chi0', 'FractionCSP3', 'atom_1', 'SMR_VSA3', 'NumAliphaticHeterocycles', 'VSA_EState9', 'NumSaturatedCarbocycles', 'EState_VSA2', 'HeavyAtomMolWt', 'NumSaturatedHeterocycles', 'NumSaturatedRings', 'NumAliphaticCarbocycles', 'Chi0v', 'nrings', 'SlogP_VSA1', 'SMR_VSA2', 'Chi0n', 'PEOE_VSA3', 'PEOE_VSA13', 'SMR_VSA9', 'SlogP_VSA7', 'NumRadicalElectrons', 'NHOHCount', 'EState_VSA3']
descRDkitTop80 = ['y_0', 'dist_to_type_mean', 'dist_to_type_0_mean', 'x_0', 'dist_y', 'y_1', 'z_0', 'dist_z', 'dist_x', 'x_1', 'z_1', 'molecule_type_dist_mean', 'dist', 'VSA_EState7', 'qed', 'BalabanJ', 'PEOE_VSA8', 'PEOE_VSA9', 'dist_to_type_1_mean', 'VSA_EState8', 'MinEStateIndex', 'MinAbsEStateIndex', 'VSA_EState5', 'MaxEStateIndex', 'VSA_EState4', 'Chi4n', 'Kappa3', 'SMR_VSA5', 'Ipc', 'MinPartialCharge', 'BertzCT', 'PEOE_VSA10', 'MaxPartialCharge', 'PEOE_VSA7', 'SlogP_VSA2', 'Chi3n', 'FpDensityMorgan3', 'VSA_EState6', 'MolLogP', 'Kappa2', 'MolMR', 'Chi2n', 'VSA_EState2', 'VSA_EState3', 'SlogP_VSA5', 'type', 'SMR_VSA1', 'FpDensityMorgan2', 'TPSA', 'EState_VSA1', 'MaxAbsPartialCharge', 'VSA_EState1', 'FpDensityMorgan1', 'SMR_VSA10', 'Kappa1', 'EState_VSA9', 'PEOE_VSA6', 'SMR_VSA6', 'SlogP_VSA4', 'PEOE_VSA11', 'Chi1v', 'PEOE_VSA1', 'NumRotatableBonds', 'MinAbsPartialCharge', 'SMR_VSA4', 'MolWt', 'Chi1', 'HallKierAlpha', 'EState_VSA8', 'LabuteASA', 'PEOE_VSA2', 'SlogP_VSA3', 'SMR_VSA7', 'SlogP_VSA6', 'Chi1n', 'PEOE_VSA12', 'Chi0', 'FractionCSP3', 'atom_1', 'SMR_VSA3']



def analyzeDataSet(plotHist=False, plotGraph=False):
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv(data_dir+r'test.csv')
    sub = pd.read_csv(data_dir+r'sample_submission.csv')
    print(train.head(20))
    print(f'There are {train.shape[0]} rows in train data.')
    print(f'There are {test.shape[0]} rows in test data.')
    print(f"There are {train['molecule_name'].nunique()} distinct molecules in train data.")
    print(f"There are {test['molecule_name'].nunique()} distinct molecules in test data.")
    print(f"There are {test['molecule_name'].nunique()+train['molecule_name'].nunique()} distinct molecules in data.")
    print(f"There are {train['atom_index_0'].nunique()} unique atoms.")
    print(f"There are {train['type'].nunique()} unique types.")
    train.describe()

    if plotHist:
        fig, ax = plt.subplots(figsize=(18, 8))
        plt.subplot(1, 2, 1);
        plt.hist(train['scalar_coupling_constant'], bins=20);
        plt.title('Basic scalar_coupling_constant histogram');
        plt.subplot(1, 2, 2);
        sns.violinplot(x='type', y='scalar_coupling_constant', data=train);
        plt.title('Violinplot of scalar_coupling_constant by type');

    if plotGraph:
        fig, ax = plt.subplots(figsize=(20, 12))
        for i, t in enumerate(train['type'].unique()):
            print(t)
            train_type = train.loc[train['type'] == t]
            print(train_type.shape)
            print(train_type.head(20))
            G = nx.from_pandas_edgelist(train_type, 'atom_index_0', 'atom_index_1', ['scalar_coupling_constant'])
            print(G)
            print(G.nodes)
            print(G.edges)

            nx.draw(G, with_labels=True)
            plt.title(f'Graph for type {t}')

    max_rows = 3
    mol_max = 9
    for i, n in enumerate(train['molecule_name'].unique()):
        if i == mol_max: break
        print(n)
        struc_str = getStructureFromName(n)
        print(struc_str)
        el_arr, struc_arr = getArrayFromStructure(struc_str)
        print(struc_arr)
        train_mol = train.loc[train['molecule_name'] == n]
        G = nx.from_pandas_edgelist(train_mol, 'atom_index_0', 'atom_index_1', ['scalar_coupling_constant'])
        print(G.nodes)
        print(G.edges)
        print(train_mol.shape)
        print(train_mol.head(20))
        plt.subplot(max_rows, mol_max/max_rows, i + 1);
        nx.draw(G, with_labels=True)
        plt.title(f'Graph for molecule {n}')

    plt.show()

def getArrayFromStructure(struc_str):
    natoms = int(struc_str[0])
    coords = np.zeros((natoms, 3))
    elements = ['Q']*natoms
    if natoms!=len(struc_str[2:]):
        sys.std.err(f"WARNING: xyz files {struc_str} seem corrupt!")
    for i,line in enumerate(struc_str[2:]):
        elements[i] = line.split()[0]
        coords[i,:] = line.split()[1:]
    return(elements,coords)

def getStructureFromName(molname):
    xyz_path = struct_dir+molname+'.xyz'
    struc_str = []
    if os.path.isfile(xyz_path):
        with open(xyz_path) as f:
            for line in f.readlines():
                struc_str.append(line.strip())

    return(struc_str)

def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    """
    Fast metric computation for this competition: https://www.kaggle.com/c/champs-scalar-coupling
    Code is from this kernel: https://www.kaggle.com/uberkinder/efficient-metric
    """
    maes = (y_true-y_pred).abs().groupby(types).mean()
    return np.log(maes.map(lambda x: max(x, floor))).mean()

def map_atom_info(df, structures, atom_idx):
    df = pd.merge(df, structures, how='left',
                  left_on=['molecule_name', f'atom_index_{atom_idx}'],
                  right_on=['molecule_name', 'atom_index'])

    df = df.drop('atom_index', axis=1)
    df = df.rename(columns={'atom': f'atom_{atom_idx}',
                            'x': f'x_{atom_idx}',
                            'y': f'y_{atom_idx}',
                            'z': f'z_{atom_idx}'})
    return df

def create_brute_force_features_kernel(df):
    df['molecule_couples'] = df.groupby('molecule_name')['id'].transform('count')
    df['molecule_dist_mean'] = df.groupby('molecule_name')['dist'].transform('mean')
    df['molecule_dist_min'] = df.groupby('molecule_name')['dist'].transform('min')
    df['molecule_dist_max'] = df.groupby('molecule_name')['dist'].transform('max')
    df['atom_0_couples_count'] = df.groupby(['molecule_name', 'atom_index_0'])['id'].transform('count')
    df['atom_1_couples_count'] = df.groupby(['molecule_name', 'atom_index_1'])['id'].transform('count')

    df[f'molecule_atom_index_0_x_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['x_1'].transform('std')
    df[f'molecule_atom_index_0_y_1_mean'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('mean')
    df[f'molecule_atom_index_0_y_1_mean_diff'] = df[f'molecule_atom_index_0_y_1_mean'] - df['y_1']
    df[f'molecule_atom_index_0_y_1_mean_div'] = df[f'molecule_atom_index_0_y_1_mean'] / df['y_1']
    df[f'molecule_atom_index_0_y_1_max'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('max')
    df[f'molecule_atom_index_0_y_1_max_diff'] = df[f'molecule_atom_index_0_y_1_max'] - df['y_1']
    df[f'molecule_atom_index_0_y_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('std')
    df[f'molecule_atom_index_0_z_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['z_1'].transform('std')
    df[f'molecule_atom_index_0_dist_mean'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('mean')
    df[f'molecule_atom_index_0_dist_mean_diff'] = df[f'molecule_atom_index_0_dist_mean'] - df['dist']
    df[f'molecule_atom_index_0_dist_mean_div'] = df[f'molecule_atom_index_0_dist_mean'] / df['dist']
    df[f'molecule_atom_index_0_dist_max'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('max')
    df[f'molecule_atom_index_0_dist_max_diff'] = df[f'molecule_atom_index_0_dist_max'] - df['dist']
    df[f'molecule_atom_index_0_dist_max_div'] = df[f'molecule_atom_index_0_dist_max'] / df['dist']
    df[f'molecule_atom_index_0_dist_min'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('min')
    df[f'molecule_atom_index_0_dist_min_diff'] = df[f'molecule_atom_index_0_dist_min'] - df['dist']
    df[f'molecule_atom_index_0_dist_min_div'] = df[f'molecule_atom_index_0_dist_min'] / df['dist']
    df[f'molecule_atom_index_0_dist_std'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('std')
    df[f'molecule_atom_index_0_dist_std_diff'] = df[f'molecule_atom_index_0_dist_std'] - df['dist']
    df[f'molecule_atom_index_0_dist_std_div'] = df[f'molecule_atom_index_0_dist_std'] / df['dist']
    df[f'molecule_atom_index_1_dist_mean'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('mean')
    df[f'molecule_atom_index_1_dist_mean_diff'] = df[f'molecule_atom_index_1_dist_mean'] - df['dist']
    df[f'molecule_atom_index_1_dist_mean_div'] = df[f'molecule_atom_index_1_dist_mean'] / df['dist']
    df[f'molecule_atom_index_1_dist_max'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('max')
    df[f'molecule_atom_index_1_dist_max_diff'] = df[f'molecule_atom_index_1_dist_max'] - df['dist']
    df[f'molecule_atom_index_1_dist_max_div'] = df[f'molecule_atom_index_1_dist_max'] / df['dist']
    df[f'molecule_atom_index_1_dist_min'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('min')
    df[f'molecule_atom_index_1_dist_min_diff'] = df[f'molecule_atom_index_1_dist_min'] - df['dist']
    df[f'molecule_atom_index_1_dist_min_div'] = df[f'molecule_atom_index_1_dist_min'] / df['dist']
    df[f'molecule_atom_index_1_dist_std'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('std')
    df[f'molecule_atom_index_1_dist_std_diff'] = df[f'molecule_atom_index_1_dist_std'] - df['dist']
    df[f'molecule_atom_index_1_dist_std_div'] = df[f'molecule_atom_index_1_dist_std'] / df['dist']
    df[f'molecule_atom_1_dist_mean'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('mean')
    df[f'molecule_atom_1_dist_min'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('min')
    df[f'molecule_atom_1_dist_min_diff'] = df[f'molecule_atom_1_dist_min'] - df['dist']
    df[f'molecule_atom_1_dist_min_div'] = df[f'molecule_atom_1_dist_min'] / df['dist']
    df[f'molecule_atom_1_dist_std'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('std')
    df[f'molecule_atom_1_dist_std_diff'] = df[f'molecule_atom_1_dist_std'] - df['dist']
    df[f'molecule_type_0_dist_std'] = df.groupby(['molecule_name', 'type_0'])['dist'].transform('std')
    df[f'molecule_type_0_dist_std_diff'] = df[f'molecule_type_0_dist_std'] - df['dist']
    df[f'molecule_type_dist_mean'] = df.groupby(['molecule_name', 'type'])['dist'].transform('mean')
    df[f'molecule_type_dist_mean_diff'] = df[f'molecule_type_dist_mean'] - df['dist']
    df[f'molecule_type_dist_mean_div'] = df[f'molecule_type_dist_mean'] / df['dist']
    df[f'molecule_type_dist_max'] = df.groupby(['molecule_name', 'type'])['dist'].transform('max')
    df[f'molecule_type_dist_min'] = df.groupby(['molecule_name', 'type'])['dist'].transform('min')
    df[f'molecule_type_dist_std'] = df.groupby(['molecule_name', 'type'])['dist'].transform('std')
    df[f'molecule_type_dist_std_diff'] = df[f'molecule_type_dist_std'] - df['dist']

    df = reduce_mem_usage(df)
    return df

def create_brute_force_features(df):
    #https: // www.kaggle.com / artgor / brute - force - feature - engineering
    good_columns = [
        'molecule_atom_index_0_dist_min',
        'molecule_atom_index_0_dist_max',
        'molecule_atom_index_1_dist_min',
        'molecule_atom_index_0_dist_mean',
        'molecule_atom_index_0_dist_std',
        'dist',
        'molecule_atom_index_1_dist_std',
        'molecule_atom_index_1_dist_max',
        'molecule_atom_index_1_dist_mean',
        'molecule_atom_index_0_dist_max_diff',
        'molecule_atom_index_0_dist_max_div',
        'molecule_atom_index_0_dist_std_diff',
        'molecule_atom_index_0_dist_std_div',
        'atom_0_couples_count',
        'molecule_atom_index_0_dist_min_div',
        'molecule_atom_index_1_dist_std_diff',
        'molecule_atom_index_0_dist_mean_div',
        'atom_1_couples_count',
        'molecule_atom_index_0_dist_mean_diff',
        'molecule_couples',
        'atom_index_1',
        'molecule_dist_mean',
        'molecule_atom_index_1_dist_max_diff',
        'molecule_atom_index_0_y_1_std',
        'molecule_atom_index_1_dist_mean_diff',
        'molecule_atom_index_1_dist_std_div',
        'molecule_atom_index_1_dist_mean_div',
        'molecule_atom_index_1_dist_min_diff',
        'molecule_atom_index_1_dist_min_div',
        'molecule_atom_index_1_dist_max_div',
        'molecule_atom_index_0_z_1_std',
        'y_0',
        'molecule_type_dist_std_diff',
        'molecule_atom_1_dist_min_diff',
        'molecule_atom_index_0_x_1_std',
        'molecule_dist_min',
        'molecule_atom_index_0_dist_min_diff',
        'molecule_atom_index_0_y_1_mean_diff',
        'molecule_type_dist_min',
        'molecule_atom_1_dist_min_div',
        'atom_index_0',
        'molecule_dist_max',
        'molecule_atom_1_dist_std_diff',
        'molecule_type_dist_max',
        'molecule_atom_index_0_y_1_max_diff',
        'molecule_type_0_dist_std_diff',
        'molecule_type_dist_mean_diff',
        'molecule_atom_1_dist_mean',
        'molecule_atom_index_0_y_1_mean_div',
        'molecule_type_dist_mean_div',
        'type']

    print(len(good_columns))
    df['molecule_couples'] = df.groupby('molecule_name')['id'].transform('count')
    df['molecule_dist_mean'] = df.groupby('molecule_name')['dist'].transform('mean')
    df['molecule_dist_min'] = df.groupby('molecule_name')['dist'].transform('min')
    df['molecule_dist_max'] = df.groupby('molecule_name')['dist'].transform('max')
    df['molecule_dist_std'] = df.groupby('molecule_name')['dist'].transform('std')
    df['atom_0_couples_count'] = df.groupby(['molecule_name', 'atom_index_0'])['id'].transform('count')
    df['atom_1_couples_count'] = df.groupby(['molecule_name', 'atom_index_1'])['id'].transform('count')

    #num_cols = ['x_1', 'y_1', 'z_1', 'dist', 'dist_x', 'dist_y', 'dist_z'] #remove unphysical features
    num_cols = ['dist']
    #cat_cols = ['atom_index_0', 'atom_index_1', 'type', 'atom_1', 'type_0'] #remove unphysical features
    cat_cols = ['type', 'atom_1', 'type_0']

    aggs = ['mean', 'max', 'std', 'min']
    for col in cat_cols:
        df[f'molecule_{col}_count'] = df.groupby('molecule_name')[col].transform('count')

    for cat_col in tqdm(cat_cols):
        for num_col in num_cols:
            for agg in aggs:
                f_name = f'molecule_{cat_col}_{num_col}_{agg}'
                if f_name in good_columns:
                    df[f_name] = df.groupby(['molecule_name', cat_col])[num_col].transform(agg)
                    df[f'{f_name}_diff'] = df[f'molecule_{cat_col}_{num_col}_{agg}'] - df[num_col]
                    df[f'{f_name}_div'] = df[f'molecule_{cat_col}_{num_col}_{agg}'] / df[num_col]

    df = reduce_mem_usage(df)
    return df


def map_rdkit_features(df, moldata):
    df = pd.merge(df, moldata, how='left',
                  left_on=['molecule_name'],
                  right_on=['name'])
    # drop those for now
    drop_cols = ['name', 'smiles']
    df.drop(drop_cols, axis=1, inplace=True)
    moldata.drop(drop_cols, axis=1, inplace=True)
    print(df.head())
    return(df,moldata.columns)


def getAtomicFeatures(row):
    #https://www.kaggle.com/c/champs-scalar-coupling/discussion/93972#latest-546673
    mol_str = row['molblock']
    mol = Chem.MolFromMolBlock(mol_str, removeHs=False)
    if mol is not None:
        i = row['atom_index_0']
        j = row['atom_index_1']
        a = mol.GetAtomWithIdx(i)
        b = mol.GetAtomWithIdx(j)
        atna = a.GetAtomicNum()
        atnb = b.GetAtomicNum()
        nbondsa = len(a.GetBonds())
        nbondsb = len(b.GetBonds())
        #ndegreea = a.GetDegree()
        #ndegreeb = b.GetDegree()
        hybra = a.GetHybridization()
        hybrb = b.GetHybridization()
        implVala = a.GetExplicitValence()
        implValb = b.GetExplicitValence()
        aroma = a.GetIsAromatic()
        aromb = b.GetIsAromatic()
        numHa = a.GetNumImplicitHs()
        numHb = b.GetNumImplicitHs()
        inRinga = a.IsInRing()
        inRingb = b.IsInRing()
    else:
        atna = 0
        atnb = 0
        nbondsa = 0
        nbondsb = 0
        #ndegreea = 0
        #ndegreeb = 0
        hybra = 0
        hybrb = 0
        implVala = 0
        implValb = 0
        aroma = False
        aromb = False
        numHa = 0
        numHb = 0
        inRinga = False
        inRingb = False

    res = pd.Series({'atna':atna,'atnb':atnb,'nbondsa':nbondsa,'nbondsb':nbondsb, 'hybra':hybra,'hybrb':hybrb,
                     'implVala':implVala,'implValb':implValb,'aroma':aroma,'aromb':aromb,'numHa':numHa,'numHb':numHb,
                     'inRinga':inRinga, 'inRingb': inRingb})
    return (res)


def getMorganFP(row):
    radius_at1 = 1
    radius_at2 = 0 #otherwise too many descriptors
    radius_max = max(radius_at1,radius_at2)
    mol_str = row['molblock']
    mol = Chem.MolFromMolBlock(mol_str,removeHs=False)
    if mol is None:
        return(0)
    #print(row['molecule_name'])
    #print(mol_str)
    #print(Chem.MolToSmiles(Chem.RemoveHs(mol)))
    i = row['atom_index_0']
    j = row['atom_index_1']
    #at0 = mol.GetAtomWithIdx(i)
    #at1 = mol.GetAtomWithIdx(j)
    bi = {}
    fp = rdMolDescriptors.GetHashedMorganFingerprint(mol, radius=radius_max, fromAtoms=[i,j], bitInfo=bi)
    bits_atom1 = np.zeros((radius_max+1,))
    bits_atom2 = np.zeros((radius_max+1,))
    for bit,values in bi.items():
        for val in values:
            idx, radius = val
            if idx==i:
                bits_atom1[radius] =bit
            if idx==j:
                bits_atom2[radius] = bit

    return(pd.Series(np.hstack((bits_atom1[radius_at1],bits_atom2[radius_at2]))))


def getPairFP_old(row):
    mol_str = row['molblock']
    mol = Chem.MolFromMolBlock(mol_str,removeHs=False)
    if mol is None:
        return(0)
    #print(row['molecule_name'])
    #print(mol_str)
    #print(Chem.MolToSmiles(Chem.RemoveHs(mol)))
    i = row['atom_index_0']
    j = row['atom_index_1']
    at0 = mol.GetAtomWithIdx(i)
    at1 = mol.GetAtomWithIdx(j)
    dist = rdmolops.GetShortestPath(mol,i,j)
    dist = len(dist)
    score = Pairs.pyScorePair(at0, at1, dist)
    #print(score)
    #print(Pairs.ExplainPairScore(score))
    c0 = Pairs.Utils.GetAtomCode(at0)
    c1 = Pairs.Utils.GetAtomCode(at1)
    #dm = Chem.Get3DDistanceMatrix(mol)
    #bnd_dist = dm[i, j]
    # if bnd_at1 == bnd_at2 and bnd_at2=='C':
    #     print(mol_str)
    #     print(Chem.MolToSmiles(mol))
    #     print('at1:'+bnd_at1)
    #     print('at2:'+bnd_at2)
    #
    #     print('dist:'+str(bnd_dist))
    #
    #     input()

    # bnd = mol.GetBondBetweenAtoms(i, j)
    # if bnd is not None:
    #     bnd_type = int(bnd.GetBondTypeAsDouble())
    #     if bnd.GetIsAromatic():
    #         bnd_type = 4
    # else:
    #     bnd_type = 0
    #
    # print('bnd_type:' + str(bnd_type))

    return(score)

def createSDFileViaOpenBabel(struct_dir ='./data/structures', ext='.xyz', outfile='nmr.sdf'):
    """
    Use open babel to convert xyz to sdf format, could also be done more efficiently at the command line

    """
    from os import listdir
    from os.path import isfile, join
    import subprocess
    xyzfiles = [f for f in listdir(struct_dir) if isfile(join(struct_dir, f)) and f.endswith((ext))]
    sdf_str = ''
    cwdir = os.getcwd()
    os.chdir(struct_dir)
    for i,fname in enumerate(xyzfiles):
        cmd_call = ["obabel", "-ixyz", fname,"-osdf"]
        p = subprocess.Popen(cmd_call, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        molblock, err = p.communicate()
        sdf_str+=molblock.decode("utf-8")
        if i%1000==0:
            print(f'iteration: {i}')
    os.chdir(cwdir)

    with open(outfile,'w') as f:
        f.write(sdf_str)

def extractBondsFromOriginalSDF():
    """
    For molecules that fail in rdkit
    :return:
    """
    pass

def constructMolecularDataFrame(infile='nmr.sdf',outfile='moldata.csv'):
    #nmr.sdf was previously made with OB
    #There are 130775 distinct molecules in data.
    #descriptorFuncs = Descriptors._descList
    descriptorFuncs = [x for x in Descriptors._descList if x[0] in descriptorList]

    ndesc = len(descriptorFuncs)
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in descriptorFuncs])
    suppl = Chem.SDMolSupplier(infile, removeHs=False, sanitize=False)
    nok = 0
    ntotal = 0
    data = []
    for i, mol in enumerate(suppl):
        ntotal +=1
        if mol is not None:
            name = mol.GetProp('_Name').replace('.xyz','')
            mol_str = Chem.MolToMolBlock(mol)
            try:
                rdmolops.SanitizeMol(mol)
                molweight = Descriptors.MolWt(mol)
                nelecs = Descriptors.NumValenceElectrons(mol)
                nrings = rdMolDescriptors.CalcNumRings(mol)
                nrot = rdMolDescriptors.CalcNumRotatableBonds(mol)
                smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))

                desc_list = list(calc.CalcDescriptors(mol))

                nok += 1

            except ValueError:
                molweight = -999
                nelecs = -999
                nrings = -999
                nrot = -999
                smiles = Chem.MolToSmiles(mol)
                desc_list = [-999]*ndesc

        data.append([name, mol_str, smiles, molweight, nelecs, nrot, nrings]+desc_list)
        if i%1000==0:
            print(f"Parsing mol {i} {name}")

    print(f"There are {ntotal} molecules.")
    print(f"There are {nok} fine molecules.")
    print(f"There are {130775-ntotal} missing molecules.")

    colnames = ['name','molblock','smiles','MolWt','NumValenceElectrons','NumRotatableBonds','nrings']+ [x[0] for x in descriptorFuncs]
    df = pd.DataFrame(data, columns = colnames )
    print(df.info())
    print(df.describe())
    print(df.head(20))
    df.to_csv(outfile,index=False)
    df[['name', 'molblock','smiles']].to_csv(outfile+'_short',index=False)


def get_interaction_data_frame(df_distance, num_nearest=5):
    #https: // www.kaggle.com / hervind / speed - up - coulomb - interaction - 56x - faster
    time_start = dt.datetime.now()
    print("START", time_start)

    # get nearest 5 (num_nearest) by distances
    df_temp = df_distance.groupby(['molecule_name', 'atom_index_0', 'atom_1'])['distance'].nsmallest(num_nearest)

    # make it clean
    df_temp = pd.DataFrame(df_temp).reset_index()[['molecule_name', 'atom_index_0', 'atom_1', 'distance']]
    df_temp.columns = ['molecule_name', 'atom_index', 'atom', 'distance']

    time_nearest = dt.datetime.now()
    print("Time Nearest", time_nearest - time_start)

    # get rank by distance
    df_temp['distance_rank'] = df_temp.groupby(['molecule_name', 'atom_index', 'atom'])['distance'].rank(ascending=True,
                                                                                                         method='first').astype(
        int)

    time_rank = dt.datetime.now()
    print("Time Rank", time_rank - time_nearest)

    # pivot to get nearest distance by atom type
    df_distance_nearest = pd.pivot_table(df_temp, index=['molecule_name', 'atom_index'],
                                         columns=['atom', 'distance_rank'], values='distance')

    time_pivot = dt.datetime.now()
    print("Time Pivot", time_pivot - time_rank)
    del df_temp

    columns_distance_nearest = np.core.defchararray.add('distance_nearest_',
                                                        np.array(df_distance_nearest.columns.get_level_values(
                                                            'distance_rank')).astype(str) +
                                                        np.array(df_distance_nearest.columns.get_level_values('atom')))
    df_distance_nearest.columns = columns_distance_nearest

    # 1 / r^2 to get the square inverse same with the previous kernel
    df_distance_sq_inv_farthest = 1 / (df_distance_nearest ** 2)

    columns_distance_sq_inv_farthest = [col.replace('distance_nearest', 'distance_sq_inv_farthest') for col in
                                        columns_distance_nearest]

    df_distance_sq_inv_farthest.columns = columns_distance_sq_inv_farthest
    time_inverse = dt.datetime.now()
    print("Time Inverse Calculation", time_inverse - time_pivot)

    #df_interaction = pd.concat([df_distance_sq_inv_farthest, df_distance_nearest], axis=1)
    df_interaction = df_distance_sq_inv_farthest
    df_interaction.reset_index(inplace=True)
    df_interaction.fillna(0.0,inplace=True)

    time_concat = dt.datetime.now()
    print("Time Concat", time_concat - time_inverse)

    return df_interaction

def read_ob_molecule(molecule_name, datadir="./data/structures"):
    mol = ob.OBMol()
    path = f"{datadir}/{molecule_name}.xyz"
    if not obConversion.ReadFile(mol, path):
        raise FileNotFoundError(f"Could not read molecule {path}")
    return mol


def get_charges_df(molecule_names,structures_idx, ob_methods = ['eem']):
    ob_methods_charges = [[] for _ in ob_methods]
    ob_molecule_name = []  # container for output  DF
    ob_atom_index = []  # container for output  DF
    ob_error = []
    for i,molecule_name in enumerate(molecule_names):
        if i%10000 ==0:
            print("OB molecule %d"%(i))
        # fill data for output DF
        ms = structures_idx.loc[molecule_name].sort_index()
        natoms = len(ms)
        ob_molecule_name.extend([molecule_name] * natoms)
        ob_atom_index.extend(ms.atom_index.values)

        # calculate open babel charge for each method
        mol = read_ob_molecule(molecule_name)
        assert (mol.NumAtoms() == natoms)  # consistency
        error = 0
        for method, charges in zip(ob_methods, ob_methods_charges):
            ob_charge_model = ob.OBChargeModel.FindType(method)
            if not ob_charge_model.ComputeCharges(mol):
                error = 1
            charges.extend(ob_charge_model.GetPartialCharges())
        ob_error.extend([error] * natoms)

    ob_charges = pd.DataFrame({
        'molecule_name': ob_molecule_name,
        'atom_index': ob_atom_index}
    )
    for method, charges in zip(ob_methods, ob_methods_charges):
        ob_charges[method] = charges
    ob_charges["error"] = ob_error
    print(ob_charges.head())
    return ob_charges


def prepareDataset(seed = 42, nsamples = -1,storedata=True, quickload=None, makeDistMat=False, makeTrainType=False, plotDist = False, makeDistMean=False, makeMolNameMean=False, bruteForceFeatures=False, dropFeatures=None, keepFeatures=None, makeLabelEncode = False, makeRDKitFeatures=False,makeRDKitFingerPrints=False, makeRDKitAtomFeatures=False, coulombMatrix=False, oneHotenc=None, obCharges=False, removeLowVariance=False, removeCorr=False, dimReduce=None):
    np.random.seed(seed)

    if storedata is not None:
        store = pd.HDFStore('./data/store.h5')
        print("Loading data store with keys: %r"%(list(store.keys())))

    if quickload is not None:
        print("Loading previous dataset...")
        store2 = pd.HDFStore('./data/store.h5')
        Xtest = store2['Xtest']
        Xtrain = store2['Xtrain']
        ytrain = store2['ytrain']
        Xval = store2['Xval']
        yval = store2['yval']
        df_param = store2['parameters']
        #print("Loaded params: %s" % (df_param))
        act_params = inspect.signature(prepareDataset)
        df_act = pd.DataFrame(act_params.parameters.items())
        #print("Current params: %s"%(df_act))
        cv_labels = None
        sample_weight = None
        #return only labels from list
        if isinstance(quickload,list):
            le = LabelEncoder()
            le.fit(j_list)
            Xtest = Xtest.loc[Xtest.type.isin(le.transform(quickload))]
            train_mask = Xtrain.type.isin(le.transform(quickload)).values
            Xtrain = Xtrain.loc[train_mask]
            ytrain = ytrain.loc[train_mask]
            val_mask = Xval.type.isin(le.transform(quickload)).values
            Xval = Xval.loc[val_mask]
            yval = yval.loc[val_mask]
            #drop type

        return(Xtest, Xtrain, ytrain, None, None, Xval, yval)

    Xtrain = pd.read_csv('./data/train.csv')
    Xtest = pd.read_csv(data_dir + r'test.csv')

    if nsamples != -1:
        if isinstance(nsamples, str) and 'shuffle' in nsamples:
            print("Shuffle train data...")
            rows = np.random.choice(len(Xtrain.index), size=len(Xtrain.index), replace=False)
        else:
            rows = np.random.choice(len(Xtrain.index), size=nsamples, replace=False)

        print("unique rows: %6.2f in pc" % (float(np.unique(rows).shape[0]) / float(rows.shape[0])))
        Xtrain = Xtrain.iloc[rows, :]
        Xtest = Xtest.iloc[:1000, :] # for not test set for prototyping

    ytrain = Xtrain['scalar_coupling_constant']
    Xtrain = Xtrain.drop(['scalar_coupling_constant'], axis=1)

    print("Xtrain shape: %s %s" % (Xtrain.shape))
    print("Xtest  shape: %s %s " % (Xtest.shape))

    df_structures = pd.read_csv('./data//structures.csv')
    Xtrain = map_atom_info(Xtrain, df_structures, 0)
    Xtrain = map_atom_info(Xtrain, df_structures, 1)

    Xtest = map_atom_info(Xtest, df_structures, 0)
    Xtest = map_atom_info(Xtest, df_structures, 1)

    train_p_0 = Xtrain[['x_0', 'y_0', 'z_0']].values
    train_p_1 = Xtrain[['x_1', 'y_1', 'z_1']].values
    test_p_0 = Xtest[['x_0', 'y_0', 'z_0']].values
    test_p_1 = Xtest[['x_1', 'y_1', 'z_1']].values

    if makeDistMat:
        Xtrain['dist'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)
        Xtest['dist'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)
        Xtrain['dist_x'] = (Xtrain['x_0'] - Xtrain['x_1']) ** 2
        Xtest['dist_x'] = (Xtest['x_0'] - Xtest['x_1']) ** 2
        Xtrain['dist_y'] = (Xtrain['y_0'] - Xtrain['y_1']) ** 2
        Xtest['dist_y'] = (Xtest['y_0'] - Xtest['y_1']) ** 2
        Xtrain['dist_z'] = (Xtrain['z_0'] - Xtrain['z_1']) ** 2
        Xtest['dist_z'] = (Xtest['z_0'] - Xtest['z_1']) ** 2

    if makeTrainType:
        Xtrain['type_0'] = Xtrain['type'].apply(lambda x: x[0])
        Xtest['type_0'] = Xtest['type'].apply(lambda x: x[0])
        Xtrain['type_1'] = Xtrain['type'].apply(lambda x: x[1:])
        Xtest['type_1'] = Xtest['type'].apply(lambda x: x[1:])

    if plotDist:
        fig, ax = plt.subplots(figsize=(18, 8))
        plt.subplot(1, 2, 1)
        plt.hist(Xtrain['dist'], bins=20)
        plt.title('Basic dist_speedup histogram')
        plt.subplot(1, 2, 2)
        sns.violinplot(x='type', y='dist', data=Xtrain)
        plt.title('Violinplot of dist_speedup by type')

    if makeDistMean:
        Xtrain['dist_to_type_mean'] = Xtrain['dist'] / Xtrain.groupby('type')['dist'].transform('mean')
        Xtest['dist_to_type_mean'] = Xtest['dist'] / Xtest.groupby('type')['dist'].transform('mean')

        Xtrain['dist_to_type_0_mean'] = Xtrain['dist'] / Xtrain.groupby('type_0')['dist'].transform('mean')
        Xtest['dist_to_type_0_mean'] = Xtest['dist'] / Xtest.groupby('type_0')['dist'].transform('mean')

        Xtrain['dist_to_type_1_mean'] = Xtrain['dist'] / Xtrain.groupby('type_1')['dist'].transform('mean')
        Xtest['dist_to_type_1_mean'] = Xtest['dist'] / Xtest.groupby('type_1')['dist'].transform('mean')

    # be aware of overfitting here
    if makeMolNameMean:
        Xtrain[f'molecule_type_dist_mean'] = Xtrain.groupby(['molecule_name', 'type'])['dist'].transform('mean')
        Xtest[f'molecule_type_dist_mean'] = Xtest.groupby(['molecule_name', 'type'])['dist'].transform('mean')


    if bruteForceFeatures:
        Xtrain = create_brute_force_features_kernel(Xtrain)
        Xtest = create_brute_force_features_kernel(Xtest)


    if makeLabelEncode:
        for f in ['atom_0', 'atom_1', 'type_0', 'type_1', 'type']:
            if f in Xtrain.columns:
                lbl = LabelEncoder()
                if f == 'type':
                    lbl.fit(j_list)
                else:
                    lbl.fit(list(Xtrain[f].values) + list(Xtest[f].values))
                Xtrain[f] = lbl.transform(list(Xtrain[f].values))
                Xtest[f] = lbl.transform(list(Xtest[f].values))

    Xall = pd.concat([Xtest, Xtrain], ignore_index=True)

    if makeRDKitFeatures:
        if os.path.isfile('rdkit_featuresX.csv'):
            print("Loading RDKit features")
            Xrdkit = pd.read_csv('rdkit_features.csv',index_col=0)
            Xrdkit = reduce_mem_usage(Xrdkit)
            print("Merging RDKit features")
            Xall = pd.merge(Xall, Xrdkit, how='left',
                          left_on='id',
                          right_index=True)

        else:
            print("Loading RDKit molecular features...")
            # load moldata.csv
            # join features for each molecule
            moldata = pd.read_csv('moldata.csv')
            moldata = reduce_mem_usage(moldata)
            # moldata = removeLowVar(moldata, threshhold=1E-5)

            print("Mapping RDKit molecular features...")
            Xall,rdkit_colnames = map_rdkit_features(Xall,moldata)
            Xall = reduce_mem_usage(Xall)

            print(Xall.head())
            #save data
            Xall[rdkit_colnames].to_csv('rdkit_features.csv')

        print(Xall.columns)


    if makeRDKitFingerPrints:
        print("Loading RDKit molblock...")
        moldata = pd.read_csv('moldata_short.csv')
        print("Mapping RDKit mol string...")
        Xall, rdkit_colnames = map_rdkit_features(Xall, moldata)
        # compute some descriptors pair FP are bad
        #https://apassionatechie.wordpress.com/2017/12/27/create-multiple-pandas-dataframe-columns-from-applying-a-function-with-multiple-returns/
        print("Computing Morgan fingerprints...")
        df_tmp = Xall.apply(getMorganFP, axis=1)
        new_colums = ['mfp_rad1_at1','mfp_rad1_at2']
        df_tmp.columns = new_colums
        #should do an index based merge here...
        Xall[[new_colums]] = df_tmp
        print(Xall.head(20))
        Xall = reduce_mem_usage(Xall)

    if makeRDKitAtomFeatures:
        print("Loading RDKit molblock...")
        moldata = pd.read_csv('moldata_short.csv')
        print("Mapping RDKit mol string...")
        Xall, rdkit_colnames = map_rdkit_features(Xall, moldata)
        print("Computing some atomic features...")
        df_tmp = Xall.apply(getAtomicFeatures, axis=1)
        Xall = Xall.merge(df_tmp,left_index=True, right_index=True)
        print(Xall.head())
        Xall = reduce_mem_usage(Xall)

    if coulombMatrix:
        #expensive!!
        if not os.path.isfile("df_interact.csv"):
            print("Creating Coulomb matrix features")
            #https://www.kaggle.com/hervind/speed-up-coulomb-interaction-56x-faster
            #self join, yields atom x atom for each molecule
            df_distance = df_structures.merge(df_structures, how='left', on='molecule_name', suffixes=('_0', '_1'))
            print("Remove identical atom from rows...")
            df_distance = df_distance.loc[df_distance['atom_index_0'] != df_distance['atom_index_1']]
            print("Computing distance...")
            df_distance['distance'] = np.linalg.norm(df_distance[['x_0', 'y_0', 'z_0']].values -
                                                     df_distance[['x_1', 'y_1', 'z_1']].values, axis=1, ord=2)
            cutoff = 2.0
            df_distance = df_distance.loc[df_distance['distance']<cutoff]
            df_distance = reduce_mem_usage(df_distance)
            print("Computing interactions...")
            # estimate time about 40h!!!
            first_100_molecules = Xall['molecule_name'].unique()[:Xall.shape[0]]

            df_interact = get_interaction_data_frame(df_distance.loc[df_distance['molecule_name'].isin(first_100_molecules)])
            df_interact.to_csv("df_interact.csv")
        else:
            df_interact = pd.read_csv("df_interact.csv",index_col=0)

            Xall = pd.merge(Xall, df_interact, how='left',
                          left_on=['molecule_name', 'atom_index_0'],
                          right_on=['molecule_name', 'atom_index'], suffixes=('_at1', '_at2'))
            #merge on other atom
            Xall = pd.merge(Xall, df_interact, how='left',
                            left_on=['molecule_name', 'atom_index_1'],
                            right_on=['molecule_name', 'atom_index'], suffixes=('_at1', '_at2'))

    if obCharges:
        if not os.path.isfile("xall_ob_charges.csv"):
            print("Computing OB charges...")
            ob_methods = ["eem", "mmff94", "gasteiger", "qeq", "qtpie",
                          "eem2015ha", "eem2015hm", "eem2015hn", "eem2015ba", "eem2015bm", "eem2015bn"]
            #ob_methods = ["eem", "gasteiger"]
            xall_molecules = Xall.molecule_name.unique()
            structures_idx = df_structures.set_index(["molecule_name"])
            xall_ob_charges = get_charges_df(xall_molecules,structures_idx = structures_idx, ob_methods = ob_methods)
            xall_ob_charges.to_csv("xall_ob_charges.csv")
        else:
            print("Loading OB charges...")
            xall_ob_charges = pd.read_csv("xall_ob_charges.csv", index_col=0)

        Xall = pd.merge(Xall, xall_ob_charges, how='left',
                        left_on=['molecule_name', 'atom_index_0'],
                        right_on=['molecule_name', 'atom_index'],suffixes=('_at1', '_at2'))

        Xall = pd.merge(Xall, xall_ob_charges, how='left',
                        left_on=['molecule_name', 'atom_index_1'],
                        right_on=['molecule_name', 'atom_index'],suffixes=('_at1', '_at2'))


    Xtrain = Xall[len(Xtest.index):]
    Xtest = Xall[:len(Xtest.index)]

    if oneHotenc is not None:
        print("1-0 Encoding categoricals...", oneHotenc)
        Xall = pd.concat([Xtest, Xtrain], ignore_index=True)
        for col in oneHotenc:
            if col in Xall.columns:
                uv = np.unique(Xall[col].values)
                print("Unique values for col:", col, " -", uv)
                encoder = OneHotEncoder()
                X_onehot = pd.DataFrame(encoder.fit_transform(Xall[[col]].values).todense())
                X_onehot.columns = [col + "_" + str(column) for column in X_onehot.columns]
                print("One-hot-encoding of %r...new shape: %r" % (col, X_onehot.shape))
                Xall.drop([col], axis=1, inplace=True)
                Xall = pd.concat([Xall, X_onehot], axis=1)
                print("One-hot-encoding final shape:", Xall.shape)

    if keepFeatures is not None:
        dropcols = [col for col in Xall.columns if col not in keepFeatures]
        for col in dropcols:
                print("Dropping: ", col)
                Xall.drop([col], axis=1, inplace=True)

    if dropFeatures is not None:
        for col in dropFeatures:
            print("Dropping: ", col)
            Xall.drop([col], axis=1, errors='ignore', inplace=True)

    Xall = Xall.drop(['id', 'molecule_name'], errors='ignore',axis=1)

    if removeLowVariance:
        print("remove low var using threshhold...")
        Xall = removeLowVar(Xall, threshhold=1E-5)

    if removeCorr:
        Xall = removeCorrelations(Xall, None, 0.995)


    if dimReduce is not None:
        print("Reducing dimensions...")
        if isinstance(dimReduce,int):
            reducer = TruncatedSVD(n_components=dimReduce)
        #reducer = PCA(n_components=dimReduce)
        else:
            reducer = dimReduce
        print(reducer)
        Xtrain = Xall[len(Xtest.index):]
        reducer.fit(Xtrain)
        print("Explained variance ratio:", reducer.explained_variance_ratio_)
        print("Explained variance ratio sum: %6.3f"%(reducer.explained_variance_ratio_.sum()))
        Xall = pd.DataFrame(reducer.transform(Xall))
        Xall.columns = ["d_" + str(column) for column in range(Xall.shape[1])]

    Xtrain = Xall[len(Xtest.index):]
    Xtest = Xall[:len(Xtest.index)]

    cv_labels = None
    sample_weight = None
    #create validation set
    Xtrain, Xval, ytrain, yval = train_test_split(Xtrain, ytrain, test_size = 0.25, random_state = 42)
    print("Xval shape: %s %s" % (Xval.shape))

    if storedata:
        store['Xtest'] = Xtest
        store['Xtrain'] = Xtrain
        store['ytrain'] = ytrain
        store['Xval'] = Xval
        store['yval'] = yval
        param_dict = inspect.signature(prepareDataset)
        store['parameters'] = pd.DataFrame(param_dict.parameters.items())
        store.close()

    print("Finished feature preparation...")
    return (Xtest,Xtrain,ytrain,cv_labels, sample_weight, Xval, yval)

def adversarialFeatureRemovel(train,test):
    #https://www.kaggle.com/artgor/validation-feature-selection-interpretation-etc
    n_fold = 5

    folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)

    feature_importance = {}
    for t in ['1JHN', '2JHN', '3JHN']:
        print(f'Type: {t}. {time.ctime()}')
        X = train.loc[train['type'] == t, good_columns]
        y = train.loc[train['type'] == t, 'scalar_coupling_constant']
        X_test = test.loc[test['type'] == t, good_columns]

        features = X.columns
        X['target'] = 0
        X_test['target'] = 1

        train_test = pd.concat([X, X_test], axis=0)
        target = train_test['target']

        result_dict_lgb = train_model_classification(train_test, X_test, target, params, folds, model_type='lgb',
                                                     columns=features, plot_feature_importance=True, model=None,
                                                     verbose=500,
                                                     early_stopping_rounds=200, n_estimators=500)
        plt.show();

        feature_importance[t] = result_dict_lgb['feature_importance']

    bad_advers_columns = {}
    for t in feature_importance.keys():
        cols = feature_importance[t][["feature", "importance"]].groupby("feature").mean().sort_values(
            by="importance", ascending=False).index[:7]
        bad_advers_columns[t] = cols


def makePredictions(model=None, Xtest=None, filename='nmr'):
    from datetime import datetime
    from sklearn.exceptions import NotFittedError
    now = datetime.now().strftime("%Y%m%d")
    filename = filename + '_' + now + '.csv'

    sub = pd.read_csv('./data/sample_submission.csv')

    if model is not None:
        try:
            if not hasattr(model, 'predict_proba'):
                preds = model.predict(Xtest)
            else:
                preds = model.predict_proba(Xtest)[:, 1]
        except NotFittedError as e:
            print(repr(e))
            return

    else:
        preds = Xtest

    if sub.shape[0] != preds.shape[0]:
        print("Prediction do not have the right shape!")
    else:
        sub['scalar_coupling_constant'] = preds
        sub.to_csv(filename, index=False)
        print(f'Saved submission to: {filename}')
        print(sub.head())
        logging.info(f'Saved submission to: {filename}')


def main():
    #check https://www.kaggle.com/asauve/training-set-molecule-visualization
    #https://www.kaggle.com/artgor/brute-force-feature-engineering
    #check karplus and dihedrals  https://www.kaggle.com/c/champs-scalar-coupling/discussion/93793#latest-548247
    #https: // socratic.org / questions / what - factors - affect - coupling - constants
    #https://www.chem.wisc.edu/areas/reich/nmr/05-hmr-04-2j.htm
    #https://www.annualreviews.org/doi/pdf/10.1146/annurev.biophys.33.110502.133350
    #https://www.kaggle.com/asauve/v7-estimation-of-mulliken-charges-with-open-babel
    #lgbm: https://www.kaggle.com/c/home-credit-default-risk/discussion/58332#latest-476077
    #lgbm parameters: https://sites.google.com/view/lauraepp/parameters
    # build coulomb matrix features
    # build distance total
    # atom features
    # https://www.kaggle.com/jmtest/molecule-with-openbabel
    # remove unphysical features
    # check steric i.e. average distance nearest atom via dm
    # use mamba features
    # instead of separate models: use sample_weights, use type as meta_feature in stage 2
    # use 1/R as descriptor
    # use is ring member as descriptor
    # use angles as descriptor
    # generate fragments via rdkit
    # https://www.kaggle.com/artgor/using-meta-features-to-improve-model

    #CV mean score: 0.3579, std: 0.0006
    # learning curve lgb:
    # 10000: MAE:3.0026 +/-0.0714 group MAE :1.0397 +/-0.0255
    # 20000: MAE:2.8938 +/-0.0471 group MAE :0.9968 +/-0.0257
    # 30000: MAE:2.8027 +/-0.0259 group MAE :0.9512 +/-0.0101
    # 50000: MAE:2.7112 +/-0.0439 group MAE :0.9097 +/-0.0136
    # 50000: CV mean score: 0.7452, std: 0.0221. [early stopping]
    # 100000: CV mean score: 0.6013, std: 0.1665. [default features]
    # 100000: CV mean score: 0.6948, std: 0.3610. [default features]
    # 100000: CV mean score: 0.7113, std: 0.0170. [default features]
    # 100000: CV mean score: 0.6619, std: 0.0154. [with rdkit mol features & atompairs FP]
    # 100000: CV mean score: 0.6631, std: 0.1450. [with rdkit mol features & atompairs FP]
    # 200000:                                     [with rdkit mol features & atompairs FP]
    # -1:     CV mean score: 0.2273, std: 0.0009. [with rdkit mol features & atompairs FP]
    # -1:     CV mean score: 0.5482, std: 0.0074  [default features]
    # 100000: CV mean score: 0.5050, std: 0.0956. [with rdkit 200 mol features]
    # -1:     CV mean score: 0.2814, std: 0.0062. [with rdkit 100 mol features]
    # 1333333.CV mean score: 0.5473, std: 0.0297 [with rdkit  mol features]
    # 200000: CV mean score: 0.4704, std: 0.0764. [with rdkit 100 mol features]
    # 133333: CV mean score: 0.6480, std: 0.1652. [default features]
    # 133333: CV mean score: 0.1485, std: 0.1166  [coulomb matrix]
    # 133333: CV mean score: 0.0612, std: 0.1584. [ob charges]
    # 133333: CV mean score: -0.0849, std: 0.1564 [coulomb matrix & oob charges]
    # 133333: CV mean score: CV mean score: 0.5560, std: 0.1586. [morganFP r0 r0]
    # 133333: CV mean score: CV mean score: 0.3397, std: 0.0936. [morganFP r1 r0]
    # 133333: CV mean score: CV mean score: 0.3166,              [morganFP r1 r1]
    # 1333333: CV mean score: 0.4441, std: 0.0194               [morganFP r2 r0]
    # 266666: CV mean score: CV mean score: 0.5447, std: 0.0668. [morganFP r1 r1]
    # 1333333: CV mean score: 0.6202, std: 0.1759. [brute force basics]??
    # 266666: CV mean score: 0.8016, std: 0.0926. [brute force basics]
    # 500000: CV mean score: 0.7595, std: 0.0788. [brute force basics]
    # 400000: CV mean score: 0.7091, std: 0.0346. [brute force basics]
    # 400000: CV mean score: 0.6669, std: 0.0636. [default features]
    # 400000: CV mean score: 0.7081, std: 0.0509. [brute force  170 features]
    # 400000: CV mean score: 0.6795, std: 0.0671. [brute force  good columns]
    # 800000: CV mean score: 0.7497, std: 0.0372. [brute force  good columns]
    # 800000: CV mean score: 0.7864, std: 0.0267. [brute force  physical]
    # 800000: CV mean score: 0.7493, std: 0.0367. [features from kernel]
    # 133333: CV mean score: 0.5714, std: 0.1981. [RDkit atomic features]
    # -1    : OOF score: 0.3408         [8 sep models] [default features]


    result_dict = {}
    unphysical = ['x_0','y_0','z_0','x_1','y_2','z_3','dist_x','dist_y','dist_z']
    data_params = {
        'seed' : 42,
        'nsamples' : -1,
        'storedata' : True,
        'makeDistMat' : True,
        'makeTrainType' : True,
        'plotDist' : False,
        'makeDistMean' : True,
        'makeMolNameMean' : True,
        'bruteForceFeatures' :False,
        'makeLabelEncode' : True,
        'makeRDKitFeatures': False,
        'makeRDKitFingerPrints': False,
        'makeRDKitAtomFeatures': False,
        'coulombMatrix': False,
        'obCharges' : True,
        'oneHotenc' : ['mfp_rad0_at1', 'mfp_rad1_at1', 'mfp_rad2_at1', 'mfp_rad3_at1', 'mfp_rad0_at2', 'mfp_rad1_at2',
              'mfp_rad2_at2', 'mfp_rad3_at2'],
        'removeLowVariance' : False,
        'keepFeatures' : None,
        'dropFeatures' : ['atom_index_0','atom_index_1','molblock'],
        'dimReduce' : None
    }
    Xtest, Xtrain, ytrain, _, _, Xval, yval = prepareDataset(**data_params)

    print("Xtrain shape: %s %s" % (Xtrain.shape))
    print("Xtrain columns: %r"%(list(Xtrain.columns)))

    n_fold = 5
    lgb_params = {'num_leaves': 128,
              'min_child_samples': 79,
              'objective': 'regression',
              'max_depth': 12,
              'learning_rate': 0.2,
              "boosting_type": "gbdt",
              "subsample_freq": 1,
              "subsample": 1.0,
              "bagging_seed": 11,
              "metric": 'mae',
              "verbosity": -1,
              'reg_alpha': 0.1,
              'reg_lambda': 0.3,
              'colsample_bytree': 1.0,
              'n_estimators': 5000,
               'n_jobs': -1,
              }

    logging.info("\n -MODEL INFO-")
    logging.info('data_params: %r' %(data_params))
    #logging.info('params: %r' % (lgb_params))

    Xtrain_f = pd.DataFrame({'ind': list(Xtrain.index), 'type': Xtrain['type'].values, 'oof': [0] * len(Xtrain), 'target': ytrain.values})
    Xval_f = pd.DataFrame({'ind': list(Xval.index), 'type': Xval['type'].values, 'prediction': [0] * len(Xval), 'target': yval.values})
    Xtest_f = pd.DataFrame({'ind': list(Xtest.index), 'type': Xtest['type'].values, 'prediction': [0] * len(Xtest)})

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    model = lgb.LGBMRegressor(**lgb_params)
    # model = RidgeCV()
    logging.info('model:  %r ' % (str(model)))

    fit_types = j_list
    # fit_types = ['2JHN']
    for i,t  in enumerate(fit_types):
        print("\n>>%d - Coupling-type %s<<"%(i,t))
        Xtest, Xtrain, ytrain, _, _, Xval, yval = prepareDataset(**data_params, quickload=[t])

        print("Xtest shape:    %s %s" % (Xtest.shape))
        print("Xtrain shape:   %s %s" % (Xtrain.shape))
        print("Xtrain columns: %r" % (list(Xtrain.columns)))
        #result_dict = buildXvalModel(model, Xtrain, ytrain, sample_weight=None, class_names=None, refit=False, cv=cv)
        #eval = lgb.cv(lgb_params,lgb.Dataset(Xtrain, ytrain),  nfold=5,stratified=False, early_stopping_rounds=200, verbose_eval=100,seed=5,show_stdv=True)
        result_dict = train_model_regression(X=Xtrain, X_test=Xtest, y=ytrain, params=lgb_params, folds=cv,model_type='lgb', eval_metric='group_mae',plot_feature_importance=True,verbose=1000, early_stopping_rounds=200)

        le = LabelEncoder()
        le.fit(j_list)
        type_int = le.transform([t])[0]

        Xtrain_f.loc[Xtrain_f['type'] == type_int, 'oof'] = result_dict['oof']
        Xtest_f.loc[Xtest_f['type'] == type_int, 'prediction'] = result_dict['prediction']

        doVal = False
        if Xval is not None and doVal:
            if isinstance(lgb.LGBMRegressor):
                early_stopping_rounds = 200
                model.fit(Xtrain,ytrain,early_stopping_rounds=early_stopping_rounds)
            else:
                model.fit(Xtrain, ytrain)
            ypred = model.predict(Xval)
            score = group_mean_log_mae(yval, ypred, Xval['type'])
            print("Validation score: %6.4f"%(score))
            #plot_types(yval, ypred, Xval['type'].astype(int),t)
            Xval_f.loc[Xval_f['type'] == type_int, 'prediction'] = yval

        #logging.info('cv:     %r ' % (result_dict['cv']))
        #logging.info('scores: %r '%(result_dict['scores']))
        logging.info('<scores>: %r ' % (np.mean(result_dict['scores'])))

        if 'feature_importance' in result_dict.keys():
            logging.info(result_dict['feature_importance'])
            logging.info(list(result_dict['feature_importance'].index))

    score_oob = group_mean_log_mae(Xtrain_f['target'], Xtrain_f['oof'], Xtrain_f['type'])
    print("OOF score: %6.4f" % (score_oob))
    score_val = group_mean_log_mae(Xval_f['target'], Xval_f['prediction'], Xval_f['type'])
    print("VAL score: %6.4f" % (score_val))
    for i, t in enumerate(fit_types):
        plot_types(Xtrain_f['target'], Xtrain_f['oof'], Xtrain_f['type'], t)
        

    makePredictions(model,Xtest_f,filename='submissions/nmr')

    plt.show()

if __name__ == "__main__":
    logging.basicConfig(filename='log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    #createSDFileViaOpenBabel()
    #constructMolecularDataFrame()
    #analyzeDataSet(plotHist=False, plotGraph=False)
    main()



