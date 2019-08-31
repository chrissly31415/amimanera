#!/usr/bin/python
# coding: utf-8

import cProfile

import logging

import os,sys

import gc

import inspect

import datetime as dt

import numpy as np
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 100)

import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (15, 5)

import seaborn as sns

import networkx as nx

from tqdm import tqdm

import pickle as pkl

import time


#machine learning libraries
from sklearn.linear_model import RidgeCV,LassoCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import KFold,GroupKFold,train_test_split
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import GenericUnivariateSelect,mutual_info_regression,f_regression,SelectFromModel

import xgboost as xgb
from xgboost import XGBRegressor

import lightgbm as lgb

from keras_tools import *
from keras.wrappers.scikit_learn import KerasRegressor

from artgor_utils import train_model_regression

#cheminformatics library
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import rdMolTransforms
from rdkit.Chem import rdmolops

from rdkit.Chem.AtomPairs import Pairs


import multiprocessing as mp

from ase.io import read
import ase
from dscribe.descriptors import SOAP

import openbabel as ob
obConversion = ob.OBConversion()

from nmr_utils import *
from qsprLib import buildXvalModel,removeLowVar,showAVGCorrelations,reduce_mem_usage,group_mean_log_mae
from qsprLib import duplicate_columns,makeGridSearch,removeCorrelations

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
descRDkitTop20 = ['VSA_EState7', 'PEOE_VSA8', 'qed', 'BalabanJ', 'PEOE_VSA9', 'VSA_EState8', 'MinAbsEStateIndex', 'MinEStateIndex', 'VSA_EState5', 'VSA_EState4', 'MaxEStateIndex', 'Kappa3', 'Chi4n', 'SMR_VSA5', 'BertzCT', 'Ipc', 'MinPartialCharge', 'PEOE_VSA7', 'Chi3n', 'FpDensityMorgan3']


desc1JHCTOP = ['type','molecule_type_1_dist_mean', 'mc_nb1', 'dist_y', 'mc_nb2', 'mc_at0', 'distance_sq_inv_farthest_1C_at2', 'dist_z', 'dist_x', 'eem_at1', 'mc_at1', 'mc_nb4', 'z_0', 'y_0', 'gasteiger_at2', 'mc_nb3', 'x_0', 'molecule_type_dist_mean', 'z_1', 'gasteiger_at1', 'molecule_type_0_dist_mean', 'x_1', 'y_1', 'eem_at2', 'dist', 'distance_sq_inv_farthest_1H_at1', 'distance_sq_inv_farthest_2C_at2', 'qeq_at1', 'distance_sq_inv_farthest_1H_at2', 'qtpie_at1', 'distance_sq_inv_farthest_2H_at2', 'directdist', 'distance_sq_inv_farthest_1C_at1', 'mmff94_at2', 'qtpie_at2', 'qeq_at2', 'nb2', 'atom_index_0', 'nb3', 'distance_sq_inv_farthest_1O_at2', 'nb1', 'distance_sq_inv_farthest_1N_at2', 'distance_sq_inv_farthest_2H_at1', 'atom_index_1', 'distance_sq_inv_farthest_3H_at2', 'nb4', 'distance_sq_inv_farthest_3C_at2', 'dist_to_type_1_mean', 'dist_to_type_mean', 'inRingb', 'nbelem1', 'dist_to_type_0_mean', 'mmff94_at1', 'nbelem2', 'distance_sq_inv_farthest_1O_at1', 'nbelem3', 'distance_sq_inv_farthest_4C_at2', 'distance_sq_inv_farthest_2O_at2', 'distance_sq_inv_farthest_2N_at2', 'aromb', 'implValb', 'distance_sq_inv_farthest_3H_at1', 'nneighbors']
desc1JHNTOP = ['molecule_type_0_dist_mean', 'dist_y', 'molecule_type_1_dist_mean', 'dist_x', 'dist_z', 'gasteiger_at2', 'gasteiger_at1', 'mc_nb1', 'mc_at0', 'eem_at1', 'mc_nb3', 'z_0', 'x_0', 'mc_at1', 'molecule_type_dist_mean', 'y_0', 'distance_sq_inv_farthest_1C_at2', 'qeq_at1', 'mc_nb2', 'z_1', 'qtpie_at1', 'x_1', 'eem_at2', 'y_1', 'distance_sq_inv_farthest_1N_at1', 'qtpie_at2', 'qeq_at2', 'dist', 'directdist', 'distance_sq_inv_farthest_2C_at2', 'distance_sq_inv_farthest_1H_at2', 'atom_index_0', 'distance_sq_inv_farthest_1H_at1', 'mmff94_at2', 'distance_sq_inv_farthest_2H_at2', 'nb2', 'atom_index_1', 'nb3', 'distance_sq_inv_farthest_1C_at1', 'nb1', 'dist_to_type_mean', 'mc_nb4', 'dist_to_type_0_mean', 'mmff94_at1', 'distance_sq_inv_farthest_1N_at2', 'distance_sq_inv_farthest_2N_at1', 'dist_to_type_1_mean', 'inRingb', 'implValb', 'nbelem2', 'nb4', 'nbelem1', 'aromb', 'distance_sq_inv_farthest_2H_at1', 'distance_sq_inv_farthest_1O_at1', 'distance_sq_inv_farthest_2N_at2', 'nneighbors']
desc2JHCTOP=['angle', 'gasteiger_at1', 'total_dist_type2', 'mc_at0', 'distance_sq_inv_farthest_1C_at1', 'eem_at1', 'molecule_type_1_dist_mean', 'mc_nb1', 'dist', 'gasteiger_at2', 'mc_nb4', 'y_0', 'z_0', 'dist_y', 'x_0', 'mc_nb3', 'dist_z', 'mc_nb2', 'molecule_type_dist_mean', 'dist_x', 'qtpie_at1', 'distance_sq_inv_farthest_1C_at2', 'mc_at1', 'molecule_type_0_dist_mean', 'y_1', 'z_1', 'qeq_at1', 'mmff94_at2', 'x_1', 'distance_sq_inv_farthest_2C_at2', 'eem_at2', 'distance_sq_inv_farthest_1H_at2', 'distance_sq_inv_farthest_1H_at1', 'qtpie_at2', 'qeq_at2', 'distance_sq_inv_farthest_1O_at2', 'distance_sq_inv_farthest_3C_at2', 'atom_index_0', 'nb1', 'nb3', 'nb2', 'distance_sq_inv_farthest_1N_at2', 'directdist', 'atom_index_1', 'nb4', 'distance_sq_inv_farthest_2H_at2', 'nbelem1', 'distance_sq_inv_farthest_2H_at1', 'mmff94_at1', 'distance_sq_inv_farthest_1N_at1', 'distance_sq_inv_farthest_4C_at2', 'implValb', 'dist_to_type_mean', 'inRingb', 'distance_sq_inv_farthest_3H_at2', 'distance_sq_inv_farthest_2N_at2', 'dist_to_type_0_mean', 'distance_sq_inv_farthest_1O_at1', 'aromb', 'nbelem4', 'nbelem3', 'dist_to_type_1_mean', 'distance_sq_inv_farthest_2O_at2', 'nbelem2', 'nneighbors', 'distance_sq_inv_farthest_5C_at2', 'distance_sq_inv_farthest_3N_at2', 'distance_sq_inv_farthest_1F_at2', 'distance_sq_inv_farthest_2N_at1', 'distance_sq_inv_farthest_2O_at1', 'jkarplus']
desc3JHCTOP=['angle_dih1', 'angle', 'angle_dih2', 'total_dist_type2', 'total_dist_type3', 'dist', 'dihedral', 'distance_sq_inv_farthest_1C_at1', 'gasteiger_at1', 'mc_nb3', 'mc_nb4', 'mc_at0', 'mc_nb1', 'mc_nb2', 'molecule_type_1_dist_mean', 'y_0', 'distance_sq_inv_farthest_1C_at2', 'molecule_type_dist_mean', 'x_0', 'distance_sq_inv_farthest_1H_at1', 'z_0', 'gasteiger_at2', 'dist_y', 'eem_at1', 'y_1', 'dist_z', 'molecule_type_0_dist_mean', 'dist_x', 'qtpie_at1', 'x_1', 'z_1', 'mc_at1', 'mmff94_at2', 'distance_sq_inv_farthest_1H_at2', 'qeq_at1', 'distance_sq_inv_farthest_2C_at2', 'eem_at2', 'qtpie_at2', 'qeq_at2', 'jkarplus', 'atom_index_0', 'nb3', 'distance_sq_inv_farthest_1O_at2', 'distance_sq_inv_farthest_3C_at2', 'distance_sq_inv_farthest_2H_at2', 'nb1', 'nb2', 'distance_sq_inv_farthest_1N_at2', 'atom_index_1', 'distance_sq_inv_farthest_2H_at1', 'nb4', 'mmff94_at1', 'directdist', 'implValb', 'distance_sq_inv_farthest_1N_at1', 'nbelem1', 'distance_sq_inv_farthest_3H_at2', 'distance_sq_inv_farthest_1O_at1', 'distance_sq_inv_farthest_4C_at2', 'inRingb', 'dist_to_type_mean', 'distance_sq_inv_farthest_2N_at2', 'nbelem3', 'aromb', 'nbelem4', 'distance_sq_inv_farthest_2O_at2', 'nbelem2', 'nneighbors', 'distance_sq_inv_farthest_5C_at2', 'dist_to_type_0_mean', 'dist_to_type_1_mean', 'distance_sq_inv_farthest_2O_at1', 'distance_sq_inv_farthest_3H_at1', 'distance_sq_inv_farthest_1F_at2', 'distance_sq_inv_farthest_3N_at2', 'distance_sq_inv_farthest_2N_at1', 'distance_sq_inv_farthest_3O_at2', 'distance_sq_inv_farthest_2F_at2', 'distance_sq_inv_farthest_3F_at2', 'jch_est']



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


def getASEMOLfromName(molname,structure):
    tmp_structure = structure.loc[structure.molecule_name == molname, :].copy()
    species = tmp_structure.atom.unique()  # array(['C', 'H'], dtype=object)
    molecule_atoms = tmp_structure.loc[:, 'atom']
    molecule_positions = tmp_structure.loc[:, ['x', 'y', 'z']]
    molecule_system = ase.atoms.Atoms(symbols=molecule_atoms, positions=molecule_positions)
    return(molecule_system)


def getStructureFromName(molname):
    xyz_path = struct_dir+molname+'.xyz'
    struc_str = []
    if os.path.isfile(xyz_path):
        with open(xyz_path) as f:
            for line in f.readlines():
                struc_str.append(line.strip())

    return(struc_str)


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

def sanitize_molecules(mol_str):
    mol = Chem.MolFromMolBlock(mol_str, removeHs=False, sanitize=False)
    sanitFail = Chem.SanitizeMol(mol, sanitizeOps=Chem.rdmolops.SanitizeFlags.SANITIZE_ALL, catchErrors=True)
    if sanitFail: # 2nd try
        sanitFail = Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_KEKULIZE|Chem.SANITIZE_SYMMRINGS|Chem.SANITIZE_SETCONJUGATION|Chem.SANITIZE_SETHYBRIDIZATION, catchErrors=True)
        #print("WARNING: 2nd try: %s" % (row['molecule_name']))
    if sanitFail:
        print("Sanitization failed")
        return None
    return mol


def getMAMBAFeatures(row):
    mol_str = row['molblock']
    row['molblock'] = None
    mol = sanitize_molecules(mol_str)
    df = extract_features(mol, row['name'], 0, fillNa=np.nan,  OrderAtoms=True)
    df['molecule_name'] = row['name']
    return(df)


def soap_core_func(name,df,soap_desc):
    asemol = getASEMOLfromName(name, df)
    soap_feat = soap_desc.create(asemol)
    df2 = pd.DataFrame(soap_feat)
    df2['molecule_name'] = name
    df2.reset_index(inplace=True,drop=False)
    return(df2)


def getSOAPfeatures_new(df_structures,soap_desc=None):
    #df_structures = df_structures.iloc[:100]
    chunk_iter = df_structures.groupby(['molecule_name'])
    pool = mp.Pool(4)  # use 4 CPU cores
    funclist = []
    print("Defining functions ...")
    for name, df in tqdm(chunk_iter):
        #asemol = getASEMOLfromName(name, df)
        # process each data frame, parallel over each moleculue...?
        #f = pool.apply_async(soap_desc.create, asemol)
        f = pool.apply_async(soap_core_func, args=(name,df,soap_desc))
        funclist.append(f)

    print("Collecting results ...")
    result = []
    for f in tqdm(funclist):
        result.append(f.get())

    # combine chunks with transformed data into a single training set
    structures_soap = pd.concat(result)
    structures_soap.rename(columns={"index": "atom_index"},inplace=True)

    return(structures_soap)


def getSOAPfeatures(row,soap_desc=None):
    from dscribe.descriptors import SOAP
    from dscribe.descriptors import CoulombMatrix
    molname = row['molecule_name']
    j = row['atom_index_1']
    asemol = read('./data/structures/'+molname+'.xyz')
    soap_at1 = soap_desc.create(asemol, positions=[j])
    if row['id'] % 1000 ==0:
        print("id: %d mol: %s"%(row['id'],row['molecule_name']))
    sindex = ['soap'+str(i) for i in range(len(soap_at1[0]))]
    res = pd.Series(soap_at1[0], index=sindex)
    return (res)

def getBondNeighbors(row):
    mol_str = row['molblock']
    row['molblock'] = None
    if row['id'] % 10000 ==0:
        print(row['id'])
        gc.collect()
    mol = sanitize_molecules(mol_str)
    nb_index = np.full((4,),-1)
    nb_elem = np.full((4,),-1)
    if mol is None:
        print("WARNING: mol defect!")
    else:
        #i = row['atom_index_0']
        j = row['atom_index_1']
        neighbors = mol.GetAtomWithIdx(j).GetNeighbors()
        for k,atom in enumerate(neighbors):
            nb_index[k]= atom.GetIdx()
            nb_elem[k] = atom.GetAtomicNum()

        sort_indices = np.argsort(nb_elem)[::-1]
        nb_elem = nb_elem[sort_indices]
        nb_index = nb_index[sort_indices]

        #for idx in nb_index:
            #nneighbors = mol.GetAtomWithIdx(idx).GetNeighbors()
            #for k,atom in enumerate(neighbors):
                #nb_index[k]= atom.GetIdx()
                #nb_elem[k] = atom.GetAtomicNum()



    prop_dict = {'nneighbors': len(neighbors),
                 'nb1': nb_index[0], 'nb2': nb_index[1], 'nb3': nb_index[2],'nb4': nb_index[3],
                 'nbelem1': nb_elem[0], 'nbelem2': nb_elem[1], 'nbelem3': nb_elem[2],'nbelem4': nb_elem[3]}

    res = pd.Series(prop_dict)
    return res

def getAtomicFeatures(row):
    #https://www.kaggle.com/c/champs-scalar-coupling/discussion/93972#latest-546673
    mol_str = row['molblock']
    mol = sanitize_molecules(mol_str)
    if mol is None:
        print("WARNING: mol defect!")
        # atna = 0
        atnb = 0
        # nbondsa = 0
        nbondsb = 0
        # ndegreea = 0
        # ndegreeb = 0
        # hybra = 0
        # hybrb = 0
        # implVala = 0
        implValb = 0
        # aroma = False
        aromb = False
        # numHa = 0
        numHb = 0
        # inRinga = False
        inRingb = False
        jch_est = 0
    else:
        i = row['atom_index_0']
        j = row['atom_index_1']
        #a = mol.GetAtomWithIdx(i)
        b = mol.GetAtomWithIdx(j)
        #atna = a.GetAtomicNum()
        atnb = b.GetAtomicNum()
        #nbondsa = len(a.GetBonds())
        nbondsb = len(b.GetBonds())
        #ndegreea = a.GetDegree()
        #ndegreeb = b.GetDegree()
        #hybra = a.GetHybridization()
        #hybrb = b.GetHybridization()
        #implVala = a.GetExplicitValence()
        implValb = b.GetExplicitValence()
        #aroma = a.GetIsAromatic()
        aromb = b.GetIsAromatic()
        #numHa = a.GetNumImplicitHs()
        numHb = b.GetNumImplicitHs()
        #inRinga = a.IsInRing()
        inRingb = b.IsInRing()
        jch_est = 500 * 1/(nbondsb)

    res = pd.Series({'atnb':atnb,'nbondsb':nbondsb, 'implValb':implValb,'aromb':aromb,'numHb':numHb,
                     'inRingb': inRingb, 'jch_est':jch_est})
    return (res)


def getMorganFP(row):
    radius_at1 = 1
    radius_at2 = 4  # otherwise too many descriptors
    radius_max = max(radius_at1, radius_at2)
    bits_atom1 = np.zeros((radius_max + 1,), dtype=np.int)
    bits_atom2 = np.zeros((radius_max + 1,), dtype=np.int)

    mol_str = row['molblock']
    mol  = sanitize_molecules(mol_str)
    if mol is None:
        print("WARNING: mol defect!")
    else:
        #print(Chem.MolToSmiles(Chem.RemoveHs(mol)))
        i = row['atom_index_0']
        j = row['atom_index_1']
        #at0 = mol.GetAtomWithIdx(i)
        #at1 = mol.GetAtomWithIdx(j)
        bi = {}
        fp = rdMolDescriptors.GetHashedMorganFingerprint(mol, radius=radius_max, nBits = 256, fromAtoms=[i,j], bitInfo=bi)
        for bit,values in bi.items():
            for val in values:
                idx, radius = val
                if idx==i:
                    bits_atom1[radius] =bit
                if idx==j:
                    bits_atom2[radius] = bit
    res = pd.Series({'mfp_rad0_at0': bits_atom1[1], 'mfp_rad0_at1': bits_atom2[0], 'mfp_rad1_at1':  bits_atom2[1],'mfp_rad2_at1':  bits_atom2[2], 'mfp_rad3_at1':  bits_atom2[3],'mfp_rad4_at1':  bits_atom2[4]})
    return(res)


def getPairFPAndAngles(row):
    #along path
    #radius
    #core charge
    #valence electrons
    if row['id'] % 10000 == 0:
        print(row['id'])
    mol_str = row['molblock']
    mol = sanitize_molecules(mol_str)
    if mol is None:
        mol_ok = False
        sanitization_ok = False
        path_len = 0
        score = 0
        c0 = 0
        c1 = 0
        directdist = 0.0
        path_mismatch = False
        path_too_long = False
        total_dist_type2 = 0.0
        angle = 0.0
        angle_dih1 = 0.0
        angle_dih2 = 0.0
        total_dist_type3 = 0.0
        dihedral = 0.0

    else:
        conf = mol.GetConformer(0)
        i = row['atom_index_0']
        j = row['atom_index_1']
        at0 = mol.GetAtomWithIdx(i)
        at1 = mol.GetAtomWithIdx(j)
        path = rdmolops.GetShortestPath(mol,i,j)

        #print(Pairs.ExplainPairScore(score))
        dm = Chem.Get3DDistanceMatrix(mol)

        path_len = len(path)
        #?rdkit.Chem.rdMolDescriptors.GetAtomPairFingerprint : https://www.rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html
        score = Pairs.pyScorePair(at0, at1, path_len)
        c0 = Pairs.Utils.GetAtomCode(at0)
        c1 = Pairs.Utils.GetAtomCode(at1)
        directdist = 0.0
        path_mismatch = False
        path_too_long = False
        total_dist_type2 = 0.0
        angle = 0.0
        angle_dih1 = 0.0
        angle_dih2 = 0.0
        total_dist_type3 = 0.0
        dihedral = 0.0

        if row['type_0'] != path_len-1:
            path_mismatch = True

        if path_len >1:
            directdist = dm[i, j]

        #angles
        if path_len >2:
            a = i
            for patom in path[1:]:
                b = patom
                total_dist_type2 += dm[a, b]
                a = patom
            angle = rdMolTransforms.GetAngleRad(conf,path[0],path[1],path[-1])

        #dihedrals
        if path_len == 4:
            for patom in path[1:]:
                b = patom
                total_dist_type3 += dm[a, b]
                a = patom
            dihedral = rdMolTransforms.GetDihedralRad(conf,path[0],path[1],path[2],path[3])
            angle_dih1 = rdMolTransforms.GetAngleRad(conf, path[0], path[1], path[2])
            angle_dih2 = rdMolTransforms.GetAngleRad(conf, path[1], path[2], path[3])

        # too long
        if path_len > 4:
            path_too_long = True
            for patom in path[1:]:
                b = patom
                total_dist_type3 += dm[a, b]
                a = patom
                dihedral = rdMolTransforms.GetDihedralRad(conf, path[0], path[1], path[2], path[-1])
                angle_dih1 = rdMolTransforms.GetAngleRad(conf, path[0], path[1], path[2])
                angle_dih2 = rdMolTransforms.GetAngleRad(conf, path[1], path[2], path[3])

    res = pd.Series({'score': score,'directdist': directdist, 'path_mismatch': path_mismatch, 'path_too_long': path_too_long,
                         'total_dist_type2': total_dist_type2, 'angle': angle, 'angle_dih1': angle_dih1, 'angle_dih2': angle_dih2, 'total_dist_type3': total_dist_type3, 'dihedral': dihedral})
    return(res)

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
    df_interaction.drop(['atom_index_at1','atom_index_at2'],axis=1,inplace=True)
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


def prepareDataset(seed = 42, nsamples = -1,storedata=True, selectType=None, quickload=None,keepID=False,  makeDistMat=False, makeTrainType=False, plotDist = False, makeDistMean=False, makeMolNameMean=False, bruteForceFeatures=False, bruteForceFeaturesPhysical=False,dropFeatures=None, keepFeatures=None, makeLabelEncode = False,getNeighbours=False, makeRDKitFeatures=False,makeRDKitFingerPrints=False, makeRDKitAtomFeatures=False, makeRDKitAnglesPairFP=False, coulombMatrix=False, useMulliken_acsf=False, oneHotenc=None, obCharges=False,makeSOAP=False, oof_fermi=False,load_oof_fermi=False, makeMAMBAfeatures=False, yukawaPotentials=False, cosineFeatures=False,distanceIsAllYouNeed=False,loadGibaFeatures=False, loadQM9Features=False, dropNonPhysicalFeatures=None, removeLowVariance=False, removeCorr=False,featureSelect=None, dimReduce=None):
    np.random.seed(seed)

    if storedata is not None:
        store = pd.HDFStore('./data/store.h5')
        #print("Opening previous data store with keys: %r"%(list(store.keys())))

    if quickload is not None:
        print("Loading previous dataset...")
        store2 = pd.HDFStore('./data/store.h5')
        Xtest = store2['Xtest']
        Xtrain = store2['Xtrain']
        cv_labels = store2['cv_labels']
        ytrain = store2['ytrain']
        if 'Xval' in store2 and not oof_fermi:
            Xval = store2['Xval']
            yval = store2['yval']
        else:
            Xval = None
            yval = None
        df_param = store2['parameters']
        #print("Loaded params: %s" % (df_param))
        act_params = inspect.signature(prepareDataset)
        df_act = pd.DataFrame(act_params.parameters.items())
        #print("Current params: %s"%(df_act))

        sample_weight = None
        #return only labels from list
        if isinstance(quickload,list):
            le = LabelEncoder()
            le.fit(j_list)
            test_mask = Xtest.type.isin(le.transform(quickload)).values
            Xtest = Xtest.loc[test_mask]
            train_mask = Xtrain.type.isin(le.transform(quickload)).values
            Xtrain = Xtrain.loc[train_mask]
            cv_labels = cv_labels.loc[train_mask]
            ytrain = ytrain.loc[train_mask]
            if Xval is not None:
                val_mask = Xval.type.isin(le.transform(quickload)).values
                Xval = Xval.loc[val_mask]
                yval = yval.loc[val_mask]

        print("Xtrain shape: %s %s" % (Xtrain.shape))
        return(Xtest, Xtrain, ytrain, cv_labels, None, Xval, yval)

    Xtrain = pd.read_csv(data_dir + r'train.csv')
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

    if oof_fermi:
        print("Changing target - using fermi c")
        Xtrain_fermi = pd.read_csv('./data/scalar_coupling_contributions.csv',
                                   usecols=['fc'])

        Xtrain = pd.merge(Xtrain, Xtrain_fermi, how='left',
                              left_on='id',
                              right_index=True)

        ytrain = Xtrain['fc']
        Xtrain.drop(['fc','scalar_coupling_constant'], axis=1, inplace=True)

    else:
        ytrain = Xtrain['scalar_coupling_constant']
        Xtrain = Xtrain.drop(['scalar_coupling_constant'], axis=1)

    print("Xtrain shape: %s %s" % (Xtrain.shape))
    print("Xtest  shape: %s %s " % (Xtest.shape))

    df_structures = pd.read_csv('./data//structures.csv')
    Xtrain = map_atom_info(Xtrain, df_structures, 0)
    Xtrain = map_atom_info(Xtrain, df_structures, 1)

    Xtest = map_atom_info(Xtest, df_structures, 0)
    Xtest = map_atom_info(Xtest, df_structures, 1)

    if makeDistMat:
        train_p_0 = Xtrain[['x_0', 'y_0', 'z_0']].values
        train_p_1 = Xtrain[['x_1', 'y_1', 'z_1']].values
        test_p_0 = Xtest[['x_0', 'y_0', 'z_0']].values
        test_p_1 = Xtest[['x_1', 'y_1', 'z_1']].values
        Xtrain['dist'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)
        Xtest['dist'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)

        Xtrain['dist_x'] = (Xtrain['x_0'] - Xtrain['x_1']) ** 2
        Xtest['dist_x'] = (Xtest['x_0'] - Xtest['x_1']) ** 2
        Xtrain['dist_y'] = (Xtrain['y_0'] - Xtrain['y_1']) ** 2
        Xtest['dist_y'] = (Xtest['y_0'] - Xtest['y_1']) ** 2
        Xtrain['dist_z'] = (Xtrain['z_0'] - Xtrain['z_1']) ** 2
        Xtest['dist_z'] = (Xtest['z_0'] - Xtest['z_1']) ** 2

    if makeTrainType:
        Xtrain['type_0'] = Xtrain['type'].apply(lambda x: x[0]).astype(int)
        Xtest['type_0'] = Xtest['type'].apply(lambda x: x[0]).astype(int)
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

    # be aware of overfitting here ->no, important features why?
    if makeMolNameMean:
        Xtrain[f'molecule_type_dist_mean'] = Xtrain.groupby(['molecule_name', 'type'])['dist'].transform('mean')
        Xtest[f'molecule_type_dist_mean'] = Xtest.groupby(['molecule_name', 'type'])['dist'].transform('mean')
        Xtrain[f'molecule_type_0_dist_mean'] = Xtrain.groupby(['molecule_name', 'type_0'])['dist'].transform('mean')
        Xtest[f'molecule_type_0_dist_mean'] = Xtest.groupby(['molecule_name', 'type_0'])['dist'].transform('mean')
        Xtrain[f'molecule_type_1_dist_mean'] = Xtrain.groupby(['molecule_name', 'type_1'])['dist'].transform('mean')
        Xtest[f'molecule_type_1_dist_mean'] = Xtest.groupby(['molecule_name', 'type_1'])['dist'].transform('mean')

        #Xtrain[f'molecule_type_dist_std'] = Xtrain.groupby(['molecule_name', 'type'])['dist'].transform('std')
        #Xtest[f'molecule_type_dist_std'] = Xtest.groupby(['molecule_name', 'type'])['dist'].transform('std')
        #Xtrain[f'molecule_type_0_dist_std'] = Xtrain.groupby(['molecule_name', 'type_0'])['dist'].transform('std')
        #Xtest[f'molecule_type_0_dist_std'] = Xtest.groupby(['molecule_name', 'type_0'])['dist'].transform('std')
        #Xtrain[f'molecule_type_1_dist_std'] = Xtrain.groupby(['molecule_name', 'type_1'])['dist'].transform('std')
        #Xtest[f'molecule_type_1_dist_std'] = Xtest.groupby(['molecule_name', 'type_1'])['dist'].transform('std')

    if load_oof_fermi:
        print("Loading fermi features...")
        train_oof_fermi = pd.read_csv('./data/oof_fermi_train.csv', index_col=0, header=None)
        test_oof_fermi = pd.read_csv('./data/oof_fermi_test.csv', index_col=0, header=None)
        Xtrain['oofermi'] = train_oof_fermi
        Xtest['oofermi'] = test_oof_fermi


    if bruteForceFeatures:
        print("Making bruteForceFeatures")
        Xtrain = create_brute_force_features_kernel(Xtrain)
        Xtest = create_brute_force_features_kernel(Xtest)

    if bruteForceFeaturesPhysical:
        print("Making mostly physicall bruteForceFeatures")
        Xtrain = create_brute_force_features(Xtrain)
        Xtest = create_brute_force_features(Xtest)

    if makeLabelEncode:
        for f in ['atom_0', 'atom_1',  'type_1', 'type']:
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
        print("Loading moldata.csv ...")
        # load moldata.csv
        # join features for each molecule
        moldata = pd.read_csv('moldata.csv',index_col='name', usecols = ['name']+descRDkitTop20)
        moldata = reduce_mem_usage(moldata)
        # moldata = removeLowVar(moldata, threshhold=1E-5)
        print("Mapping RDKit features...")
        Xall = pd.merge(Xall, moldata, how='left',
                      left_on=['molecule_name'],
                      right_index=True)
        Xall = reduce_mem_usage(Xall)
        print(Xall.head())
        #save data
        #Xall[rdkit_colnames].to_csv('rdkit_features.csv')
        print(Xall.columns)


    if makeRDKitFingerPrints:
        if not os.path.isfile('rdkit_morganfp.csv'):
            print("Loading RDKit molblock...")
            if not 'molblock' in Xall.columns:
                moldata = pd.read_csv('moldata_short.csv')
                print("Mapping RDKit mol string...")
                Xall, rdkit_colnames = map_rdkit_features(Xall, moldata)
            # compute some descriptors pair FP are bad
            #https://apassionatechie.wordpress.com/2017/12/27/create-multiple-pandas-dataframe-columns-from-applying-a-function-with-multiple-returns/
            print("Computing Morgan fingerprints...")
            df_tmp = Xall.apply(getMorganFP, axis=1)
            df_tmp.to_csv('rdkit_morganfp.csv')
        else:
            print("Loading RDkit morgan FP from file ...")
            df_tmp = pd.read_csv('rdkit_morganfp.csv', index_col=0)
            df_tmp = reduce_mem_usage(df_tmp)

        Xall = Xall.merge(df_tmp, left_index=True, right_index=True)
        print(Xall.head(20))
        Xall = reduce_mem_usage(Xall)

    if getNeighbours:
        if not os.path.isfile('neighbours.csv'):
            if not 'molblock' in Xall.columns:
                moldata = pd.read_csv('moldata_short.csv')
                print("Mapping RDKit mol string...")
                moldata = reduce_mem_usage(moldata)
                Xall, rdkit_colnames = map_rdkit_features(Xall, moldata)
                Xall = reduce_mem_usage(Xall)
                print("Computing direct neighbours indices...")
                df_tmp = Xall.apply(getBondNeighbors, axis=1)
                df_tmp.to_csv('neighbours.csv')
        else:
            print("Loading neighbor index from file ...")
            df_tmp = pd.read_csv('neighbours.csv', index_col=0)
        df_tmp = reduce_mem_usage(df_tmp)

        Xall = Xall.merge(df_tmp, left_index=True, right_index=True)
        print(Xall.head(20))

    if makeRDKitAtomFeatures:
        if not os.path.isfile('rdkit_atomic_features.csv'):
            if not 'molblock' in Xall.columns:
                print("Loading RDKit molblock...")
                moldata = pd.read_csv('moldata_short.csv')
                print("Mapping RDKit mol string...")
            Xall, rdkit_colnames = map_rdkit_features(Xall, moldata)
            print("Computing some atomic features...")
            df_tmp = Xall.apply(getAtomicFeatures, axis=1)
            df_tmp.to_csv('rdkit_atomic_features.csv')
        else:
            print("Loading RDkit atomic features from file ...")
            df_tmp = pd.read_csv('rdkit_atomic_features.csv', index_col=0)
            df_tmp = reduce_mem_usage(df_tmp)

        Xall = Xall.merge(df_tmp, left_index=True, right_index=True)
        #jch_est = 500 * 1 / (nbondsb)
        Xall['jch_est'] = 500 / Xall['nbondsb']
        #Xall['nbondsb_type_mean'] = Xall.groupby(['nbondsb', 'type'])['dist'].transform('mean')

        print(Xall.head())
        Xall = reduce_mem_usage(Xall)

    if makeRDKitAnglesPairFP:
        if not os.path.isfile('rdkit_anglesfp.csv'):
            print("Loading RDKit molblock...")
            if not 'molblock' in Xall.columns:
                moldata = pd.read_csv('moldata_short.csv')
                print("Mapping RDKit mol string...")
                Xall, rdkit_colnames = map_rdkit_features(Xall, moldata)
            Xall = reduce_mem_usage(Xall)
            print("Computing some new RDKit features...")
            df_tmp = Xall.apply(getPairFPAndAngles, axis=1)
            print("Reduce mem usage...")
            df_tmp = reduce_mem_usage(df_tmp)
            print("Saving new RDKit features...")
            df_tmp.to_csv('rdkit_anglesfp.csv')
        else:
            print("Loading Angles FP features from file ...")
            df_tmp = pd.read_csv('rdkit_anglesfp.csv',index_col=0)
            df_tmp = reduce_mem_usage(df_tmp)

        Xall = Xall.merge(df_tmp, left_index=True, right_index=True)
        Xall['jkarplus'] = Xall['angle'].apply(lambda x : 9.0*np.cos(x)**2)
        print(Xall.head(50))
        #print("molecules ok %d" % ((Xall['mol_ok'] == True).sum()))
        #print("molecules not ok %d"%((Xall['mol_ok'] == False).sum()))
        #print("molecules not sanizited %d"%((Xall['sanitization_ok'] == False).sum()))
        #print("path mismatch %d" % ((Xall['path_mismatch'] == True).sum()))
        #print("path too long %d" % ((Xall['path_too_long'] == True).sum()))
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
            first_molecules = Xall['molecule_name'].unique()[:Xall.shape[0]]

            df_interact = get_interaction_data_frame(df_distance.loc[df_distance['molecule_name'].isin(first_molecules)])
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

        Xall.drop(['atom_index_at1', 'atom_index_at2'], axis=1, inplace=True)



    if useMulliken_acsf:
        print("Loading Mullikan charges (test set estimated via wascf)")
        train_mul = pd.read_csv('/home/loschen/calc/amimanera/data/mulliken_charges.csv')
        test_mul = pd.read_csv('/home/loschen/calc/amimanera/data/mulliken_charges_test_set.csv')

        Xmulliken  = pd.concat([test_mul, train_mul], ignore_index=True)

        Xall = pd.merge(Xall, Xmulliken, how='left',
                        left_on=['molecule_name', 'atom_index_0'],
                        right_on=['molecule_name', 'atom_index'])
        Xall.rename({'mulliken_charge': 'mc_at0'}, axis=1, inplace=True)
        Xall.drop(['atom_index'], axis=1, inplace=True)

        Xall = pd.merge(Xall, Xmulliken, how='left',
                        left_on=['molecule_name', 'atom_index_1'],
                        right_on=['molecule_name', 'atom_index'])
        Xall.rename({'mulliken_charge': 'mc_at1'}, axis=1, inplace=True)
        Xall.drop(['atom_index'], axis=1, inplace=True)

        if 'nb1' in Xall.columns:
            nblist = ['nb1','nb2','nb3','nb4']
            for nb in nblist:
                Xall = pd.merge(Xall, Xmulliken, how='left',
                            left_on=['molecule_name', nb],
                            right_on=['molecule_name', 'atom_index'])
                Xall.rename({'mulliken_charge': f'mc_{nb}'}, axis=1, inplace=True)
                Xall.drop(['atom_index'], axis=1, inplace=True)
        print(Xall.head())

    if obCharges:
        # ob_methods = ["eem", "mmff94", "gasteiger", "qeq", "qtpie", "eem2015ha", "eem2015hm", "eem2015hn", "eem2015ba", "eem2015bm", "eem2015bn"]
        ob_methods = ["eem", "gasteiger", "qeq", "qtpie", "mmff94"]
        if not os.path.isfile("xall_ob_charges.csv"):
            print("Computing OB charges...")
            xall_molecules = Xall.molecule_name.unique()
            structures_idx = df_structures.set_index(["molecule_name"])
            xall_ob_charges = get_charges_df(xall_molecules,structures_idx = structures_idx, ob_methods = ob_methods)
            xall_ob_charges.to_csv("xall_ob_charges.csv")

        else:
            print("Loading OB charges...")
            xall_ob_charges = pd.read_csv("xall_ob_charges.csv", index_col=0)
            xall_ob_charges = xall_ob_charges[['molecule_name', 'atom_index']+ob_methods]

        Xall = pd.merge(Xall, xall_ob_charges, how='left',
                        left_on=['molecule_name', 'atom_index_0'],
                        right_on=['molecule_name', 'atom_index'],suffixes=('_at1', '_at2'))

        Xall = pd.merge(Xall, xall_ob_charges, how='left',
                        left_on=['molecule_name', 'atom_index_1'],
                        right_on=['molecule_name', 'atom_index'],suffixes=('_at1', '_at2'))

        Xall = reduce_mem_usage(Xall)
        Xall.drop(['atom_index_at1', 'atom_index_at2'], axis=1, inplace=True)

    #makeSOAP = False
    if makeSOAP:
        if not os.path.isfile("structures_soap.csv"):
            print("Creating SOAP features...")
            soap_desc = SOAP(species=["C", "H", "O", "N", "F"], rcut=6, nmax=3, lmax=2, crossover=False, periodic=False)
            structures_soap = getSOAPfeatures_new(df_structures,soap_desc)
            structures_soap.to_csv('structures_soap.csv')
        else:
            print("Loading SOAP features...")
            structures_soap = pd.read_csv('structures_soap.csv',index_col=0)

        print(structures_soap.head(40))
        structures_soap = reduce_mem_usage(structures_soap)
        Xall = reduce_mem_usage(Xall)

        print("Merging SOAP features...")
        Xall = pd.merge(Xall, structures_soap, how='left',
                        left_on=['molecule_name', 'atom_index_0'],
                        right_on=['molecule_name', 'atom_index'], suffixes=('_at1', '_at2'))



        Xall = pd.merge(Xall, structures_soap, how='left',
                        left_on=['molecule_name', 'atom_index_1'],
                        right_on=['molecule_name', 'atom_index'], suffixes=('_at1', '_at2'))

        print(Xall.head())
        Xall.drop(['atom_index_at1', 'atom_index_at2'], axis=1, inplace=True)



    makeSOAP_old = False
    if makeSOAP_old:
        if not os.path.isfile("Xsoap.csv"):
            # use structre df instead...!
            if not 'molblock' in Xall.columns:
                moldata = pd.read_csv('moldata_short.csv')
                print("Mapping RDKit mol string...")
            moldata = reduce_mem_usage(moldata)
            Xall =  reduce_mem_usage(moldata)
            moldata.set_index('name',inplace=True)
            Xall, rdkit_colnames = map_rdkit_features(Xall, moldata)
            soap_desc = SOAP(species=["C", "H", "O", "N", "F"], rcut=4, nmax=2, lmax=2, crossover=False, periodic=False)
            Xsoap = Xall.apply(getSOAPfeatures, axis=1,soap_desc=soap_desc)
            Xsoap.to_csv("Xsoap.gz",compression='gzip')
        else:
            Xsoap = pd.read_csv("Xsoap.gz", index_col=0,compression='gzip')

        Xall = Xall.merge(Xsoap, left_index=True, right_index=True)

    if yukawaPotentials:
        if not os.path.isfile("structures_yukawa.csv"):
            print("Creating yukawa potentials...")
            nuclear_charge = {'H': 1.0, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
            df_structures['nuclear_charge'] = [nuclear_charge[x] for x in df_structures['atom'].values]
            #https://www.kaggle.com/scaomath/no-memory-reduction-workflow-for-each-type-lb-1-28
            #https://www.kaggle.com/scaomath/parallelization-of-coulomb-yukawa-interaction
            #https://www.kaggle.com/gvyshnya/parallel-operations-over-a-pandas-df
            chunk_iter = df_structures.groupby(['molecule_name'])
            pool = mp.Pool(4)  # use 4 CPU cores
            funclist = []
            print("compute_all_yukawa ...")
            for df in tqdm(chunk_iter):
                # process each data frame, parallel over each moleculue...?
                f = pool.apply_async(compute_all_yukawa, [df[1]])
                funclist.append(f)

            print("collecting results ...")
            result = []
            for f in tqdm(funclist):
                result.append(f.get())

            # combine chunks with transformed data into a single training set
            structures_yukawa = pd.concat(result)
            structures_yukawa.head(10)
            structures_yukawa.to_csv('structures_yukawa.csv')
        else:
            print("Loading yukawa potentials...")
            structures_yukawa = pd.read_csv('structures_yukawa.csv',index_col=0)

        structures_yukawa = pd.concat([df_structures, structures_yukawa], axis=1)
        structures_yukawa.drop(['atom','x','y','z'],axis=1,inplace=True)
        structures_yukawa.head(10)


        Xall = pd.merge(Xall, structures_yukawa, how='left',
                      left_on=['molecule_name', 'atom_index_1'],
                      right_on=['molecule_name', 'atom_index'])
        del structures_yukawa
        Xall.drop(['atom_index'],axis=1,inplace=True)

    if loadGibaFeatures:
        global giba_columns
        print("Loading Giba features")
        print("Before:", Xall.shape)
        print("Before:", list(Xall.columns))

        train_giba = pd.read_csv('./data/train_giba.csv',usecols=giba_columns)
        train_giba = reduce_mem_usage(train_giba)
        test_giba = pd.read_csv('./data/test_giba.csv',usecols=giba_columns)
        test_giba = reduce_mem_usage(test_giba)
        all_giba = pd.concat([test_giba, train_giba], ignore_index=True)
        Xall = reduce_mem_usage(Xall)

        all_giba_columns = list(set(all_giba.columns).difference(set(Xall.columns)))
        Xall = pd.concat((Xall, all_giba[all_giba_columns]), axis=1)
        print("After:",Xall.shape)
        print("After:", list(Xall.columns))
        del train_giba
        del test_giba
        del all_giba

    if cosineFeatures:
        print("Now cosine features...")
        #https://www.kaggle.com/scaomath/no-memory-reduction-workflow-for-each-type-lb-1-28
        Xall = reduce_mem_usage(Xall)
        Xall = find_closest_atom(Xall)
        Xall = add_cos_features(Xall,df_structures)
        Xall = Xall.fillna(0.0)
        print(Xall.head())

    if distanceIsAllYouNeed:
        #https://www.kaggle.com/criskiev/distance-is-all-you-need-lb-1-481
        print("distanceIsAllYouNeed...")


    if makeMAMBAfeatures:
        print("Finally MAMBA features...")

        moldata = pd.read_csv('moldata_short.csv')
        moldata = moldata.loc[moldata.name.isin(Xall['molecule_name'].unique())]
        df_mol = pd.DataFrame()
        print("Total %d unique molecules!"%(len(Xall['molecule_name'].unique())))
        for index, row in moldata.iterrows():
            if index%1000==0: print(index)
            df_tmp = getMAMBAFeatures(row)
            df_mol = df_mol.append(df_tmp, ignore_index=True)
        df_mol.to_csv('mamba.csv')

        Xall = pd.merge(Xall, df_mol, how='left',
                      left_on=['molecule_name', 'atom_index_0', 'atom_index_1'],
                      right_on=['molecule_name', 'id1','id2'])

        print(df_mol)

    if loadQM9Features:
        print("Loading QM9 features...")
        df_qm9 = pkl.load(open("./data/data.covs.pickle", "rb"))
        df_qm9.drop(['atom_index_0','atom_index_1','type','molecule_name','scalar_coupling_constant','U', 'G', 'H',
               'mulliken_mean', 'r2', 'U0','linear'],axis=1,inplace=True)
        Xall = pd.merge(Xall, df_qm9, how='left',
                        left_on=['id'],
                        right_on=['id'], suffixes=('_orig', '_qm9'))
        print(Xall.head())


    #Xtrain = Xall[len(Xtest.index):]
    #Xtest = Xall[:len(Xtest.index)]

    if oneHotenc is not None:
        print("1-0 Encoding categoricals...", oneHotenc)
        #Xall = pd.concat([Xtest, Xtrain], ignore_index=True)
        for col in oneHotenc:
            if col in Xall.columns:
                uv = np.unique(Xall[col].values)
                Xall[col] = Xall[col].astype('category')
                print("%d unique values for col: %s %r"%(len(uv),col, uv))
                encoder = OneHotEncoder(categories='auto',sparse=False)
                X_onehot = pd.DataFrame(encoder.fit_transform(Xall[[col]].values))
                X_onehot.columns = [col + "_" + str(column) for column in X_onehot.columns]
                print("One-hot-encoding of %r...new shape: %r" % (col, X_onehot.shape))
                Xall.drop([col], axis=1, inplace=True)
                Xall = pd.concat([Xall, X_onehot], axis=1)
                print("One-hot-encoding final shape:", Xall.shape)

    #split beforehaned to save memory during dropping
    Xtrain = Xall[len(Xtest.index):]
    Xtest = Xall[:len(Xtest.index)]

    if keepFeatures is not None:
        print("Keeping only subset of features: "+str(keepFeatures))
        keepFeatures.append('molecule_name')
        dropcols = [col for col in Xall.columns if col not in keepFeatures]
        print("Dropping %d  features: %s"%(len(dropcols),dropcols))
        Xtrain.drop(dropcols, axis=1, errors='ignore', inplace=True)
        Xtest.drop(dropcols, axis=1, errors='ignore', inplace=True)

    if dropFeatures is not None:
        print("Try dropping: ", dropFeatures)

        if not keepID:
            dropFeatures = dropFeatures + ['id']
        else:
            print('WARNING: Keeping ID')
        Xtrain.drop(dropFeatures, axis=1, errors='ignore', inplace=True)
        Xtest.drop(dropFeatures, axis=1, errors='ignore', inplace=True)
        print("Features left: ", list(Xall.columns))


    if dropNonPhysicalFeatures is not None:
        for col in Xall.columns:
            for feat in dropNonPhysicalFeatures:
                if feat in col:
                    print("Dropping: %s"%(col))
                    Xall.drop(col, axis=1, errors='ignore', inplace=True)

    Xall = pd.concat([Xtest, Xtrain], ignore_index=True)

    if removeLowVariance:
        print("remove low var using threshhold...")
        Xall = removeLowVar(Xall, threshhold=1E-10)

    if removeCorr:
        Xall = removeCorrelations(Xall, None, None, 0.995)

    if featureSelect is not None:
        #transformer = GenericUnivariateSelect(f_regression, 'k_best', param=featureSelect)
        #transformer.fit(Xall[len(Xtest.index):].values, ytrain)
        #Xall = transformer.transform(Xall)
        # Set a minimum threshold of 0.25
        transformer = SelectFromModel(LassoCV(cv=3), threshold=0.25)
        transformer.fit(Xall[len(Xtest.index):].values, ytrain)
        Xall = transformer.transform(Xall)


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

    sample_weight = None

    #create validation set
    if not oof_fermi:
        Xtrain, Xval, ytrain, yval = train_test_split(Xtrain, ytrain, test_size = 0.25, random_state = 42, shuffle=True)
        Xval.set_index('molecule_name', inplace=True)
        Xval = reduce_mem_usage(Xval)
    else:
        Xval = None
        yval = None

    #cv_lables
    cv_labels = Xtrain['molecule_name']
    Xtrain = Xtrain.drop(['molecule_name'], errors='ignore', axis=1)
    Xtest = Xtest.drop(['molecule_name'], errors='ignore', axis=1)

    Xtrain = reduce_mem_usage(Xtrain)
    Xtest = reduce_mem_usage(Xtest)

    if Xtrain.isnull().sum().sum()>0:
        print("Xtrain NaN values:")
        print(Xtrain.isnull().sum())
        time.sleep(5)

    if storedata:
        clist = list(Xtest.columns)
        dlist = [x for x in clist if clist.count(x) > 1]
        print("Storing data... - Duplicate columns: %r"%(dlist))
        store['Xtest'] = Xtest
        store['Xtrain'] = Xtrain
        store['cv_labels'] = cv_labels
        store['ytrain'] = ytrain
        if Xval is not None:
            store['Xval'] = Xval
            store['yval'] = yval

        param_dict = inspect.signature(prepareDataset)
        store['parameters'] = pd.DataFrame(param_dict.parameters.items())
        store.close()


    if isinstance(selectType,list) and selectType is not None:
        le = LabelEncoder()
        le.fit(j_list)
        Xtest = Xtest.loc[Xtest.type.isin(le.transform(selectType))]
        train_mask = Xtrain.type.isin(le.transform(selectType)).values
        Xtrain = Xtrain.loc[train_mask]
        cv_labels = cv_labels.loc[train_mask]
        ytrain = ytrain.loc[train_mask]
        val_mask = Xval.type.isin(le.transform(selectType)).values
        Xval = Xval.loc[val_mask]
        yval = yval.loc[val_mask]

    if Xval is not None:
        print("Xval shape: %s %s" % (Xval.shape))
    print("Xtrain shape: %s %s" % (Xtrain.shape))
    print("Xtrain columns: %r" % (list(Xtrain.columns)))
    print("Finished feature preparation...\n")

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


def makePredictions(model=None, Xtest=None, filename='nmr',jtype=None, verbose=True):
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

    if jtype is not None:
        tmp_file = './data/tmp.csv'
        tmp = pd.read_csv(tmp_file,index_col=0)
        if preds.shape[0] != tmp.loc[tmp.type_int.isin(jtype),:].shape[0]:
            print("ERRROR: Length preds: %d Length tmp: %d"%(preds.shape[0],tmp.loc[tmp.type_int.isin(jtype),:].shape[0]))
        tmp.loc[tmp.type_int.isin(jtype),'scalar_coupling_constant'] = preds
        tmp.to_csv(tmp_file)
        print("Saving submission to tmp file %s - NaN elements left: %d\n"%(tmp_file,tmp['scalar_coupling_constant'].isna().sum()))

    else:
        if sub.shape[0] != preds.shape[0]:
            print("Prediction do not have the right shape:")
            print(preds.shape)
        else:
            sub['scalar_coupling_constant'] = preds
            sub.to_csv(filename, index=False)
            print(f'Saved submission to: {filename}')
            print(sub.head())
            logging.info(f'Saved submission to: {filename}')

def reset_tmpfile():
    test = pd.read_csv('./data/test.csv')
    j_list = ['1JHC', '1JHN', '2JHC', '2JHH', '2JHN', '3JHC', '3JHH', '3JHN']
    le = LabelEncoder()
    le.fit(j_list)
    le.transform(['2JHC'])
    le.inverse_transform([1])
    test['type_int'] = le.transform(test['type'])
    test['scalar_coupling_constant'] = np.nan
    print('Re-setting tmp file. Length: %d' % (test.shape[0]))
    test[['id', 'type', 'type_int', 'scalar_coupling_constant']].to_csv('./data/tmp.csv')

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
    #lgbm parameters: https://sites.google.com/view/lauraepp/parameters
    # CNN: https://www.kaggle.com/c/champs-scalar-coupling/discussion/101091#latest-583418
    #https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629
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
    #work on smallest set as it improves the group mae most
    # use neigbour index atoms as descriptors...!
    # onehotencode on natoms
    # GCNN:
    # https://www.kaggle.com/fnands/1-mpnn
    # https://www.kaggle.com/fnands/makegraphinput

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
    # -1    : OOF score: 0.3408                       [8 sep models] [default features]
    # -1    : OOF score: -0.1701 VAL score: -0.2311   [8 sep models] ob charges & makeRDKitAtomFeatures]
    # -1    : OOF score: -0.4618 VAL score: -0.5131  [ 8 sep models brute force kernel]
    # -1    : OOF score: 0.0788 - VAL score: 0.0384  [ 8 sep models brute force physical]
    # -1    : OOF score: -0.0771 - VAL score: -0.1388 [Coulomb matrix]
    # -1    : OOF score: -0.4293 - VAL score: -0.4988 [obCharges	makeRDKitAtomFeatures	makeRDKitAnglesPairFP]
    # -1    : 8 [obCharges	makeRDKitAtomFeatures	makeRDKitAnglesPairFP] 3 groups
    #https://www.kaggle.com/adrianoavelar/eachtype

    #-1
    result_dict = {}
    unphysical = ['x_0','y_0','z_0','x_1','y_1','z_1']

    data_params = {
        'quickload':None, #
        'seed' : 42,
        'nsamples' : -1,
        'storedata' : True,
        'plotDist': False,
        'makeDistMat' : True, #X
        'makeTrainType' : False,
        'makeDistMean' : False,
        'makeMolNameMean' : False,
        'bruteForceFeatures' : False,
        'bruteForceFeaturesPhysical': False, # -> check it out
        'makeLabelEncode' : True,
        'getNeighbours' : True, #X
        'makeRDKitFeatures': False,
        'makeRDKitFingerPrints': False,
        'makeRDKitAtomFeatures': True,
        'makeRDKitAnglesPairFP': True,
        'makeMAMBAfeatures': False,
        'useMulliken_acsf': False, #X
        'coulombMatrix': False, #X
        'obCharges' : True, #X
        'yukawaPotentials': False,
        'cosineFeatures' : False,
        'loadGibaFeatures' : True,
        'makeSOAP' : False,
        'distanceIsAllYouNeed': True,
        'oof_fermi' : True,
        'load_oof_fermi' : False,
        'loadQM9Features': False, # not so good :-(
        'oneHotenc' : None,
        'removeLowVariance' : False,
        'keepFeatures' : None,#desc1JHCTOP,
        #'dropFeatures' : ['atom_index_0','atom_index_1','molblock','atomcodea','atomcodeb','score','path_len','mol_ok','sanitization_ok','path_too_long','path_mismatch'],
        'dropFeatures': dropFeatures,
        'featureSelect': None,
        #'dropNonPhysicalFeatures': ['atom_index'],
        'dimReduce' : None
    }

    Xtest, Xtrain, ytrain, cv_labels, _, Xval, yval = prepareDataset(**data_params)

    print("Xtrain shape: %s %s" % (Xtrain.shape))
    print("Xtrain columns: %r"%(list(Xtrain.columns)))

    n_fold = 5
    lgb_params = {'num_leaves': 128,
              'min_child_samples': 79,
              'objective': 'regression',
              'max_depth': 12,
              'learning_rate': 0.15,#<-change to 0.1
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
    if Xval is not None:
        Xval_f = pd.DataFrame({'ind': list(Xval.index), 'type': Xval['type'].values, 'prediction': [0] * len(Xval), 'target': yval.values})
    Xtest_f = pd.DataFrame({'ind': list(Xtest.index), 'type': Xtest['type'].values, 'prediction': [0] * len(Xtest)})
    if 'atom_index_0' in Xtrain.columns:
        Xtrain_f['atom_index_0'] = Xtrain['atom_index_0'].values
        Xtrain_f['atom_index_1'] = Xtrain['atom_index_1'].values
        if Xval is not None:
            Xval_f['atom_index_0'] = Xval['atom_index_0'].values
            Xval_f['atom_index_1'] = Xval['atom_index_1'].values

    #cv = KFold(n_splits=n_fold, shuffle=True, random_state=42)
    cv = GroupKFold(n_splits=n_fold)
    model = lgb.LGBMRegressor(**lgb_params)
    # model = RidgeCV()
    logging.info('model:  %r ' % (str(model)))

    fit_types = [['1JHC'], ['1JHN'], ['2JHC'], ['2JHH'], ['2JHN'], ['3JHC'], ['3JHH'], ['3JHN']]
    #fit_types = [['1JHC', '1JHN'], ['2JHC', '2JHH','2JHN'], ['3JHC', '3JHH', '3JHN']]

    gridsearch = False
    #fit_types = [['2JHC'],['3JHC']]
    #fit_types = [['1JHC'], ['2JHC'], ['3JHC']]
    #fit_types = [['1JHC'],['2JHC'],['3JHC']]
    scores = []
    for i,t  in enumerate(fit_types):
        print("\n>>%d - Coupling-type %r<<"%(i,t))

        data_params['quickload'] = t
        Xtest, Xtrain, ytrain, cv_labels, _, Xval, yval = prepareDataset(**data_params)
        if Xval is  None:
            print("Xval is None!")

        print("Xtest shape:    %s %s" % (Xtest.shape))
        print("Xtrain shape:   %s %s" % (Xtrain.shape))
        print("Xtrain columns: %r" % (list(Xtrain.columns)))
        #result_dict = buildXvalModel(model, Xtrain, ytrain, sample_weight=None, class_names=None, refit=False, cv=cv)
        #eval = lgb.cv(lgb_params,lgb.Dataset(Xtrain, ytrain),  nfold=5,stratified=False, early_stopping_rounds=200, verbose_eval=100,seed=5,show_stdv=True)
        if gridsearch:
            parameters = {'n_estimators': [5000], 'max_depth': [15,20], 'learning_rate': [0.1,0.2],'subsample': [0.5,1.0]}
            model = makeGridSearch(model, Xtrain, ytrain, n_jobs=1, refit=True, cv=3, scoring='neg_mean_absolute_error', random_iter=-1, parameters=parameters)

        else:
            result_dict = train_model_regression(X=Xtrain, X_test=Xtest, y=ytrain, params=lgb_params, folds=cv, cv_labels=cv_labels,model_type='lgb', eval_metric='group_mae',plot_feature_importance=True,verbose=1000, early_stopping_rounds=200)

            le = LabelEncoder()
            le.fit(j_list)

            #Xtest.loc[Xtest.type.isin(le.transform(quickload))]
            Xtrain_f.loc[Xtrain_f.type.isin(le.transform(t)), 'oof'] = result_dict['oof']
            Xtest_f.loc[Xtest_f.type.isin(le.transform(t)), 'prediction'] = result_dict['prediction']

            doVal = False
            valscore = 0
            if isinstance(Xval,pd.DataFrame) and doVal:
                if isinstance(model,lgb.LGBMRegressor):
                    early_stopping_rounds = 200
                    model.fit(Xtrain,ytrain,verbose=1000,early_stopping_rounds=early_stopping_rounds,eval_metric='mse',eval_set=[(Xtrain, ytrain), (Xval, yval)],)
                else:
                    model.fit(Xtrain, ytrain)
                ypred = model.predict(Xval)
                valscore = group_mean_log_mae(yval, ypred, Xval['type'])
                print(">>Validation score: %6.4f"%(valscore))
                #plot_types(yval, ypred, Xval['type'].astype(int),t)
                #Xval_f.loc[Xval_f['type'] == type_int, 'prediction'] = ypred
                Xval_f.loc[Xval_f.type.isin(le.transform(t)), 'prediction'] = ypred
                Xval_f['SDEV'] = (Xval_f['prediction'] - Xval_f['target'])**2
                print(Xval_f.loc[Xval_f.type.isin(le.transform(t))].sort_values(axis=0,by=['type','SDEV'],ascending=[True,False]).head(100))

            #logging.info('cv:     %r ' % (result_dict['cv']))
            #logging.info('scores: %r '%(result_dict['scores']))
            logging.info('<scores>: %r ' % (np.mean(result_dict['scores'])))
            scores.append([t,Xtrain.shape[0],Xtrain.shape[1],np.mean(result_dict['scores']),valscore])
            if 'feature_importance' in result_dict.keys():
                logging.info(result_dict['feature_importance'])
                logging.info(list(result_dict['feature_importance'].index))

    df_scores = pd.DataFrame(scores,columns=['type','nsamples','nfeatures','oob_score','val_score'])
    print(df_scores.head(8))
    logging.info(df_scores.head(8))

    score_oob = group_mean_log_mae(Xtrain_f['target'], Xtrain_f['oof'], Xtrain_f['type'])
    if Xval is not None:
        score_val = group_mean_log_mae(Xval_f['target'], Xval_f['prediction'], Xval_f['type'])
    else:
        score_val =0
    print("OOF score: %6.4f - VAL score: %6.4f" % (score_oob,score_val))
    logging.info("OOF score: %6.4f - VAL score: %6.4f" % (score_oob,score_val))

    if 'oof_fermi' in data_params.keys():
        print("Saving OOF fermi features...")
        print(Xtrain_f.head())
        Xtrain_f['oof'].to_csv('./data/oof_fermi_train.csv')
        Xtest_f['prediction'].to_csv('./data/oof_fermi_test.csv')

    for i, t in enumerate(fit_types):
        for sub_t in t:
            plot_types(Xtrain_f['target'], Xtrain_f['oof'], Xtrain_f['type'], sub_t)

    makePredictions(None,Xtest_f,filename='submissions/nmr')

    plt.show()

if __name__ == "__main__":
    logging.basicConfig(filename='log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')
    #createSDFileViaOpenBabel()
    #constructMolecularDataFrame()
    #analyzeDataSet(plotHist=False, plotGraph=False)
    main()



