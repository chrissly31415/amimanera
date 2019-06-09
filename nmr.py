
import os

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

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from artgor_utils import train_model_regression


data_dir = r'./data/'
struct_dir = r'./data/structures/'

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

def prepareDataset(nsamples = -1, makeDistMat=True, makeTrainType=True, plotDist = False, makeDistMean=True, makeMolNameMean=True, dropFeatures=True):
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv(data_dir + r'test.csv')
    structures = pd.read_csv('./data//structures.csv')
    train = map_atom_info(train, structures, 0)
    train = map_atom_info(train, structures, 1)

    if nsamples != -1:
        if isinstance(nsamples, str) and 'shuffle' in nsamples:
            print("Shuffle train data...")
            rows = np.random.choice(len(train.index), size=len(train.index), replace=False)
        else:
            rows = np.random.choice(len(train.index), size=nsamples, replace=False)

        print("unique rows: %6.2f" % (float(np.unique(rows).shape[0]) / float(rows.shape[0])))
        train = train.iloc[rows, :]

    test = map_atom_info(test, structures, 0)
    test = map_atom_info(test, structures, 1)

    train_p_0 = train[['x_0', 'y_0', 'z_0']].values
    train_p_1 = train[['x_1', 'y_1', 'z_1']].values
    test_p_0 = test[['x_0', 'y_0', 'z_0']].values
    test_p_1 = test[['x_1', 'y_1', 'z_1']].values

    if makeDistMat:
        train['dist'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)
        test['dist'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)
        train['dist_x'] = (train['x_0'] - train['x_1']) ** 2
        test['dist_x'] = (test['x_0'] - test['x_1']) ** 2
        train['dist_y'] = (train['y_0'] - train['y_1']) ** 2
        test['dist_y'] = (test['y_0'] - test['y_1']) ** 2
        train['dist_z'] = (train['z_0'] - train['z_1']) ** 2
        test['dist_z'] = (test['z_0'] - test['z_1']) ** 2

    if makeTrainType:
        train['type_0'] = train['type'].apply(lambda x: x[0])
        test['type_0'] = test['type'].apply(lambda x: x[0])
        train['type_1'] = train['type'].apply(lambda x: x[1:])
        test['type_1'] = test['type'].apply(lambda x: x[1:])

    if plotDist:
        fig, ax = plt.subplots(figsize=(18, 8))
        plt.subplot(1, 2, 1)
        plt.hist(train['dist'], bins=20)
        plt.title('Basic dist_speedup histogram')
        plt.subplot(1, 2, 2)
        sns.violinplot(x='type', y='dist', data=train)
        plt.title('Violinplot of dist_speedup by type')

    if makeDistMean:
        train['dist_to_type_mean'] = train['dist'] / train.groupby('type')['dist'].transform('mean')
        test['dist_to_type_mean'] = test['dist'] / test.groupby('type')['dist'].transform('mean')

        train['dist_to_type_0_mean'] = train['dist'] / train.groupby('type_0')['dist'].transform('mean')
        test['dist_to_type_0_mean'] = test['dist'] / test.groupby('type_0')['dist'].transform('mean')

        train['dist_to_type_1_mean'] = train['dist'] / train.groupby('type_1')['dist'].transform('mean')
        test['dist_to_type_1_mean'] = test['dist'] / test.groupby('type_1')['dist'].transform('mean')

    # be aware of overfitting here
    if makeMolNameMean:
        train[f'molecule_type_dist_mean'] = train.groupby(['molecule_name', 'type'])['dist'].transform('mean')
        test[f'molecule_type_dist_mean'] = test.groupby(['molecule_name', 'type'])['dist'].transform('mean')

    for f in ['atom_0', 'atom_1', 'type_0', 'type_1', 'type']:
        lbl = LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))

    if dropFeatures is not None:
        for col in dropFeatures:
            if col in train.columns:
                print("Dropping: ", col)
                train.drop([col], axis=1, inplace=True)
                test.drop([col], axis=1, inplace=True)

    #prepare training
    X = train.drop(['id', 'molecule_name', 'scalar_coupling_constant'], axis=1)
    y = train['scalar_coupling_constant']
    Xtest = test.drop(['id', 'molecule_name'], axis=1)

    return (X,Xtest,y)


def bondPerceptionWithOB(struct_dir = './data/structures',ext='.xyz',outfile='nmr.sdf'):
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


def constructMolecularFeatures(infile='nmr.sdf'):
    #There are 130775 distinct molecules in data.
    from rdkit.Chem import AllChem as Chem
    from rdkit.Chem.AtomPairs import Pairs
    from rdkit.Chem import Descriptors
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem import rdmolops
    #https: // www.rdkit.org / docs / source / rdkit.Chem.AtomPairs.Pairs.html
    suppl = Chem.SDMolSupplier(infile, removeHs=False, sanitize=False)
    nok = 0
    ntotal = 0
    data = []
    for i, mol in enumerate(suppl):
        ntotal +=1
        if mol is not None:
            name = mol.GetProp('_Name')

            # pairFps = Pairs.GetAtomPairFingerprint(mol)
            # pair_dict = pairFps.GetNonzeroElements()
            # print(pair_dict)
            # for fk in pair_dict.keys():
            #     print(fk,Pairs.ExplainPairScore(fk))

            mol_str = Chem.MolToMolBlock(mol)
            smiles = Chem.MolToSmiles(mol)
            try:
                rdmolops.SanitizeMol(mol)
                molweight = Descriptors.MolWt(mol)
                nelecs = Descriptors.NumValenceElectrons(mol)
                nrings = rdMolDescriptors.CalcNumRings(mol)
                nrot = rdMolDescriptors.CalcNumRotatableBonds(mol)
                nok += 1

            except ValueError:
                molweight = -999
                nelecs = -999
                nrings = -999
                nrot = -999


        data.append([name, mol_str, smiles, molweight, nelecs, nrot, nrings])
        if i%1000==0:
            print(f"Parsing mol {i} {name}")

    print(f"There are {ntotal} molecules.")
    print(f"There are {nok} fine molecules.")
    print(f"There are {130775-ntotal} missing molecules.")

    df = pd.DataFrame(data, columns = ['name','molblock','smiles','MolWt','NumValenceElectrons','NumRotatableBonds','nrings'])
    print(df.info())
    print(df.describe())
    print(df.head(20))
    df.to_csv('moldata.csv',index=False)


def getPairFingerPrint(mol,idx1,idx2):
    pass


def construct3DMolecularFeatures(infile='./data/nmr.sdf'):
    #There are 130775 distinct molecules in data.
    from rdkit.Chem import AllChem as Chem


    suppl = Chem.SDMolSupplier(infile, removeHs=False, sanitize=False)
    nok = 0
    ntotal = 0
    for i, mol in enumerate(suppl):
        ntotal +=1
        if mol is not None:
            nok +=1
        if i%1000==0:
            print(f"Parsing mol {i}")

    print(f"There are {ntotal} molecules.")
    print(f"There are {nok} fine molecules.")
    print(f"There are {130775-ntotal} missing molecules.")

def constructAtomicFeatures(infile='./data/nmr.sdf'):
    pass


def main():
    #check https://www.kaggle.com/asauve/training-set-molecule-visualization
    #https://www.kaggle.com/artgor/brute-force-feature-engineering
    #analyzeDataSet()
    data_params = {
        'nsamples' : -1,
        'makeDistMat' : True,
        'makeTrainType' : True,
        'plotDist' : False,
        'makeDistMean' : True,
        'makeMolNameMean' : True,
        'dropFeatures' : ['atom_index_0','atom_index_1']
    }
    X, Xtest, y = prepareDataset(**data_params)

    n_fold = 3
    folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)
    params = {'num_leaves': 128,
              'min_child_samples': 79,
              'objective': 'regression',
              'max_depth': 12,
              'learning_rate': 0.3,
              "boosting_type": "gbdt",
              "subsample_freq": 1,
              "subsample": 1.0,
              "bagging_seed": 11,
              "metric": 'mae',
              "verbosity": -1,
              'reg_alpha': 0.1,
              'reg_lambda': 0.3,
              'colsample_bytree': 1.0
              }

    print(params)
    result_dict_lgb = train_model_regression(X=X, X_test=Xtest, y=y, params=params, folds=folds,
                                                          model_type='lgb', eval_metric='group_mae',
                                                          plot_feature_importance=True,
                                                          verbose=1000, early_stopping_rounds=200, n_estimators=15000)

    sub = pd.read_csv('./data/sample_submission.csv')
    sub['scalar_coupling_constant'] = result_dict_lgb['prediction']
    sub.to_csv('submission_nmr.csv', index=False)
    sub.head()

    plt.show()

if __name__ == "__main__":
    print(os.getcwd())
    #analyzeDataSet(plotHist=False, plotGraph=False)
    #main()
    #constructMolecularFeatures()
    bondPerceptionWithOB()

