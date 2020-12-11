import numpy as np
import sys
from rdkit import Chem
from sklearn.model_selection import train_test_split
import os

def adj_k(adj, k):
    ret = adj
    for i in range(0, k - 1):
        ret = np.dot(ret, adj)

    return convertAdj(ret)


def convertAdj(adj):
    dim = len(adj)
    a = adj.flatten()
    b = np.zeros(dim * dim)
    c = (np.ones(dim * dim) - np.equal(a, b)).astype('float64')
    d = c.reshape((dim, dim))

    return d


def convertToGraph(smiles_list, k):
    adj = []
    features = []
    maxNumAtoms = 50
    target = []
    if len(smiles_list[0].split(','))==2:
        for i in smiles_list:
            if i.split(',')[0]=='smiles':
                continue
            try:
                iMol = Chem.MolFromSmiles(i.split(',')[0])
                # Adj
                iAdjTmp = Chem.rdmolops.GetAdjacencyMatrix(iMol)
                # Feature
                if (iAdjTmp.shape[0] <= maxNumAtoms):
                    # Feature-preprocessing
                    iFeature = np.zeros((maxNumAtoms, 58))
                    iFeatureTmp = []
                    for atom in iMol.GetAtoms():
                        iFeatureTmp.append(atom_feature(atom))  ### atom features only
                    iFeature[0:len(iFeatureTmp), 0:58] = iFeatureTmp  ### 0 padding for feature-set
                    features.append(iFeature)

                    # Adj-preprocessing
                    iAdj = np.zeros((maxNumAtoms, maxNumAtoms))
                    iAdj[0:len(iFeatureTmp), 0:len(iFeatureTmp)] = iAdjTmp + np.eye(len(iFeatureTmp))
                    adj.append(adj_k(np.asarray(iAdj), k))
                    target.append((i.split(',')[1][:-1]))
            except:
                continue
        features = np.asarray(features)
        target = np.asarray(target)
        # target = [i[0] for i in np.asarray(target)]
        target = [float(i) for i in np.asarray(target)]

    else:
        fail=0
        for i in smiles_list:
            if i.split(',')[0]=='smiles':
                continue
            try:
                iMol = Chem.MolFromSmiles(i.split(',')[0])
                # Adj
                iAdjTmp = Chem.rdmolops.GetAdjacencyMatrix(iMol)
                # Feature
                if (iAdjTmp.shape[0] <= maxNumAtoms):
                    # Feature-preprocessing
                    iFeature = np.zeros((maxNumAtoms, 58))
                    iFeatureTmp = []
                    for atom in iMol.GetAtoms():
                        iFeatureTmp.append(atom_feature(atom))  ### atom features only
                    iFeature[0:len(iFeatureTmp), 0:58] = iFeatureTmp  ### 0 padding for feature-set
                    features.append(iFeature)

                    # Adj-preprocessing
                    iAdj = np.zeros((maxNumAtoms, maxNumAtoms))
                    iAdj[0:len(iFeatureTmp), 0:len(iFeatureTmp)] = iAdjTmp + np.eye(len(iFeatureTmp))
                    adj.append(adj_k(np.asarray(iAdj), k))
                    targets_list = []
                    for k in range(len(i.split(','))):
                        if k==0:
                            continue
                        if k==(len(i.split(','))-1):
                            targets_list.append((i.split(',')[k][:-1]))
                        else:
                            targets_list.append((i.split(',')[k]))
                    target.append([targets_list])
            except:
                fail = fail+1
        features = np.asarray(features)
        target = [i[0] for i in np.asarray(target)]
        print('fail:',fail)
    return adj, features, target

def convertToGraph_add_smiles(smiles_list, k):
    adj = []
    features = []
    maxNumAtoms = 50
    target = []
    smiles_set = []
    if len(smiles_list[0].split(','))==2:
        for i in smiles_list:
            if i.split(',')[0]=='smiles':
                continue
            try:
                iMol = Chem.MolFromSmiles(i.split(',')[0])
                # Adj
                iAdjTmp = Chem.rdmolops.GetAdjacencyMatrix(iMol)
                # Feature
                if (iAdjTmp.shape[0] <= maxNumAtoms):
                    # Feature-preprocessing
                    iFeature = np.zeros((maxNumAtoms, 58))
                    iFeatureTmp = []
                    for atom in iMol.GetAtoms():
                        iFeatureTmp.append(atom_feature(atom))  ### atom features only
                    iFeature[0:len(iFeatureTmp), 0:58] = iFeatureTmp  ### 0 padding for feature-set
                    features.append(iFeature)

                    # Adj-preprocessing
                    iAdj = np.zeros((maxNumAtoms, maxNumAtoms))
                    iAdj[0:len(iFeatureTmp), 0:len(iFeatureTmp)] = iAdjTmp + np.eye(len(iFeatureTmp))
                    adj.append(adj_k(np.asarray(iAdj), k))
                    target.append((i.split(',')[1][:-1]))
                    smiles_set.append(i.split(',')[0])
            except:
                continue
        features = np.asarray(features)
        target = np.asarray(target)
        # target = [i[0] for i in np.asarray(target)]
        target = [float(i) for i in np.asarray(target)]

    else:
        fail=0
        for i in smiles_list:
            if i.split(',')[0]=='smiles':
                continue
            try:
                iMol = Chem.MolFromSmiles(i.split(',')[0])
                # Adj
                iAdjTmp = Chem.rdmolops.GetAdjacencyMatrix(iMol)
                # Feature
                if (iAdjTmp.shape[0] <= maxNumAtoms):
                    # Feature-preprocessing
                    iFeature = np.zeros((maxNumAtoms, 58))
                    iFeatureTmp = []
                    for atom in iMol.GetAtoms():
                        iFeatureTmp.append(atom_feature(atom))  ### atom features only
                    iFeature[0:len(iFeatureTmp), 0:58] = iFeatureTmp  ### 0 padding for feature-set
                    features.append(iFeature)

                    # Adj-preprocessing
                    iAdj = np.zeros((maxNumAtoms, maxNumAtoms))
                    iAdj[0:len(iFeatureTmp), 0:len(iFeatureTmp)] = iAdjTmp + np.eye(len(iFeatureTmp))
                    adj.append(adj_k(np.asarray(iAdj), k))
                    targets_list = []
                    for k in range(len(i.split(','))):
                        if k==0:
                            continue
                        if k==(len(i.split(','))-1):
                            targets_list.append((i.split(',')[k][:-1]))
                        else:
                            targets_list.append((i.split(',')[k]))
                    target.append([targets_list])
                    smiles_set.append(i.split(',')[0])
            except:
                fail = fail+1
        features = np.asarray(features)
        target = [i[0] for i in np.asarray(target)]
        print('fail:',fail)
    return adj, features, target,smiles_set


def atom_feature(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'H', 'Si', 'P', 'Cl', 'Br',
                                           'Li', 'Na', 'K', 'Mg', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',
                                           'V', 'Tl', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
                                           'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'Mn', 'Cr', 'Pt', 'Hg', 'Pb']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    [atom.GetIsAromatic()])  # (40, 6, 5, 6, 1)


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        return list(map(lambda s: x - 1 == s, allowable_set))
        # raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    # print list((map(lambda s: x == s, allowable_set)))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def get_target(args):
    import pandas as pd
    smiles_f = open('./' + args.dataset + '/train.csv')
    smiles_list = smiles_f.readlines()
    train, test = train_test_split(smiles_list, test_size=0.2)

    smiles_train = []
    for i in train:
        iMol = Chem.MolFromSmiles(i.split(',')[0])
        # Adj
        iAdjTmp = Chem.rdmolops.GetAdjacencyMatrix(iMol)
        # Feature
        if (iAdjTmp.shape[0] <= 50):
            smiles_train.append((i.split(',')[1][:-2]))

    smiles_test = []
    for i in test:
        iMol = Chem.MolFromSmiles(i.split(',')[0])
        # Adj
        iAdjTmp = Chem.rdmolops.GetAdjacencyMatrix(iMol)
        # Feature
        if (iAdjTmp.shape[0] <= 50):
            smiles_test.append((i.split(',')[1][:-2]))
    smiles_train = np.float32(pd.DataFrame(smiles_train))
    smiles_test = np.float32(pd.DataFrame(smiles_test))
    return smiles_train,smiles_test



if __name__=="__main__":

    dbName = 'benchmark_molnet/molnet_random_qm8_r'
    # dbName = sys.argv[1]
    k = 1  # neighbor distance

    smiles_t = open('./' + dbName + '/train.csv')
    smiles_v = open('./' + dbName + '/val.csv')
    smiles_train = smiles_t.readlines()
    smiles_val = smiles_v.readlines()
    # smiles_train, smiles_val = train_test_split(smiles_list, test_size=0.1)
    print("train_number:",len(smiles_train))
    print("test_number:",len(smiles_val))


    if not os.path.exists(dbName + '/adj'):
        os.mkdir(dbName + '/adj')
    if not os.path.exists(dbName + '/features'):
        os.mkdir(dbName + '/features')
    train_adj, train_features, train_target = convertToGraph(smiles_train, 1)
    val_adj, val_features, val_target = convertToGraph(smiles_val, 1)
    np.save(dbName + '/adj/' + 'train.npy', train_adj)
    np.save(dbName + '/features/' + 'train.npy', train_features)
    np.save(dbName + '/train_target.npy', train_target)
    np.save(dbName + '/adj/' + 'val.npy', val_adj)
    np.save(dbName + '/features/' + 'val.npy', val_features)
    np.save(dbName + '/val_target.npy', val_target)

    smiles_f = open('./' + dbName + '/test.csv')
    smiles_list = smiles_f.readlines()
    test_adj, test_features, test_target = convertToGraph(smiles_list, 1)
    np.save(dbName + '/adj/' + 'test.npy', test_adj)
    np.save(dbName + '/features/' + 'test.npy', test_features)
    np.save(dbName + '/test_target.npy', test_target)

def decrease_learning_rate(optimizer, decrease_by=0.01):
    """Multiplies the learning rate of the optimizer by 1 - decrease_by"""
    for param_group in optimizer.param_groups:
        param_group['lr'] *= (1 - decrease_by)

