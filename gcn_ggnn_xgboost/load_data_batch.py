
import numpy as np
import torch
import torch.nn as nn
from data_deal import convertToGraph,convertToGraph_add_smiles
from torch.utils.data import Dataset
import os
from rdkit.Chem import AllChem
import pandas as pd

class ToxicDataset(Dataset):
    def __init__(self, x, y,z):
        super(ToxicDataset, self).__init__()
        self.x = x
        self.y = y
        self.z = z
    def __len__(self):
        return len(self.z)
    def __getitem__(self, idx):
        return (torch.from_numpy(np.float32(self.x[idx])), torch.from_numpy(np.float32(self.y[idx])),torch.from_numpy(np.float32(self.z[idx])))

class newToxicDataset(Dataset):
    def __init__(self, x, y,z):
        super(newToxicDataset, self).__init__()
        self.x = x
        self.y = y
        self.z = z
    def __len__(self):
        return len(self.z)
    def __getitem__(self, idx):
        return (torch.from_numpy(np.float32(self.x[idx])), torch.from_numpy(np.float32(self.y[idx])),torch.from_numpy(self.z[idx]))


def loadInputs_train(args):
    adj = np.load(args.dataset+'/adj/'+'train.npy')
    features = np.load(args.dataset+'/features/'+ 'train.npy')
    y_train = np.load(args.dataset + '/train_target.npy')
    y_train = y_train[:, None]
    return features, adj ,y_train

def loadInputs_feature_smiles(args,names):
    adj = np.load(args.dataset+'/adj/'+ names + '.npy')
    features = np.load(args.dataset+'/features/'+ names + '.npy')
    y = np.load(args.dataset + '/' +names + '_target.npy')
    y = y[:, None]
    smiles = np.load(args.dataset + '/' + names + '_smiles.npy')
    return features, adj ,y ,smiles

def loadInputs_val(args):
    adj = np.load(args.dataset+'/adj/'+'val.npy')
    features = np.load(args.dataset+'/features/'+ 'val.npy')
    y_val = np.load(args.dataset + '/val_target.npy')
    y_val = y_val[:, None]
    return features, adj, y_val

def loadInputs_test(args):
    adj = np.load(args.dataset+'/adj/'+'test.npy')
    features = np.load(args.dataset+'/features/'+ 'test.npy')
    y_test = np.load(args.dataset + '/test_target.npy')
    y_test = y_test[:, None]
    return features, adj, y_test

def ZW6(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol,3,nBits=256).ToBitString()
def get_mol(smile):
    try:
        N = len(smile)
        nump = []
        for i in range(0, N):
            mol = AllChem.MolFromSmiles(smile[i])
            nump.append(mol)
        return nump
    except:
        nump = []
        mol = AllChem.MolFromSmiles(smile)
        nump.append(mol)
        return nump

def get_morgan_feature(smile):
    mol = get_mol(smile)
    data = []
    for i in range(len(mol)):
        try:
            data.append([smile[i], ZW6(mol[i])])
        except:
            continue
    jak_feature = pd.DataFrame(data, columns=['smiles','ZW6'])
    num_frame6 = []
    for i in range(len(jak_feature['ZW6'])):
        num_frame6.append([x for x in jak_feature['ZW6'][i]])
    jak_zw6 = pd.DataFrame(num_frame6,dtype=np.float)
    return jak_zw6


def load_data(args):
    smiles_t = open(args.dataset + '/train.csv')
    smiles_v = open(args.dataset + '/val.csv')
    smiles_train = smiles_t.readlines()
    smiles_val = smiles_v.readlines()
    # smiles_train, smiles_val = train_test_split(smiles_list, test_size=0.1)
    print("train_number:", len(smiles_train))
    print("test_number:", len(smiles_val))
    if not os.path.exists(args.dataset + '/adj'):
        os.mkdir(args.dataset + '/adj')
    if not os.path.exists(args.dataset + '/features'):
        os.mkdir(args.dataset + '/features')
    train_adj, train_features, train_target ,train_smiles = convertToGraph_add_smiles(smiles_train, 1)
    val_adj, val_features, val_target ,val_smiles = convertToGraph_add_smiles(smiles_val, 1)
    np.save(args.dataset + '/adj/' + 'train.npy', train_adj)
    np.save(args.dataset + '/features/' + 'train.npy', train_features)
    np.save(args.dataset + '/train_target.npy', train_target)
    np.save(args.dataset + '/adj/' + 'val.npy', val_adj)
    np.save(args.dataset + '/features/' + 'val.npy', val_features)
    np.save(args.dataset + '/val_target.npy', val_target)
    smiles_f = open(args.dataset + '/test.csv')
    smiles_list = smiles_f.readlines()
    test_adj, test_features, test_target ,test_smiles = convertToGraph_add_smiles(smiles_list, 1)
    np.save(args.dataset + '/adj/' + 'test.npy', test_adj)
    np.save(args.dataset + '/features/' + 'test.npy', test_features)
    np.save(args.dataset + '/test_target.npy', test_target)
    np.save(args.dataset + '/train_smiles.npy',train_smiles)
    np.save(args.dataset + '/val_smiles.npy',val_smiles)
    np.save(args.dataset + '/test_smiles.npy',test_smiles)
