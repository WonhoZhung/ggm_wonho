import torch
import torch.nn as nn
from torch.utils.data import Dataset

import rdkit
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem.AllChem import GetAdjacencyMatrix, CalcNumRotatableBonds

import glob
import os
import pickle
import random
import numpy as np
from utils import ATOM_TYPES, BOND_TYPES
from graph import make_graphs_from_mol


def get_period_group(a):                                                       
    PERIODIC_TABLE = """                                                       
H,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,HE
LI,BE,1,1,1,1,1,1,1,1,1,1,B,C,N,O,F,NE
NA,MG,1,1,1,1,1,1,1,1,1,1,AL,SI,P,S,CL,AR
K,CA,SC,TI,V,CR,MN,FE,CO,NI,CU,ZN,GA,GE,AS,SE,BR,KR
RB,SR,Y,ZR,NB,MO,TC,RU,RH,PD,AG,CD,IN,SN,SB,TE,I,XE
CS,BA,LU,HF,TA,W,RE,OS,IR,PT,AU,HG,TL,PB,BI,PO,AT,RN
    """
    pt = dict()
    for i, per in enumerate(PERIODIC_TABLE.split()):
        for j, ele in enumerate(per.split(',')):
            pt[ele] = (i, j)
    period, group = pt[a.GetSymbol().upper()]
    return one_of_k_encoding(period, [0, 1, 2, 3, 4, 5]) + \
           one_of_k_encoding(group, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def get_atom_feature(atom):
    return torch.FloatTensor(
            one_of_k_encoding(atom.GetSymbol().upper(), ATOM_TYPES)
            ) # --> total 9

def get_bond_feature(bond):
    return torch.FloatTensor(
            one_of_k_encoding(bond.GetBondType(), BOND_TYPES)
            ) # --> total 4

def mol_to_sample(mol, args):
    """
    [input]
    
    mol: mol object
    args: arguments

    [output]
    
    sample: dictionary{
        whole: graph object
        scaff: graph object
    }
    """

    scaff, whole = make_graphs_from_mol(mol, args.max_num_nodes, False)

    sample = {
            "whole": whole,
            "scaff": scaff
    }

    return sample


class GraphDataset(Dataset):

    def __init__(self, args, mode="train"):
        super().__init__()

        self.args = args
        
        self.data_dir = args.data_dir
        self.key_dir = args.key_dir
        with open(self.key_dir+f"{mode}_keys.pkl", 'rb') as f: 
            self.key_list = pickle.load(f)

    def __len__(self):
        return len(self.key_list)

    def __getitem__(self, idx):
        
        key = self.key_list[idx]
        try: 
            with open(self.data_dir+key, 'rb') as f: 
                scaff, whole = pickle.load(f)
            sample = {"scaff": scaff, "whole": whole}
        except Exception as e: 
            print(key, e); exit()
        sample['key'] = key
        return sample


# def check_dimension(tensors):
#     size = []
#     for tensor in tensors:
#         if isinstance(tensor, np.ndarray):
#             size.append(tensor.shape)
#         else:
#             size.append(0)
#     size = np.asarray(size)
# 
#     return np.max(size, 0)
# 
# def collate_tensor(tensor, max_tensor, batch_idx):
#     if isinstance(tensor, np.ndarray):
#         dims = tensor.shape
#         max_dims = max_tensor.shape
#         slice_list = tuple([slice(0, dim) for dim in dims])
#         slice_list = [slice(batch_idx, batch_idx + 1), *slice_list]
#         max_tensor[tuple(slice_list)] = tensor
#     elif isinstance(tensor, str):
#         max_tensor[batch_idx] = tensor
#     else:
#         max_tensor[batch_idx] = tensor
# 
#     return max_tensor
# 
# def tensor_collate_fn(batch):
#     batch_items = [it for e in batch for it in e.items()]
#     dim_dict = dict()
#     total_key, total_value = list(zip(*batch_items))
#     batch_size = len(batch)
#     n_element = int(len(batch_items) / batch_size)
#     total_key = total_key[0:n_element]
#     for i, k in enumerate(total_key):
#         value_list = [v for j, v in enumerate(total_value) if j % n_element == i]
#         if isinstance(value_list[0], np.ndarray):
#             dim_dict[k] = np.zeros(np.array(
#                 [batch_size, *check_dimension(value_list)])
#             )
#         elif isinstance(value_list[0], str):
#             dim_dict[k] = ["" for _ in range(batch_size)]
#         else:
#             dim_dict[k] = np.zeros((batch_size,))
# 
#     ret_dict = {}
#     for j in range(batch_size):
#         if batch[j] == None: continue
#         keys = []
#         for key, value in dim_dict.items():
#             value = collate_tensor(batch[j][key], value, j)
#             if not isinstance(value, list):
#                 value = torch.from_numpy(value).float()
#             ret_dict[key] = value
# 
#     return ret_dict


if __name__ == "__main__":

    pass
