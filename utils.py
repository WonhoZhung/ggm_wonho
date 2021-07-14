import torch
import torch.nn as nn

import os
import math
import copy
import graph
import numpy as np

from rdkit import Chem


STOP = None
ATOM_TYPES = ['C', 'N', 'O', 'F', 'P', 'S', 'CL', 'BR', STOP]
ATOMIC_NUM = [6, 7, 8, 9, 15, 16, 17, 35]
BOND_TYPES = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, \
              Chem.BondType.TRIPLE, STOP]

NUM_ATOM_TYPES = len(ATOM_TYPES)
NUM_BOND_TYPES = len(BOND_TYPES)

ATOM_COLORS = ['black', 'tab:blue', 'tab:red', 'limegreen', \
               'tab:purple', 'tab:olive', 'dimgrey', 'brown']
BOND_COLORS = ['dimgrey', 'navy', 'darkgreen']

ATOM_COLOR_DICT = {ATOM_TYPES[i]:ATOM_COLORS[i] for i in range(NUM_ATOM_TYPES-1)}
BOND_COLOR_DICT = {BOND_TYPES[i]:BOND_COLORS[i] for i in range(NUM_BOND_TYPES-1)}


# settings for using GPU
def dic_to_device(dic, device):
    for dic_key, dic_value in dic.items():
        if isinstance(dic_value, torch.Tensor):
            dic_value = dic_value.to(device)
            dic[dic_key] = dic_value
    
    return dic
        
def set_cuda_visible_device(ngpus):
    import subprocess
    import os
    import numpy as np
    empty = []
    if ngpus>0:
        fn = f'/tmp/empty_gpu_check_{np.random.randint(0,10000000,1)[0]}'
        for i in range(4):
            os.system(f'nvidia-smi -i {i} | grep "No running" | wc -l > {fn}')
            with open(fn) as f:
                out = int(f.read())
            if int(out)==1:
                empty.append(i)
            if len(empty)==ngpus: break
        if len(empty)<ngpus:
            print ('avaliable gpus are less than required', len(empty), ngpus)
            exit(-1)
        os.system(f'rm -f {fn}')        
    
    cmd = ''
    for i in range(ngpus):        
        cmd+=str(empty[i])+','

    return cmd

def stat_cuda(msg):
    print("--", msg)
    print("allocated: %dM, max allocated: %dM, cached: %dM, max cached: %dM" %
          (torch.cuda.memory_allocated() / 1024 / 1024,
           torch.cuda.max_memory_allocated() / 1024 / 1024,
           torch.cuda.memory_reserved() / 1024 / 1024,
           torch.cuda.max_memory_reserved() / 1024 / 1024))

def initialize_model(model, device, load_save_file=False):
    if load_save_file:
        if device.type=='cpu':
            save_file_dict = torch.load(load_save_file, map_location='cpu')
            new_save_file_dict = dict()
            for k in save_file_dict:
                new_key = k.replace("module.", "")
                new_save_file_dict[new_key] = save_file_dict[k]
            model.load_state_dict(new_save_file_dict, strict=False)
        else:
            save_file_dict = torch.load(load_save_file)
            new_save_file_dict = dict()
            for k in save_file_dict:
                new_key = k.replace("module.", "")
                new_save_file_dict[new_key] = save_file_dict[k]
            model.load_state_dict(new_save_file_dict, strict=False)
    else:
        for param in model.parameters():
            if not param.requires_grad: 
                continue
            if param.dim() == 1:
                #print("muyaho")
                nn.init.normal_(param, 0, 1)
            else:
                #print("muyaho-muyaho")
                nn.init.xavier_normal_(param)

    if torch.cuda.device_count() > 1:
        print("---> Using", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)
    return model

def get_dataset_dataloader(batch_size, num_workers, **kwargs):
    from torch.utils.data import DataLoader
    from dataset import QM9Dataset, tensor_collate_fn
    train_dataset = QM9Dataset(mode='train', **kwargs)
    train_dataloader = DataLoader(train_dataset,
                                   batch_size,
                                   num_workers=num_workers,
                                   collate_fn=tensor_collate_fn,
                                   shuffle=True)

    test_dataset = QM9Dataset(mode='test', **kwargs)
    test_dataloader = DataLoader(test_dataset,
                                  batch_size,
                                  num_workers=num_workers,
                                  collate_fn=tensor_collate_fn,
                                  shuffle=False)
    return train_dataset, train_dataloader, test_dataset, test_dataloader

# write result file
def write_result(fn, true_dict, pred_dict):
    lines = []
    for k, v in true_dict.items():
        assert pred_dict.get(k) is not None
        lines.append(f"{k}\t{float(v):.3f}\t{float(pred_dict[k]):.3f}\n")

    with open(fn, 'w') as w: w.writelines(lines)
    return

def prob_to_one_hot(vector):
    N = len(vector)
    new_vector = vector.new_zeros(vector.shape)
    new_vector[vector.argmax()] = 1
    return new_vector

def int_to_one_hot(x, length, device=None):
    return torch.eye(length)[x].to(device)


if __name__ == '__main__':
    
    pass
