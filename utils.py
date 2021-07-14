import torch
import torch.nn as nn

import os
import math
import copy
import graph
import numpy as np

from rdkit import Chem

__all__ = ['dic_to_device', 'set_cuda_visible_device', 'stat_cuda', \
           'initialize_model', 'get_dataset_dataloader', 'write_result' \
           'check_equivariance']


STOP = None
ATOM_TYPES = ['C', 'N', 'O', 'F', 'P', 'S', 'CL', 'BR', STOP]
BOND_TYPES = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, \
              Chem.rdchem.BondType.TRIPLE, STOP]

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

# visualizing tensor
def print_tensor(tensor):
    print(tensor.shape)
    m, n = tensor.shape
    for i in range(m):
        for j in range(n):
            print(f"{tensor[i][j]:.1f}", end=', ')
        print('\n')
    return

def draw_tensor(tensor, filename='tmp', scale=1):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    tensor = tensor.cpu().numpy()

    fig, ax = plt.subplots()
    im = ax.imshow(tensor, cmap="Greys", vmin=0, vmax=scale)
    cbar = ax.figure.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.savefig(filename)
    plt.close()
    return

def draw_graph_3D(graph, msg="", fn=None):                                                     
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D                                     


    fig = plt.figure()                                                          
    ax = fig.gca(projection='3d')                                               

    vh = np.array(graph.get_node_feature())
    vr = np.array(graph.get_node_coord())
    eh = np.array(graph.get_ordered_edge()[0])
    e_dict = graph.get_edge_dict()
    num_atoms = graph.get_num_nodes()
    num_edges = graph.get_num_edges()

    for a in range(num_atoms):
        ax.scatter(*vr[a], color=ATOM_COLOR_DICT[ATOM_TYPES[vh[a].argmax()]])

    seen = []
    for (i, j), b in e_dict.items():
        if b in seen: continue
        e_vector = vr[[i,j],:].T                                             
        ax.plot(*e_vector, color=BOND_COLOR_DICT[BOND_TYPES[eh[b].argmax()]])
        seen.append(b)

    x_min, x_max = min(vr[:num_atoms,0]), max(vr[:num_atoms,0])
    y_min, y_max = min(vr[:num_atoms,1]), max(vr[:num_atoms,1])
    z_min, z_max = min(vr[:num_atoms,2]), max(vr[:num_atoms,2])
    delta = max([x_max-x_min, y_max-y_min, z_max-z_min])/2
    center = [(x_max+x_min)/2, (y_max+y_min)/2, (z_max+z_min)/2]

    ax.set_xlim(center[0]-delta, center[0]+delta)
    ax.set_ylim(center[1]-delta, center[1]+delta)
    ax.set_zlim(center[2]-delta, center[2]+delta)
    # ax.view_init(45, 45)
    ax.text(
        -10, 10, 0, msg, \
        horizontalalignment='left', \
        verticalalignment='top', \
        transform=ax.transAxes
    )
    
    plt.savefig(fn)                                                             
    plt.close()                                                                 
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
    
    # sdf_fn = "/home/wonho/work/data/PDBbind_v2019/general-set/1bcu/1bcu_ligand.sdf"
    # v_max = 30
    # graph = make_graph_from_sdf(sdf_fn, v_max)
    # draw_graph_3D(graph, msg="1bcu_ligand", fn="1bcu.png")
