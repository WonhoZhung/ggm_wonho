from multiprocessing import Pool
from rdkit import Chem
from graph import make_graphs_from_mol

import pickle


data_dir = "data/id_smiles.txt"
data_dir2 = "data/data/"

with open(data_dir, 'r') as f: lines = [l.strip() for l in f.readlines()][:1000]
print(len(lines))

def run(l):
    split = l.split()
    key = split[0]
    smiles = split[1]
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        scaff, whole = make_graphs_from_mol(mol, 30)
        assert scaff is not None
        assert whole is not None
        assert scaff.get_num_nodes() < whole.get_num_nodes()
    except Exception as e:
        return

    with open(data_dir2+key, 'wb') as w: pickle.dump((scaff, whole), w)

    return


pool = Pool(8)
r = pool.map_async(run, lines)
r.wait()
pool.close()
pool.join()




