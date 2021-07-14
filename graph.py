import torch
import utils


class Graph2D(object):

    def __init__(
            self,
            num_node_features,
            num_edge_features,
            max_num_nodes,
            self_loop=False
            ):

        self.num_node_features = int(num_node_features)
        self.num_edge_features = int(num_edge_features)
        self.max_num_nodes = int(max_num_nodes)
        self.max_num_edges = int(max_num_nodes * (max_num_nodes - 1) / 2)

        self.node_feature_mat = torch.zeros(max_num_nodes, num_node_features)
        self.edge_feature_mat = \
                torch.zeros(max_num_nodes, max_num_nodes, num_edge_features)
        if self_loop:
            self.adjacency_mat = torch.eye(max_num_nodes)
        else:
            self.adjacency_mat = torch.zeros(max_num_nodes, max_num_nodes)

        self.edge_idx_dict = {} # key: (node_idx1, node_idx2), value: edge_idx

        self.num_nodes = 0
        self.num_edges = 0

        self.self_loop = self_loop

        self.device = None

    def __repr__(
            self
            ):
        return f"node: {self.get_num_nodes()}, edge: {self.get_num_edges()}"

    def __eq__(
            self,
            other
            ):
        return torch.equal(self.node_feature_mat, other.node_feature_mat) and \
               torch.equal(self.edge_feature_mat, other.edge_feature_mat) and \
               torch.equal(self.adjacency_mat, other.adjacency_mat)

    def get_num_nodes(
            self
            ):
        return self.num_nodes

    def get_num_edges(
            self
            ):
        return self.num_edges

    def get_num_node_feature(
            self
            ):
        return self.num_node_features

    def get_num_edge_feature(
            self
            ):
        return self.num_edge_features

    def get_node_feature(
            self
            ):
        return self.node_feature_mat

    def get_edge_feature(
            self
            ):
        return self.edge_feature_mat

    def get_edge_dict(
            self
            ):
        return self.edge_idx_dict

    def get_adjacency_matrix(
            self
            ):
        return self.adjacency_mat
    
    def get_max_num_nodes(
            self
            ):
        return self.max_num_nodes

    def get_num_neighbors(
            self,
            idx
            ):
        if self.self_loop: return int(sum(self.adjacency_mat[idx])) - 1
        return int(sum(self.adjacency_mat[idx]))

    def get_edges(
            self,
            idx
            ):
        """
        return list of edges (tuple: (idx2, edge_feature)) linked to node idx
        """
        edges = []
        for i, e in enumerate(self.edge_feature_mat[idx]):
            if i == idx: continue
            if self.is_edge_between(idx, i):
                edges.append((i, self.edge_feature_mat[idx, i]))
        return edges

    def get_device(
            self
            ):
        return self.device

    def add_node(
            self,
            feature,
            ):
        assert self.num_nodes < self.max_num_nodes + 1

        self.node_feature_mat[self.num_nodes] = feature
        self.num_nodes += 1
        return feature

    def add_edge_between(
            self,
            feature,
            idx1,
            idx2,
            ):
        assert self.num_nodes > 2
        assert self.num_edges < self.max_num_edges + 1
        assert not self.is_edge_between(idx1, idx2)
        
        self.adjacency_mat[idx1, idx2] = 1
        self.adjacency_mat[idx2, idx1] = 1
        self.edge_feature_mat[idx1, idx2] = feature
        self.edge_feature_mat[idx2, idx1] = feature
        self.edge_idx_dict[(idx1, idx2)] = self.num_edges
        self.edge_idx_dict[(idx2, idx1)] = self.num_edges
        self.num_edges += 1
        return feature

    def is_edge_between(
            self,
            idx1,
            idx2
            ):
        assert idx1 < self.max_num_nodes
        assert idx2 < self.max_num_nodes
        return bool(self.adjacency_mat[idx1, idx2])

    def get_node_feature_with_idx(
            self,
            idx
            ):
        if idx > self.num_nodes - 1: return
        return self.node_feature_mat[idx]

    def get_edge_feature_with_idx(
            self,
            idx1,
            idx2
            ):
        if idx1 > self.num_nodes - 1: return None
        if idx2 > self.num_nodes - 1: return None
        return self.edge_feature_mat[(idx1, idx2)]

    def set_node_with_idx(
            self,
            feature,
            idx
            ):
        assert idx < self.num_nodes
        self.node_feature_mat[idx] = torch.FloatTensor(feature)
        return torch.FloatTensor(feature)

    def set_node_feature_mat(
            self,
            node_feature_mat,
            ):
        #assert feature_mat.shape == self.node_feature_mat.shape
        self.node_feature_mat = node_feature_mat
        return node_feature_mat

    def set_edge_feature_mat(
            self,
            edge_feature_mat
            ):
        #assert edge_feature_mat.shape == self.edge_feature_mat.shape
        self.edge_feature_mat = edge_feature_mat
        return edge_feature_mat

    def set_adjacency_mat(
            self,
            adjacency_mat
            ):
        #assert adjacency_mat.shape == self.adjacency_mat.shape
        self.adjacency_mat = adjacency_mat
        return adjacency_mat

    def get_valid_node(
            self
            ):
        shape = (self.max_num_nodes, 1)
        valid_n = self.adjacency_mat.new_zeros(shape)
        valid_n[:self.num_nodes, :] = 1
        return valid_n

    def get_valid_edge(
            self
            ):
        shape = (self.max_num_nodes, self.max_num_nodes, 1)
        valid_e = self.adjacency_mat.new_zeros(shape)
        valid_e[:self.num_nodes, :self.num_nodes, :] = 1
        return valid_e

    def get_ordered_edge(
            self
            ):
        shape = (self.max_num_edges, self.num_edge_features)
        ordered_edge = self.edge_feature_mat.new_zeros(shape)
        valid_edge = self.edge_feature_mat.new_zeros((self.max_num_edges, 1))
        for i, (idx1, idx2) in enumerate(self.edge_idx_dict.keys()):
            if not i%2 == 1: continue
            ordered_edge[i//2] = self.edge_feature_mat[idx1, idx2]
            valid_edge[i//2] = 1
        return ordered_edge, valid_edge

    def graph_to_device(
            self,
            device
            ):
        self.node_feature_mat = self.node_feature_mat.to(device)
        self.edge_feature_mat = self.edge_feature_mat.to(device)
        self.adjacency_mat = self.adjacency_mat.to(device)

        self.device = device
        return

    def _visualize(
            self,
            msg="",
            fn=None
            ):
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D                                     

        fig = plt.figure()                                                          
        ax = fig.gca(projection='3d')                                               

        vh = np.array(self.get_node_feature())
        vr = np.array(self.get_node_coord())
        eh = np.array(self.get_ordered_edge()[0])
        e_dict = self.get_edge_dict()
        num_nodes = self.get_num_nodes()
        num_edges = self.get_num_edges()

        for a in range(num_nodes):
            ax.scatter(*vr[a], \
                    color=utils.ATOM_COLOR_DICT[utils.ATOM_TYPES[vh[a].argmax()]])

        seen = []
        for (i, j), b in e_dict.items():
            if b in seen: continue
            e_vector = vr[[i,j],:].T                                             
            ax.plot(*e_vector, \
                    color=utils.BOND_COLOR_DICT[utils.BOND_TYPES[eh[b].argmax()]])
            seen.append(b)

        x_min, x_max = min(vr[:num_nodes,0]), max(vr[:num_nodes,0])
        y_min, y_max = min(vr[:num_nodes,1]), max(vr[:num_nodes,1])
        z_min, z_max = min(vr[:num_nodes,2]), max(vr[:num_nodes,2])
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
        
        if fn is not None:
            plt.savefig(fn)                                                             
        plt.close()                                                                 
        return fig

    def animate(
            self,
            fn=None
            ):
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import imageio
        from PIL import Image

        # TODO
        fig = self.visualize()
        img = Image.frombytes('RGB', fig.canvas.get_width_height(), \
                fig.canvas.tostring_rgb())

        # imageio.mimsave(fn, imgs, fps=5)
        return


def is_same_graph(graph1, graph2, hard=False):
    if graph1.get_num_nodes() != graph2.get_num_nodes(): return False
    if graph1.get_num_edges() != graph2.get_num_edges(): return False
    if not torch.equal(graph1.adjacency_mat, graph2.adjacency_mat): return False

    if hard:
        return graph1 == graph2

    # Check node feature
    for i in range(graph1.get_num_nodes()):
        if graph1.node_feature_mat[i].argmax(-1) != \
                graph2.node_feature_mat[i].argmax(-1):
                    return False

    # Check edge feature
    for i in range(graph1.get_num_nodes()):
        for j in range(i, graph1.get_num_nodes()):
            if not graph1.is_edge_between(i, j): continue
            if graph1.edge_feature_mat[i,j].argmax(-1) != \
                    graph2.edge_feature_mat[i,j].argmax(-1):
                        return False

    return True

def stack_graph(graph_list, mask=False, clone=True):

    g = graph_list[0]
    h = g.get_node_feature().unsqueeze(0)
    edge = g.get_edge_feature().unsqueeze(0)
    adj = g.get_adjacency_matrix().unsqueeze(0)
    if mask:
        val_h = g.get_valid_node().unsqueeze(0)
        val_e = g.get_valid_edge().unsqueeze(0)

    if clone:
        h = h.clone()
        edge = edge.clone()
        adj = adj.clone()
        if mask:
            val_h = val_h.clone()
            val_e = val_e.clone()

    if not mask:
        return h, edge, adj
    return h, edge, adj, val_h, val_e

def make_graph_from_mol(mol, v_max, self_loop=False):
    from dataset import get_atom_feature, get_bond_feature 
    from rdkit import Chem

    graph = Graph2D(utils.NUM_ATOM_TYPES, utils.NUM_BOND_TYPES, v_max, self_loop)
    
    try: 
        Chem.SanitizeMol(mol)
        conf = mol.GetConformer(0)
        xyz = conf.GetPositions()
    except: return

    for i, atom in enumerate(mol.GetAtoms()):
        v_feature = get_atom_feature(atom)
        graph.add_node(v_feature)

    for i in range(mol.GetNumAtoms()):
        for j in range(mol.GetNumAtoms()):
            bond = mol.GetBondBetweenAtoms(i, j)
            if bond is not None and not graph.is_edge_between(i, j):
                e_feature = get_bond_feature(bond)
                graph.add_edge_between(e_feature, i, j)

    return graph

def make_graphs_from_mol(mol, v_max, self_loop=False):
    from dataset import get_atom_feature, get_bond_feature 
    from rdkit import Chem
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

    whole = Graph2D(utils.NUM_ATOM_TYPES, utils.NUM_BOND_TYPES, v_max, self_loop)
    scaff = Graph2D(utils.NUM_ATOM_TYPES, utils.NUM_BOND_TYPES, v_max, self_loop)

    try: 
        scf = Chem.MolFromSmiles(MurckoScaffoldSmiles(mol=mol))
        new_idx = list(mol.GetSubstructMatches(scf)[0])
    except: return None, None
    
    for i in range(mol.GetNumAtoms()):
        if i in new_idx: continue
        new_idx.append(i)

    for i, atom in enumerate(scf.GetAtoms()):
        v_feature = get_atom_feature(atom)
        scaff.add_node(torch.FloatTensor(v_feature))
    
    for i in range(scf.GetNumAtoms()):
        for j in range(scf.GetNumAtoms()):
            bond = scf.GetBondBetweenAtoms(i, j)
            if bond is not None and not scaff.is_edge_between(i, j):
                e_feature = get_bond_feature(bond)
                scaff.add_edge_between(torch.FloatTensor(e_feature), i, j)

    for i, atom in enumerate(mol.GetAtoms()):
        v_feature = get_atom_feature(mol.GetAtomWithIdx(new_idx[i]))
        whole.add_node(torch.FloatTensor(v_feature))
    
    for i in range(scf.GetNumAtoms()):
        for j in range(scf.GetNumAtoms()):
            bond = mol.GetBondBetweenAtoms(new_idx[i], new_idx[j])
            if bond is not None and not whole.is_edge_between(i, j):
                e_feature = get_bond_feature(bond)
                whole.add_edge_between(torch.FloatTensor(e_feature), i, j)

    for i in range(mol.GetNumAtoms()):
        for j in range(mol.GetNumAtoms()):
            bond = mol.GetBondBetweenAtoms(new_idx[i], new_idx[j])
            if bond is not None and not whole.is_edge_between(i, j):
                e_feature = get_bond_feature(bond)
                whole.add_edge_between(torch.FloatTensor(e_feature), i, j)

    return scaff, whole


if __name__ == "__main__":

    pass
