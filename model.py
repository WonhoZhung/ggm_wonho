import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import GGNN, GGNNBlock
from graph import stack_graph
from utils import prob_to_one_hot, int_to_one_hot
from copy import deepcopy

import random


class GGM(nn.Module): # Graph Generative Model

    def __init__(
            self,
            args
            ):
        super().__init__()

        self.args = args

        self.embed_graph = GraphEmbed(args)
        self.update_graph = GraphUpdate(args)
        self.encoder = Encoder(args)

        self.add_node = AddNode(args)
        self.add_edge = AddEdge(args)
        self.select_node = SelectNode(args)

        self.init_node = InitNode(args)
        self.init_edge = InitEdge(args)

    def calc_cross_entropy_loss(
            self,
            true_prob,
            pred_prob,
            eps=1e-6
            ):
        return (-true_prob*torch.log(pred_prob+eps)).sum()

    def forward_train(
            self,
            scaff,
            whole,
            ):
        
        # Initialize cross-entropy losses
        add_node_loss, select_node_loss, add_edge_loss = [], [], []

        # Copy graphs
        whole_save = deepcopy(whole)
        scaff_save = deepcopy(scaff)

        # Embed graphs
        self.embed_graph(whole)
        self.embed_graph(scaff)

        # Encode whole 
        latent_vector, mu, logvar = self.encoder(whole)

        # Initial propagation
        self.update_graph(scaff)

        # Get nodes
        nodes = [(i, whole_save.get_node_feature_with_idx(i)) for i in \
                    range(scaff.get_num_nodes(), whole.get_num_nodes())]
        if self.args.shuffle: random.shuffle(nodes)
        
        # Iterate number of nodes to be added
        for i, node in nodes:

            # Get new node feature
            new_node = self.add_node(scaff, latent_vector)
            
            # Add to reconstruction losses
            add_node_loss.append(
                    self.calc_cross_entropy_loss(node, new_node[0])
            )

            # Add new node to scaffold (teacher forcing)
            scaff_save.add_node(node)
            scaff.add_node(self.init_node(scaff, node))

            # Get neighboring edges
            edges = whole_save.get_edges(i)
            if self.args.shuffle: random.shuffle(edges)

            # Iterate number of edges to be added
            for j, edge in edges:
                if j > i: continue
                
                # Get new edge feature
                new_edge = self.add_edge(scaff, latent_vector)

                # Add to reconstruction loss
                add_edge_loss.append(
                        self.calc_cross_entropy_loss(edge, new_edge[0]) 
                )

                # Get probability of each idx to add new node
                selected = self.select_node(scaff, latent_vector)
                
                # Add to reconstruction loss
                select_node_loss.append(
                        self.calc_cross_entropy_loss(
                                int_to_one_hot(j, whole.get_max_num_nodes(), \
                                        whole.get_device()),
                                selected[0]
                        )
                )

                # Add new edge to scaffold (teacher forcing)
                scaff_save.add_edge_between(edge, j, i)
                scaff.add_edge_between(self.init_edge(scaff, edge), j, i)

            # End of adding edge
            end_edge = int_to_one_hot(self.args.num_edge_features - 1, \
                    self.args.num_edge_features, scaff.get_device())
            new_edge_feature = self.add_edge(scaff, latent_vector)
            add_edge_loss.append(
                    self.calc_cross_entropy_loss(
                            end_edge,
                            new_edge_feature[0]
                    ) 
            )
        
        # End of adding node
        end_node = int_to_one_hot(self.args.num_edge_features - 1, \
                self.args.num_node_features, scaff.get_device())
        new_node_feature = self.add_node(scaff, latent_vector)
        add_node_loss.append(
                self.calc_cross_entropy_loss(
                        end_node,
                        new_node_feature[0]
                )
        )

        # Calculating losses
        vae_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        recon_loss = sum(add_node_loss) + \
                     sum(add_edge_loss) + \
                     sum(select_node_loss)

        return scaff_save, vae_loss, recon_loss

    def forward_inference(
            self,
            scaff,
            ):

        # Copy graph
        scaff_save = deepcopy(scaff)

        # Embed graph
        self.embed_graph(scaff)

        # Sample latent vector from standard normal 
        # TODO
        latent_vector = torch.randn()

        # Initial propagation
        self.update_graph(scaff)

        for i in range(scaff.get_num_nodes(), scaff.get_max_num_nodes()):
            
            # Get new node feature
            new_node = self.add_node(scaff, latent_vector)
            
            # Break if new_node == STOP
            if new_node.argmax(-1) == scaff.num_node_features - 1: break

            # Add new node to scaffold
            scaff_save.add_node(new_node)
            scaff.add_node(self.init_node(scaff, new_node))

            for _ in range(self.args.max_add_edges):
                
                # Get new edge feature
                new_edge = self.add_edge(scaff, latent_vector)

                # Break if new_edge == STOP
                if new_edge.argmax(-1) == scaff.num_edge_features - 1: break

                # Get probability of each idx to add new node
                selected = self.select_node(scaff, latent_vector)
                
                selected_idx = int(selected.argmax(-1))

                # Add new edge to scaffold
                scaff_save.add_edge_between(new_edge, i, selected_idx)
                scaff.add_edge_between(self.init_edge(scaff, new_edge), \
                        i, selected_idx)

        return scaff_save

    def forward(
            self,
            scaff,
            whole=None,
            train=True
            ):

        if train:
            assert whole is not None
            return self.forward_train(scaff, whole)
        else:
            return self.forward_inference(scaff)


class AddNode(nn.Module):

    def __init__(
            self,
            args
            ):
        super().__init__()

        self.args = args

        self.update_graph = GraphUpdate(args)
        self.graph_vector = GGNNBlock(
                args.num_node_hidden,
                args.num_edge_hidden,
                args.num_layers,
        )
        self.fc = nn.Sequential(
                nn.Linear(args.num_node_hidden*2, args.num_node_hidden),
                nn.ReLU(),
                nn.Linear(args.num_node_hidden, args.num_node_features)
        )

    def forward(
            self,
            graph,
            latent_vector
            ):
        # Propagate
        self.update_graph(graph)

        # Get graph info from graph object
        h, e, adj, val_h, val_e = stack_graph([graph], mask=True, clone=True)

        graph_vector = self.graph_vector(h, e, adj)
        new_node = self.fc(torch.cat([graph_vector, latent_vector], -1))
        return F.softmax(new_node, -1)


class AddEdge(nn.Module):

    def __init__(
            self,
            args
            ):
        super().__init__()

        self.args = args

        self.update_graph = GraphUpdate(args)
        self.graph_vector = GGNNBlock(
                args.num_node_hidden,
                args.num_edge_hidden,
                args.num_layers,
        )
        self.fc = nn.Sequential(
                nn.Linear(args.num_node_hidden*2, args.num_node_hidden),
                nn.ReLU(),
                nn.Linear(args.num_node_hidden, args.num_edge_features)
        )

    def forward(
            self,
            graph,
            latent_vector
            ):
        # Propagate
        self.update_graph(graph)

        # Get graph info from graph object
        h, e, adj, val_h, val_e = stack_graph([graph], mask=True, clone=True)

        graph_vector = self.graph_vector(h, e, adj)
        new_edge = self.fc(torch.cat([graph_vector, latent_vector], -1))
        return F.softmax(new_edge, -1)
        

class SelectNode(nn.Module):

    def __init__(
            self,
            args
            ):
        super().__init__()

        self.args = args

        self.update_graph = GraphUpdate(args)
        self.graph_vector = GGNNBlock(
                args.num_node_hidden,
                args.num_edge_hidden,
                args.num_layers,
        )
        self.fc = nn.Sequential(
                nn.Linear(args.num_node_hidden*2, args.num_node_hidden),
                nn.ReLU(),
                nn.Linear(args.num_node_hidden, 1)
        )

    def forward(
            self,
            graph,
            latent_vector # [b, hidden]
            ):
        # Propagate
        self.update_graph(graph)

        # Get graph info from graph object
        h, e, adj, val_h, val_e = stack_graph([graph], mask=True, clone=True)
        N = graph.get_max_num_nodes()
        num_node = graph.get_num_nodes()
        latent_vector = latent_vector.unsqueeze(1).repeat(1, N, 1)

        selected_node = self.fc(torch.cat([h, latent_vector], -1)).squeeze(-1)
        selected_node[:, num_node:] = float("-Inf")

        return F.softmax(selected_node, -1)


class InitNode(nn.Module):

    def __init__(
            self,
            args
            ):
        super().__init__()

        self.args = args

        self.graph_vector = GGNNBlock(
                args.num_node_hidden,
                args.num_edge_hidden,
                args.num_layers,
        )
        self.node_embedding = nn.Linear(args.num_node_features, \
                args.num_node_hidden, bias=False)
        self.fc = nn.Linear(args.num_node_hidden*2, args.num_node_hidden)

    def forward(
            self,
            graph,
            node
            ):
        # Get graph info from graph object
        h, e, adj, val_h, val_e = stack_graph([graph], mask=True)

        graph_vector = self.graph_vector(h, e, adj).squeeze(0)
        init_node = self.fc(
                torch.cat([graph_vector, self.node_embedding(node)], -1)
        )
        return init_node
        

class InitEdge(nn.Module):

    def __init__(
            self,
            args
            ):
        super().__init__()

        self.args = args

        self.graph_vector = GGNNBlock(
                args.num_node_hidden,
                args.num_edge_hidden,
                args.num_layers,
        )
        self.edge_embedding = nn.Linear(args.num_edge_features, \
                args.num_edge_hidden, bias=False)
        self.fc = nn.Linear(args.num_edge_hidden*2, args.num_edge_hidden)

    def forward(
            self,
            graph,
            edge
            ):
        # Get graph info from graph object
        h, e, adj, val_h, val_e = stack_graph([graph], mask=True)

        graph_vector = self.graph_vector(h, e, adj).squeeze(0)
        init_edge = self.fc(
                torch.cat([graph_vector, self.edge_embedding(edge)], -1)
        )
        return init_edge
        

class GraphEmbed(nn.Module):

    def __init__(
            self,
            args
            ):
        super().__init__()

        self.args = args

        self.node_embedding = nn.Linear(args.num_node_features, \
                args.num_node_hidden, bias=False)
        self.edge_embedding = nn.Linear(args.num_edge_features, \
                args.num_edge_hidden, bias=False)

    def forward(
            self,
            graph
            ):
            # Get graph info from graph object
            h, e, adj, val_h, val_e = stack_graph([graph], mask=True, clone=False)

            new_h, new_e =  self.node_embedding(h), self.edge_embedding(e)
            graph.set_node_feature_mat(new_h[0])
            graph.set_edge_feature_mat(new_e[0])


class GraphUpdate(nn.Module):

    def __init__(
            self,
            args
            ):
        super().__init__()

        self.args = args

        self.propagate = nn.ModuleList([
                GGNN(
                        args.num_node_hidden,
                        args.num_edge_hidden
                ) for _ in range(args.num_layers)
        ])

    def forward(
            self,
            graph,
            clone=True
            ):
        # Get graph info from graph object
        h, edge, adj, val_h, val_e = stack_graph([graph], mask=True, clone=clone)
        
        for lay in self.propagate:
            h = lay(h, edge, adj)
        
        graph.set_node_feature_mat(h[0])


class Encoder(nn.Module):

    def __init__(
            self,
            args
            ):
        super().__init__()

        self.args = args

        self.graph_vector = GGNNBlock(
                args.num_node_hidden,
                args.num_edge_hidden,
                args.num_layers,
        )
        self.mean = nn.Linear(args.num_node_hidden, args.num_node_hidden, \
                bias=False)
        self.logvar = nn.Linear(args.num_node_hidden, args.num_node_hidden, \
                bias=False)

    def reparameterize(
            self,
            mean,
            std
            ):
        eps = torch.randn(std.shape, device=std.device)
        return eps*std + mean

    def forward(
            self,
            graph
            ):
        # Get graph info from graph object
        h, edge, adj, val_h, val_e = stack_graph([graph], mask=True)

        graph_vector = self.graph_vector(h, edge, adj)
        mean = self.mean(graph_vector)
        logvar = self.logvar(graph_vector)
        std = torch.exp(0.5*logvar)
        return self.reparameterize(mean, std), mean, logvar

