import torch
import torch.nn as nn
import torch.nn.functional as F

import copy


class GGNNBlock(nn.Module): # GGNN + weight tying

    def __init__(
            self,
            node_feature,
            edge_feature,
            num_layers
            ):
        super().__init__()

        self.node_feature = node_feature
        self.edge_feature = edge_feature
        self.num_layers = num_layers

        self.propagate = nn.ModuleList([
                GGNN(
                        node_feature,
                        edge_feature
                ) for _ in range(num_layers)
        ])

        self.L1 = nn.Linear(node_feature*2, node_feature)
        self.L2 = nn.Linear(node_feature, node_feature)

    def forward(
            self,
            h,
            edge,
            adj
            ):

        h0 = h.clone()
        for lay in self.propagate:
            h = lay(h, edge, adj)
        
        g = torch.sigmoid(self.L1(torch.cat([h, h0], -1)))
        h = self.L2(h)

        R = (g*h).mean(1)
        return torch.relu(R)


class GGNN(nn.Module): # Gated Graph Neural Network

    def __init__(
            self,
            node_feature,
            edge_feature,
            ):
        super().__init__()

        self.node_feature = node_feature
        self.edge_feature = edge_feature

        self.W = nn.Linear(2*node_feature+edge_feature, node_feature)
        self.C = nn.GRUCell(node_feature, node_feature)

    def forward(
            self,
            h,
            edge,
            adj
            ):

        b, N = h.shape[0], h.shape[1]
        h1 = h.unsqueeze(1).repeat(1, N, 1, 1) # [b, N, N, nf]
        h2 = h.unsqueeze(2).repeat(1, 1, N, 1) # [b, N, N, nf]
        cat = torch.cat([h1, edge, h2], -1) # [b, N, N, nf*2 + ef]
        m = self.W(cat)*adj.unsqueeze(-1) # [b, N, N, nf]
        m = m.mean(2) # [b, N, nf]

        reshape_m = m.view(-1, self.node_feature)
        reshape_h = h.view(-1, self.node_feature)
        new_h = self.C(reshape_h, reshape_m)
        new_h = new_h.view(b, N, self.node_feature) # [b, N, nf]
        return torch.relu(new_h)


class Gate(nn.Module):

    def __init__(
            self,
            in_feature,
            edge_feature,
            out_feature
            ):
        super().__init__()
        
        """
        in_feature == out_feature
        """
        self.in_feature = in_feature
        self.edge_feature = edge_feature
        self.out_feature = out_feature

        self.G = nn.Linear(in_feature+edge_feature, 1)
        self.W = nn.Linear(edge_feature, out_feature)

    def forward(
            self,
            h,
            message
            ):
        cat = torch.cat([h, message], -1) # [b, N, in_feature+edge_feature]
        coeff = torch.sigmoid(self.G(cat)).repeat(1, 1, self.out_feature)
        retval = coeff*h + (1-coeff)*self.W(message)
        return F.silu(retval)  


class MessagePassing(nn.Module):

    def __init__(
            self,
            in_feature,
            edge_feature,
            ):
        super().__init__()

        self.in_feature = in_feature
        self.edge_feature = edge_feature

        self.W = nn.Linear(2*in_feature+edge_feature, edge_feature)
        self.C = nn.GRUCell(edge_feature, edge_feature)

    def forward(
            self,
            h,
            edge,
            adj
            ):

        b, N = h.shape[0], h.shape[1]
        h1 = h.unsqueeze(1).repeat(1, N, 1, 1)
        h2 = h.unsqueeze(2).repeat(1, 1, N, 1)
        cat = torch.cat([h1, edge, h2], -1)
        m = self.W(cat)*adj.unsqueeze(-1)

        reshape_m = m.view(-1, self.edge_feature)
        reshape_e = edge.view(-1, self.edge_feature)
        retval = self.C(reshape_m, reshape_e)
        retval = retval.view(b, N, N, self.edge_feature)
        return F.silu(retval)


class soft_cutoff(nn.Module):

    def __init__(
            self,
            x_min,
            x_max,
            y_min,
            y_max,
            gamma=6
            ):
        super().__init__()

        assert x_min < x_max, f"x_min is larger than x_max"
        assert y_min < y_max, f"y_min is larger than y_max"
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.gamma= gamma

        self.sig = nn.Sigmoid()

    def forward(
            self,
            input
            ):
        x_mid = (self.x_min + self.x_max) / 2
        gamma = self.gamma / (self.x_max - self.x_min)
        retval = -self.sig(gamma * (input - x_mid)) + 1
        retval = retval * (self.y_max - self.y_min) + self.y_min
        return retval

    def visualize(
            self,
            fn,
            sigma=2,
            delta=0.01,
            draw_solid=False
            ):
        import matplotlib 
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = torch.Tensor([self.x_min - sigma + i * delta for i in \
                range(int((self.x_max - self.x_min + 2 * sigma) / delta))]) 
        y = self.forward(x)

        plt.plot(x.cpu().numpy(), y.cpu().numpy())
        if draw_solid:
            plt.hlines(self.y_min, self.x_max, self.x_max + sigma, \
                    color='red')
            plt.hlines(self.y_max, self.x_min - sigma, self.x_min, \
                    color='red')
            plt.plot([self.x_min, self.x_max], [self.y_max, self.y_min], \
                    color='red')

        plt.savefig(f"{fn}.png")
        plt.close()
        return


class soft_one_hot(nn.Module):

    def __init__(
            self,
            x_min,
            x_max,
            steps,
            gamma=10.0
            ):
        super().__init__()

        assert x_min < x_max, "x_min is larger than x_max"
        self.x_min = x_min
        self.x_max = x_max
        self.steps = steps
        self.center = torch.Tensor([x_min * (steps - i - 1) / (steps - 1) + \
                           x_max * i / (steps - 1) for i in range(steps)])

        self.gamma = gamma

    def forward(
            self,
            x
            ):
        x_repeat = x.unsqueeze(-1).repeat(1, 1, 1, self.steps)
        c_repeat = self.center.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        c_repeat = c_repeat.to(x_repeat.device)
        x_embed = torch.exp(-self.gamma * torch.pow(x_repeat - c_repeat, 2))
        return x_embed

    def visualize(
            self,
            fn
            ):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        x = torch.Tensor([self.x_min + i * (self.x_max - self.x_min) / self.steps \
                for i in range(self.steps)])
        x = x.unsqueeze(0).unsqueeze(0)
        y = self.forward(x) # [steps, steps]
        y = y.squeeze(0).squeeze(0)

        fig, ax = plt.subplots()
        im = ax.imshow(y.cpu().numpy(), cmap="Blues", vmin=0, vmax=1)
        fig.tight_layout()
        plt.savefig(f"{fn}.png")
        plt.close()
        return


class ReadOut(nn.Module):
    def __init__(self, n_hidden, pooling=None):
        super().__init__()

        self.pooling = pooling
        self.fc = nn.Linear(n_hidden, n_hidden, bias=False)

    def forward(self, x):
        return self.pooling(self.fc(x))


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, x):
        return torch.mean(x, dim=1)


class SumPooling(nn.Module):
    def __init__(self):
        super(SumPooling, self).__init__()

    def forward(self, x):
        return torch.sum(x, dim=1)


class MaxPooling(nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()

    def forward(self, x):
        return torch.max(x, dim=1)


class _identity(nn.Module):

    def __init__(
            self
            ):
        super().__init__()

    def forward(
            self,
            input
            ):
        return input


if __name__ == "__main__":

    pass
