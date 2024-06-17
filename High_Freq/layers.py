import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.init as init
from STGCN_layer import *

class STGCNGraphConv(nn.Module):
    def __init__(self, blocks, n_vertex, adj, Kt=3, Ks=3, act_func='glu', graph_conv_type='graph_conv', enable_bias=True, droprate=0.3):
        super(STGCNGraphConv, self).__init__()

        self.n_node = n_vertex

        modules = []
        for l in range(len(blocks) - 3):
            modules.append(STConvBlock(Kt, Ks, n_vertex, blocks[l][-1], blocks[l + 1], act_func,
                                              graph_conv_type, adj, enable_bias, droprate))
        self.st_blocks = nn.Sequential(*modules)
        self.fc1 = nn.Linear(in_features=blocks[-3][-1], out_features=blocks[-2][0], bias=enable_bias)
        self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0], bias=enable_bias)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.silu = nn.SiLU()
        self.do = nn.Dropout(p=droprate)
        self.output_linear = nn.Conv2d(6, 14, (1, 1))


    def forward(self, x):
        B, ts, in_dim = x.shape
        x = x.reshape(-1, self.n_node, ts, in_dim).permute(0, 3, 2, 1)
        x = self.st_blocks(x)
        x = self.output_linear(x.permute(0, 2, -1, 1))

        return x


class STGCNInferenceLayer(nn.Module):
    def __init__(self, hidden_dim, num_context, n_node, blocks, adj):
        super().__init__()

        self.hiden_dim = hidden_dim
        self.num_context = num_context
        self.gso = adj
        self.stgcn_layer = nn.ModuleList([STGCNGraphConv(blocks, n_node, self.gso) for _ in range(num_context)])

    def expert_forward(self, i, x):
        output = self.stgcn_layer[i](x)
        _, ts, _, hidden_dim = output.shape
        output = output.permute(0, 2, 1, 3)
        output = output.reshape(-1, ts, hidden_dim)

        return output

    def forward(self, x):
        experts = [self.expert_forward(i, x[:, :, i, :]) for i in range(self.num_context)]
        stacked_output = torch.stack(experts, dim=-1)

        return stacked_output


class BranchingLayer(nn.Module):
    def __init__(self, num_context, hidden_dim):
        super().__init__()
        self.num_context = num_context
        self.hidden_dim = hidden_dim
        self.activation = torch.tanh
        self.use_gumbel = True

        self.global_gate1 = nn.Parameter(torch.randn(self.num_context, self.hidden_dim, self.hidden_dim), requires_grad=True)
        self.global_gate2 = nn.Parameter(torch.randn(self.num_context, self.hidden_dim, 1), requires_grad=True)

    def forward(self, x):
        inputs = x
        weight_logits = []
        for i in range(self.num_context):
            sim = self.activation(inputs @ self.global_gate1[i])
            sim = sim @ self.global_gate2[i]
            weight_logits.append(sim.squeeze(dim=-1))

        weight_logits = torch.stack(weight_logits, dim=-1)

        if self.use_gumbel:
            weights = F.gumbel_softmax(weight_logits, tau=1, dim=-1)
        else:
            weights = F.softmax(weight_logits, dim=-1)

        return weights




