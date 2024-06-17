import torch
import torch.nn as nn
from layers import BranchingLayer, STGCNInferenceLayer


class High_Encoder(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, num_context, num_node, num_pseudo, dropout, his_ts, device, block, adj):
        super().__init__()

        self.num_layers = num_layers
        self.num_context = num_context
        self.his_ts = his_ts
        self.num_pseudo = num_pseudo
        self.input_dim = input_dim
        self.device = device
        self.adj = adj

        self.input_embed = nn.Linear(input_dim, hidden_dim)
        self.expert_layers = nn.ModuleList(
            [STGCNInferenceLayer(hidden_dim, num_context, num_node, block, self.adj) for _ in range(self.num_layers)])
        self.branch_layers = nn.ModuleList([BranchingLayer(num_context, hidden_dim) for _ in range(self.num_layers)])
        self.last_layernorm = nn.LayerNorm(hidden_dim, eps=1e-8)
        self.end_mlp = nn.Sequential(nn.Linear(hidden_dim, 128),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     )

    def forward(self, x):
        batch, ts, node_num, input_dim = x.shape
        x = self.input_embed(x.permute(0, 2, 1, 3).reshape(batch * node_num, ts, input_dim))

        x = x.unsqueeze(dim=-2).repeat(1, 1, self.num_context, 1)

        weight_list = []
        for i in range(self.num_layers):
            weights = self.branch_layers[i](x)
            if i == self.num_layers - 1:
                weights = weights.mean(dim=2, keepdim=True)
            weight_list.append(weights.transpose(-1, -2))
            output = self.expert_layers[i](x)
            x = (output.unsqueeze(dim=2) * weights.unsqueeze(dim=-2)).sum(dim=-1)

        com_feats = self.last_layernorm(x.squeeze(dim=2))
        feats_predict = com_feats.reshape(batch, node_num, ts, -1).permute(0, 2, 1, 3)
        predict_out = self.end_mlp(feats_predict)

        context_probs = self.weights_product(weight_list)
        context_probs = context_probs.flatten(start_dim=2)

        return predict_out, context_probs

    def weights_product(self, weights_list):
        result = weights_list[0]
        for i in range(len(weights_list) - 1):
            new_layer_w = weights_list[i+1].clone()
            for _ in range(i):
                new_layer_w.unsqueeze_(dim = 2)
                size = [-1 for _ in range(len(new_layer_w.shape))]
                size[2] = self.C
                new_layer_w = new_layer_w.expand(size)
            result = torch.einsum('{0}ij,{0}jk->{0}ijk'.format('abcdefg'[:i+2]), result, new_layer_w)
        return result

    def kl_loss(self, weights, max_min):
        kl_criterion = torch.nn.KLDivLoss(reduction='batchmean')
        prior = self.kl_forward(max_min).unsqueeze(dim=0).unsqueeze(dim=0).repeat(weights.shape[0], weights.shape[1], 1)
        return kl_criterion(prior.log(), weights)

    def kl_forward(self, max_min):
        n_max, n_min = max_min[0], max_min[1]
        x = torch.empty([self.num_pseudo, self.his_ts, self.input_dim]).random_(int(n_min), int(n_max)).to(self.device)
        x = self.input_embed(x)
        x = x.unsqueeze(dim=-2).repeat(1, 1, self.num_context, 1)

        weight_list = []
        for i in range(self.num_layers):
            weights = self.branch_layers[i](x)
            if i == self.num_layers - 1:
                weights = weights.mean(dim=2, keepdim=True)
            weight_list.append(weights.transpose(-1, -2))
            output = self.expert_layers[i](x)
            x = (output.unsqueeze(dim=2) * weights.unsqueeze(dim=-2)).sum(dim=-1)

        context_probs = self.weights_product(weight_list)
        context_probs = context_probs.flatten(start_dim=2)

        return context_probs.mean(dim=0, keepdim=False)[-1]
