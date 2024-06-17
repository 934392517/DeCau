import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class static_conv(nn.Module):
    def __init__(self):
        super(static_conv, self).__init__()

    def forward(self, A, x):
        x = torch.matmul(A, x)
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = nn.Linear(c_in, c_out)

    def forward(self, x):
        return F.relu(self.mlp(x), inplace=True)

class GCN(nn.Module):
    def __init__(self, c_in, c_out, depth, dropout, graph_num):
        super(GCN, self).__init__()
        self.static_conv = static_conv()
        self.mlp = linear((depth + 1) * c_in, c_out)

        self.weight = nn.Parameter(torch.FloatTensor(graph_num + 1), requires_grad=True)
        self.weight.data.fill_(1.0)

        self.dropout = dropout
        self.depth = depth
        self.graph_num = graph_num
        self.c_in = c_in

    def forward(self, x, static_adj, dyn_adj=None):
        h = x
        out = [h]

        weight = F.softmax(self.weight, dim=0)

        for _ in range(self.depth):
            h_next = weight[0] * h
            for i in range(self.graph_num):
                h_next += weight[i+1] * self.static_conv(static_adj[i], h)

            if dyn_adj is not None:
                h_next += weight[-1] * self.dyn_conv(dyn_adj, h)

            h = h_next
            out.append(h)

        hout = torch.cat(out, dim=-1)
        hout = self.mlp(hout)

        return hout


class GraphGateRNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, gcn_depth, node_num, static_norm_adjs, dropout, dropout_type='zoneout'):
        super(GraphGateRNN, self).__init__()

        self.hidden_channles = hidden_channels
        self.static_norm_adjs = static_norm_adjs
        self.dropout_type = dropout_type

        self.static_norm_adjs = static_norm_adjs

        self.start_fc = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=(1, 1))

        self.dropout = nn.Dropout(dropout)
        self.dropout_rate = dropout

        self.GCN_update = GCN(hidden_channels*2, hidden_channels, gcn_depth, dropout, len(static_norm_adjs))
        self.GCN_reset = GCN(hidden_channels*2, hidden_channels, gcn_depth, dropout, len(static_norm_adjs))
        self.GCN_cell = GCN(hidden_channels*2, hidden_channels, gcn_depth, dropout, len(static_norm_adjs))
        self.layerNorm = nn.LayerNorm([self.hidden_channles])

    def forward(self, flow, hidden_state):
        batch, node_num, his_ts, in_channels = flow.shape
        x = self.start_fc(flow.permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1)
        hidden_state = hidden_state.view(batch, node_num, self.hidden_channles)

        combined = torch.cat((x, hidden_state), -1)

        sta_norm_adj = [adj for adj in self.static_norm_adjs]

        update_gate = torch.sigmoid(self.GCN_update(combined, sta_norm_adj))
        reset_gate = torch.sigmoid(self.GCN_reset(combined, sta_norm_adj))
        temp = torch.cat((x, torch.mul(reset_gate, hidden_state)), -1)
        cell_state = torch.tanh(self.GCN_cell(temp, sta_norm_adj))
        next_hidden_static = torch.mul(update_gate, hidden_state) + torch.mul(1.0 - update_gate, cell_state)
        next_hidden = self.layerNorm(next_hidden_static)

        output = next_hidden
        if self.dropout_type == 'zoneout':
            next_hidden = self.zoneout(pre_h=hidden_state, next_h=next_hidden, rate=self.dropout_rate, training=self.training)

        return output, next_hidden

    def zoneout(self, pre_h, next_h, rate, training=True):
        if training:
            d = torch.zeros_like(next_h).bernoulli_(rate)
            next_h = d * pre_h + (1 - d) * next_h
        else:
            next_h = rate * pre_h + (1 - rate) * next_h

        return next_h

class Flow_Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, gcn_depth, node_num, device, dropout,
                 dropout_type, static_norm_adjs):
        super(Flow_Encoder, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.static_norm_adjs = static_norm_adjs

        self.RNN_layer = 1
        self.device = device

        self.RNNCell = nn.ModuleList([
            GraphGateRNN(in_channels, hidden_channels, gcn_depth=gcn_depth, node_num=node_num, static_norm_adjs=static_norm_adjs, dropout=dropout, dropout_type=dropout_type)
        ])
        self.avg_pool = nn.AdaptiveAvgPool2d((None, 1))

    def forward(self, flow, poi):
        flow = flow.permute(0, 2, 1, 3)
        batch, node_num, his_ts, in_channels = flow.shape
        hidden_state = [self.init_hidden(batch, node_num, self.hidden_channels) for _ in range(self.RNN_layer)]

        outputs, hiddens = [], []

        for i in range(his_ts):
            cur_flow = flow[:, :, i:i + 1, :]
            for j, rnn_cell in enumerate(self.RNNCell):
                cur_h = hidden_state[j]
                cur_out, cur_h = rnn_cell(cur_flow.to(self.device), cur_h.to(self.device))
                hidden_state[j] = cur_h
                input_cur = F.relu(cur_out, inplace=True)

            outputs.append(cur_out.unsqueeze(dim=2))
            hidden = torch.stack(hidden_state, dim=1).unsqueeze(dim=3)
            hiddens.append(hidden)

        hiddens = torch.cat(hiddens, dim=3)
        hiddens = hiddens.squeeze(dim=1).permute(0, 2, 1, 3)

        return hiddens

    def init_hidden(self, batch, node_num, hidden_channles):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            hidden_state = Variable(
                torch.zeros((batch, node_num, hidden_channles)).to(self.device)
            )

            nn.init.orthogonal_(hidden_state)

            return hidden_state
        else:
            hidden_state = Variable(
                torch.zeros((batch, node_num, hidden_channles))
            )
            return hidden_state

