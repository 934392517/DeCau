import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim, hidden_size):
        super().__init__()
        self.pool = nn.Conv2d(14, 4, (1, 1))
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        Activation = nn.Sigmoid

        def _dense(in_dim, out_dim):
            return nn.Sequential(nn.Linear(in_dim, out_dim), Activation(), nn.Dropout(0.3))

        dimension_pair = [embedding_dim * 4] + hidden_size
        layers = [_dense(dimension_pair[i], dimension_pair[i+1]) for i in range(len(hidden_size))]
        layers.append(nn.Linear(hidden_size[-1], 1))
        self.attention = nn.Sequential(*layers)

    def forward(self, poi, flow):
        batch, _, poi_ts, poi_type, _ = poi.shape
        flow = flow.unsqueeze(2).unsqueeze(3).expand(-1, -1, poi_ts, poi_type, -1)
        combined = torch.cat([flow, poi], dim=-1)
        scores = self.attention(combined).squeeze(-1)
        type_score = F.softmax(scores, dim=-1)
        reduce_dim = self.avg_pool(scores.reshape(-1, poi_ts, poi_type)).squeeze(-1).reshape(batch, -1, poi_ts)
        ts_score = F.softmax(reduce_dim, dim=-1)

        return type_score, ts_score


class GRUCell(nn.Module):
    def __init__(self, input_dim, embedding_dim, dropout, dropout_type='zoneout', bias=True):
        super().__init__()

        self.reset_gate = nn.Sequential(nn.Linear(input_dim * 2, embedding_dim, bias=bias), nn.Sigmoid())
        self.update_gate = nn.Sequential(nn.Linear(input_dim * 2, embedding_dim, bias=bias), nn.Sigmoid())
        self.h_hat_gate = nn.Sequential(nn.Linear(input_dim * 2, embedding_dim, bias=bias), nn.Tanh())

        self.layerNorm = nn.LayerNorm([embedding_dim])
        self.dropout_rate = dropout
        self.dropout_type = dropout_type

    def forward(self, x, h_prev):
        combined_input = torch.cat((x, h_prev), dim=-1)
        r = self.reset_gate(combined_input)
        u = self.update_gate(combined_input)
        h_hat = self.h_hat_gate(torch.cat([x, h_prev * r], dim=-1))

        h_cur = (1. - u) * h_prev + u * h_hat
        h_cur = self.layerNorm(h_cur)

        if self.dropout_type == 'zoneout':
            next_hidden = self.zoneout(pre_h=h_prev, next_h=h_cur, rate=self.dropout_rate, training=self.training)

        return next_hidden

    def zoneout(self, pre_h, next_h, rate, training=True):
        if training:
            d = torch.zeros_like(next_h).bernoulli_(rate)
            next_h = d * pre_h + (1 - d) * next_h
        else:
            next_h = rate * pre_h + (1 - rate) * next_h

        return next_h


class DynamicGRU(nn.Module):
    def __init__(self, input_dim, embedding_dim, dropout, dropout_type, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.rnn_cell = GRUCell(input_dim, embedding_dim, dropout, dropout_type, bias=bias)

    def forward(self, x_poi, h0=None):
        B, node_num, ts, poi_type, dim = x_poi.shape

        output = torch.zeros(B, node_num, ts, poi_type, self.embedding_dim).type(x_poi.type())
        h_prev = torch.zeros(B, node_num, poi_type, dim).type(x_poi.type()) if h0 == None else h0

        for t in range(ts):
            h_prev = output[:, :, t, :, :] = self.rnn_cell(x_poi[:, :, t, :, :], h_prev)

        return output

class Poi_Encoder(nn.Module):
    def __init__(self, embedding_dim, device, dropout, dropout_type):
        super().__init__()

        self.poi_embedding = nn.Linear(1, embedding_dim)
        self.attention = AttentionLayer(embedding_dim, [128, 256])
        self.dynamicGRU = DynamicGRU(embedding_dim, embedding_dim, dropout, dropout_type)
        self.avg_pool_poi = nn.AdaptiveAvgPool2d((None, embedding_dim))
        self.avg_pool_flow = nn.AdaptiveAvgPool2d((None, 1))
        self.embedding_dim = embedding_dim
        self.device = device
        self.pool_ts = nn.AdaptiveAvgPool1d(1)

    def forward(self, flow, poi):
        B, _, node_num, poi_type = poi.shape
        poi = poi.permute(0, 2, 1, 3).unsqueeze(-1)
        poi_embedding = self.poi_embedding(poi)
        output = self.dynamicGRU(poi_embedding)
        type_score, ts_score = self.attention(poi_embedding, flow)
        causal_embedding = torch.mul(type_score.unsqueeze(-1), output).sum(dim=-2).permute(0, 2, 1, 3)

        return causal_embedding

    def use_period(self, poi_embedding, score):
        _, _, _, _, dim = poi_embedding.shape
        type_score = score.mean(dim=-1, keepdim=True)
        poi_embedding = poi_embedding.sum(dim=-2).squeeze(-2)

        return torch.mul(type_score, poi_embedding)

    def use_type(self, poi_embedding, score):
        _, _, _, _, dim = poi_embedding.shape
        period_score = score.permute(0, 1, 3, 2).mean(dim=-1, keepdim=True).unsqueeze(2)

        return torch.mul(period_score, poi_embedding).sum(dim=-2)

    def init_hidden(self, batch, node_num, poi_type, hidden_channles):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            hidden_state = Variable(
                torch.zeros((batch, node_num, poi_type, hidden_channles)).to(self.device)
            )

            nn.init.orthogonal_(hidden_state)

            return hidden_state
        else:
            hidden_state = Variable(
                torch.zeros((batch, node_num, poi_type, hidden_channles))
            )
            return hidden_state



