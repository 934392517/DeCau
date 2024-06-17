import torch
import torch.nn as nn
from evolve_flow import Flow_Encoder
from evolve_poi import Poi_Encoder

class Low_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, gcn_depth, node_num, device, dropout,
                 dropout_type, pre_ts, pre_num, static_norm_adjs):
        super().__init__()

        self.flow_encoder = Flow_Encoder(input_dim, hidden_dim, gcn_depth, node_num, device, dropout,
                 dropout_type, static_norm_adjs)
        self.poi_encoder = Poi_Encoder(hidden_dim, device, dropout, dropout_type)
        self.avg_pool = nn.AdaptiveAvgPool2d((None, pre_num))
        self.predict_flow = nn.Conv2d(14, pre_ts, (1, 1))
        self.predict_poi = nn.Conv2d(4, 14, (1, 1))

        # GLU
        self.gata_FC = nn.Linear(hidden_dim * 2, hidden_dim*2)
        self.info_FC = nn.Linear(hidden_dim * 2, hidden_dim*2)
        self.dropout = nn.Dropout(dropout)

        self.end_FC = nn.Linear(hidden_dim*2, pre_num)
        self.poi2flow = nn.Linear(hidden_dim, 1)

    def forward(self, flow_x, poi_x):
        flow_embedding = self.flow_encoder(flow_x, poi_x)
        poi_embedding = self.poi_encoder(flow_embedding, poi_x)

        poi_embedding = self.predict_poi(poi_embedding)
        output = self.out_gate(flow_embedding, poi_embedding)
        poi2flow = self.poi2flow(poi_embedding)

        return output, poi2flow

    def out_gate(self, flow, poi):
        fusion = torch.cat([flow, poi], dim=-1)

        residual = fusion
        gate_x = torch.sigmoid(self.gata_FC(residual))
        info_x = torch.tanh(self.info_FC(residual))
        x = fusion + torch.mul(gate_x, info_x)
        x = self.dropout(x)

        return x

    def reconstruct_loss(self, contru_predict, flow_x):
        flow_x = flow_x.mean(dim=-1, keepdim=True)
        loss = nn.SmoothL1Loss(reduction='mean').to('cuda:0')

        contru_loss = loss(contru_predict, flow_x)

        return contru_loss





