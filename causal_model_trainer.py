import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from Flow_decomposition.disentangling import series_decomp
from High_Freq.high_encoder import High_Encoder
from Low_Freq.low_encoder import Low_Encoder
from util import *

class StepLR2(MultiStepLR):
    """StepLR with min_lr"""

    def __init__(self,
                 optimizer,
                 milestones,
                 gamma=0.1,
                 last_epoch=-1,
                 min_lr=2.0e-6):
        """

        :optimizer: TODO
        :milestones: TODO
        :gamma: TODO
        :last_epoch: TODO
        :min_lr: TODO

        """
        self.optimizer = optimizer
        self.milestones = milestones
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.min_lr = min_lr
        super(StepLR2, self).__init__(optimizer, milestones, gamma)

    def get_lr(self):
        lr_candidate = super(StepLR2, self).get_lr()
        if isinstance(lr_candidate, list):
            for i in range(len(lr_candidate)):
                lr_candidate[i] = max(self.min_lr, lr_candidate[i])

        else:
            lr_candidate = max(self.min_lr, lr_candidate)

        return lr_candidate

class Model(nn.Module):
    def __init__(self, decomp_kernel, inference_ayers, input_dim, hidden_dim_low, hidden_dim_high, num_context, num_node, num_pseudo,
                 dropout, his_ts, pre_ts, pre_num, device, block, adj, gcn_depth, dropout_type):
        super().__init__()

        self.low_dim = hidden_dim_low

        self.series_decomp = series_decomp(decomp_kernel)
        self.high_encoder = High_Encoder(inference_ayers, input_dim, hidden_dim_high, num_context, num_node, num_pseudo, dropout, his_ts, device, block, adj[0])
        self.low_encoder = Low_Encoder(input_dim, hidden_dim_low, gcn_depth, num_node, device, dropout,
                 dropout_type, pre_ts, pre_num, adj)

        # out_gate
        self.gata_FC = nn.Linear(hidden_dim_low * 4, hidden_dim_low * 4)
        self.info_FC = nn.Linear(hidden_dim_low * 4, hidden_dim_low * 4)
        self.dropout = nn.Dropout(dropout)

        self.predict = nn.Sequential(nn.Linear(hidden_dim_low * 2, hidden_dim_low * 6),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(hidden_dim_low * 6, hidden_dim_low * 4),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(hidden_dim_low * 4, pre_num),
                                     )

        self.conv = nn.Linear(his_ts, pre_ts)
        self.fc = nn.Linear(128, 1)

    def forward(self, x, poi):
        low_x, high_x = self.series_decomp(x)
        output_l, poi2flow = self.low_encoder(low_x, poi)
        construct_loss = self.low_encoder.reconstruct_loss(poi2flow, low_x)
        output_h, context_probs = self.high_encoder(high_x)
        a, b = 1, 0.5
        combined = a * output_h + b * output_l
        combined = self.predict(combined)
        pred_output = self.conv(combined.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)

        return pred_output, context_probs, construct_loss


class Trainer():
    def __init__(self, model, lr, max_min, gso, kl_ratio, weight_decay, milestones, lr_decay_ratio, min_lr, loss_type, max_grad_norm, scaler, device):

        self.scaler = scaler
        self.model = model
        self.model.to(device)
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.kl_ratio = kl_ratio
        self.max_min = max_min
        self.gso = torch.tensor(gso).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = StepLR2(optimizer=self.optimizer, milestones=milestones, gamma=lr_decay_ratio, min_lr=min_lr)

        if loss_type == 'L1':
            self.loss_pred = nn.L1Loss(reduction='mean').to(device)
        elif loss_type == 'L2':
            self.loss_pred = nn.MSELoss(reduction='mean').to(device)
        elif loss_type == 'Smooth':
            self.loss_pred = nn.SmoothL1Loss(reduction='mean').to(device)

    def train(self, x, poi, y):
        self.model.train()
        self.optimizer.zero_grad()
        output, context_probs, construct_loss = self.model(x, poi)
        predict = self.scaler.inverse_transform(output)

        y = y.mean(dim=-1, keepdim=True)
        loss_kl = self.model.high_encoder.kl_loss(context_probs, self.max_min)

        loss_pre = self.loss_pred(predict, y)
        total_loss = loss_pre + self.kl_ratio * loss_kl + construct_loss

        mae = masked_mae(predict, y, null_val=np.Inf).item()
        mape = masked_mape(predict, y, null_val=np.Inf).item()
        rmse = masked_rmse(predict, y, null_val=np.Inf).item()

        total_loss.backward()

        if self.max_grad_norm != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.optimizer.step()

        return total_loss.item(), loss_kl.item(), mae, mape, rmse

    def eval(self, x, poi, y):
        self.model.eval()
        with torch.no_grad():
            output, context_probs = self.model(x, poi)

        predict = self.scaler.inverse_transform(output)

        y = y.mean(dim=-1, keepdim=True)
        loss_pred = self.loss_pred(predict, y)

        mae = masked_mae(predict, y, null_val=np.Inf).item()
        mape = masked_mape(predict, y, null_val=np.Inf).item()
        rmse = masked_rmse(predict, y, null_val=np.Inf).item()

        return loss_pred.item(), mae, mape, rmse

    def test(self, x, poi, y):
        self.model.eval()
        with torch.no_grad():
            output, context_probs = self.model(x, poi)

        predict = self.scaler.inverse_transform(output)

        y = y.mean(dim=-1, keepdim=True)
        loss_pred = self.loss_pred(predict, y)

        mae = masked_mae(predict, y, null_val=np.Inf).item()
        mape = masked_mape(predict, y, null_val=np.Inf).item()
        rmse = masked_rmse(predict, y, null_val=np.Inf).item()

        return loss_pred.item(), mae, mape, rmse

