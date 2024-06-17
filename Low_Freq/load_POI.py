import numpy as np
import argparse
import torch
import math
import torch.nn.functional as F
import scipy.sparse as sp

def load_POI_NYC(data_file, k):
    poi_data = np.load(data_file)
    poi_data = torch.tensor(poi_data).permute(2, 0, 1)
    poi_data = poi_data[18 - k:, ...]

    return poi_data

def load_POI_BJ(data_file, k):
    poi_data = np.load(data_file)
    poi_data = torch.tensor(poi_data).permute(2, 0, 1)
    poi_data = poi_data[10 - k:, ...]

    return poi_data

def ts_sequene_to_instance(
        data, x_offsets, y_offsets
):
    num_sample = len(data)
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_sample - abs(max(y_offsets)))
    for t in range(min_t, max_t):
        x.append(data[t + x_offsets])
        y.append(data[t + y_offsets])
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)

    return x, y

def generate_ts_list(args):
    if args.data == 'NYC':
        ts = list(range(1, 29))
        residual = list(range(1, 10))
        ts_list = np.array(ts * 52 + residual)
    elif args.data == 'BJ':
        ts = list(range(1, 29))
        residual = list(range(1, 8))
        ts_list = np.array(ts * 15 + residual)
    num_sample = ts_list.shape[0]
    num_train = round(num_sample * args.train_ratio)
    num_val = round(num_sample * args.val_ratio)
    num_test = round(num_sample - num_train - num_val)
    train_data = ts_list[:num_train]
    val_data = ts_list[num_train: num_train + num_val]
    test_data = ts_list[-num_test:]

    # 0 is teh latest observed sample
    x_offsets = np.sort(np.concatenate((np.arange(-(args.x_offsets - 1), 1, 1),)))
    y_offsets = np.sort(np.arange(args.y_start, (args.y_offsets + 1), 1))
    poi_train, _ = ts_sequene_to_instance(train_data, x_offsets, y_offsets)
    poi_val, _ = ts_sequene_to_instance(val_data, x_offsets, y_offsets)
    poi_test, _ = ts_sequene_to_instance(test_data, x_offsets, y_offsets)


    return poi_train, poi_val, poi_test

def poi_repeat(poi):
    sample_num = poi.shape[0]
    repeat_poi = []
    for i in range(sample_num):
        data = poi[i]
        repeat_poi.extend(data.repeat(24, 1, 1, 1))

    poi = torch.stack(repeat_poi)

    return poi

def poi_sequene_to_instance(args):
    if args.data == 'NYC':
        poi_data = load_POI_NYC(args.NYC_POI, args.K)
    elif args.data == 'BJ':
        poi_data = load_POI_BJ(args.BJ_POI, args.K)

    poi_train, poi_val, poi_test = generate_ts_list(args)
    slide_win_flag = args.K - 1
    train, val, test = [], [], []
    for i in poi_train:
        mid = math.floor(args.x_offsets / 2) - 1
        if i[mid] == 1:
            slide_win_flag += 1

        offsets = [i for i in range(slide_win_flag-args.K+1, slide_win_flag+1)]
        train.append(poi_data[offsets, ...])

    for i in poi_val:
        mid = math.floor(args.x_offsets / 2) - 1
        if i[mid] == 1:
            slide_win_flag += 1

        offsets = [i for i in range(slide_win_flag-args.K+1, slide_win_flag+1)]
        val.append(poi_data[offsets, ...])

    for i in poi_test:
        mid = math.floor(args.x_offsets / 2) - 1
        if i[mid] == 1:
            slide_win_flag += 1

        offsets = [i for i in range(slide_win_flag-args.K+1, slide_win_flag+1)]
        test.append(poi_data[offsets, ...])

    train = torch.tensor(np.stack(train, axis=0))
    val = torch.tensor(np.stack(val, axis=0))
    test = torch.tensor(np.stack(test, axis=0))

    train = poi_repeat(train)
    val = poi_repeat(val)
    test = poi_repeat(test)

    return train, val, test


def calc_gso(dir_adj, gso_type):
    n_vertex = dir_adj.shape[0]

    if sp.issparse(dir_adj) == False:
        dir_adj = sp.csc_matrix(dir_adj)
    elif dir_adj.format != 'csc':
        dir_adj = dir_adj.tocsc()

    id = sp.identity(n_vertex, format='csc')

    # Symmetrizing an adjacency matrix   对称邻接矩阵
    adj = dir_adj + dir_adj.T.multiply(dir_adj.T > dir_adj) - dir_adj.multiply(dir_adj.T > dir_adj)  # 有向图转换为无向图
    # adj = 0.5 * (dir_adj + dir_adj.transpose())

    if gso_type == 'sym_renorm_adj' or gso_type == 'rw_renorm_adj' \
            or gso_type == 'sym_renorm_lap' or gso_type == 'rw_renorm_lap':
        adj = adj + id  # 邻接矩阵＋单位阵

    if gso_type == 'sym_norm_adj' or gso_type == 'sym_renorm_adj' \
            or gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
        row_sum = adj.sum(axis=1).A1
        row_sum_inv_sqrt = np.power(row_sum, -0.5)
        row_sum_inv_sqrt[np.isinf(row_sum_inv_sqrt)] = 0.
        deg_inv_sqrt = sp.diags(row_sum_inv_sqrt, format='csc')
        # A_{sym} = D^{-0.5} * A * D^{-0.5}
        sym_norm_adj = deg_inv_sqrt.dot(adj).dot(deg_inv_sqrt)

        if gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
            sym_norm_lap = id - sym_norm_adj
            gso = sym_norm_lap
        else:
            gso = sym_norm_adj

    elif gso_type == 'rw_norm_adj' or gso_type == 'rw_renorm_adj' \
            or gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
        row_sum = np.sum(adj, axis=1).A1
        row_sum_inv = np.power(row_sum, -1)
        row_sum_inv[np.isinf(row_sum_inv)] = 0.
        deg_inv = np.diag(row_sum_inv)
        # A_{rw} = D^{-1} * A
        rw_norm_adj = deg_inv.dot(adj)

        if gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
            rw_norm_lap = id - rw_norm_adj
            gso = rw_norm_lap
        else:
            gso = rw_norm_adj

    else:
        raise ValueError(f'{gso_type} is not defined.')

    return gso

def calc_gso_batch(batch_adj, gso_type):

    batch = batch_adj.shape[0]
    gso_batch = torch.zeros_like(batch_adj)  # 这是稠密矩阵

    for i in range(batch):
        dir_adj = batch_adj[i]  # 取出第i个adj，分别进行拉普拉斯变换
        n_vertex = dir_adj.shape[0]

        if sp.issparse(dir_adj) == False:  # 转为稀疏矩阵表示
            sp_adj = sp.csc_matrix(dir_adj)
        elif dir_adj.format != 'csc':
            sp_adj = dir_adj.tocsc()

        id = sp.identity(n_vertex, format='csc')  # 返回稀疏矩阵的单位阵

        # Symmetrizing an adjacency matrix   对称邻接矩阵
        adj = sp_adj + sp_adj.T.multiply(sp_adj.T > sp_adj) - sp_adj.multiply(sp_adj.T > sp_adj)  # 有向图转换为无向图
        # adj = 0.5 * (dir_adj + dir_adj.transpose())

        if gso_type == 'sym_renorm_adj' or gso_type == 'rw_renorm_adj' \
                or gso_type == 'sym_renorm_lap' or gso_type == 'rw_renorm_lap':
            adj = adj + id  # 邻接矩阵＋单位阵

        if gso_type == 'sym_norm_adj' or gso_type == 'sym_renorm_adj' \
                or gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
            row_sum = adj.sum(axis=1).A1
            row_sum_inv_sqrt = np.power(row_sum, -0.5)
            row_sum_inv_sqrt[np.isinf(row_sum_inv_sqrt)] = 0.
            deg_inv_sqrt = sp.diags(row_sum_inv_sqrt, format='csc')
            # A_{sym} = D^{-0.5} * A * D^{-0.5}
            sym_norm_adj = deg_inv_sqrt.dot(adj).dot(deg_inv_sqrt)

            if gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
                sym_norm_lap = id - sym_norm_adj
                gso = sym_norm_lap
            else:
                gso = sym_norm_adj

        elif gso_type == 'rw_norm_adj' or gso_type == 'rw_renorm_adj' \
                or gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
            row_sum = np.sum(adj, axis=1).A1
            row_sum_inv = np.power(row_sum, -1)
            row_sum_inv[np.isinf(row_sum_inv)] = 0.
            deg_inv = np.diag(row_sum_inv)
            # A_{rw} = D^{-1} * A
            rw_norm_adj = deg_inv.dot(adj)

            if gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
                rw_norm_lap = id - rw_norm_adj
                gso = rw_norm_lap
            else:
                gso = rw_norm_adj

        else:
            raise ValueError(f'{gso_type} is not defined.')

        # 得到的gso是稀疏矩阵
        gso_batch[i] = torch.tensor(gso.todense())# 转为稠密矩阵存储

    return gso_batch
