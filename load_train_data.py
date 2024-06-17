import torch
import numpy as np


def load_flow_data_NYC(train_flow_path, test_flow_path):
    print("data: NYC")
    train_data, test_data = np.load(train_flow_path), np.load(test_flow_path)
    flow_data = np.concatenate([train_data, test_data], axis=1)
    flow_data = flow_data.reshape(flow_data.shape[0], -1)

    return flow_data  # (160, 35280)

def load_flow_data_BJ(flow_path):
    print("data: BJ")
    flow_data = np.loadtxt(flow_path, delimiter=',')

    return flow_data  # 185, 10248

def flow_sequene_to_instance(
        data, x_offsets, y_offsets
):
    num_sample = data.shape[0]
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_sample - abs(max(y_offsets)))
    for t in range(min_t, max_t):
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)

    return x, y

def load_flow_data(args):
    if args.data == 'NYC':
        flow_data = load_flow_data_NYC(args.origin_train_flow_path, args.origin_test_flow_path)
        flow_data = torch.from_numpy(flow_data[:, 28 * 4:-8]).transpose(1, 0)
    else:
        flow_data = load_flow_data_BJ(args.flow_path_BJ)
        flow_data = torch.from_numpy(flow_data.transpose(1, 0))


    num_sample = flow_data.shape[0]
    print('num_sample:', num_sample)
    num_train = round(num_sample * args.train_ratio)
    num_val = round(num_sample * args.val_ratio)
    num_test = round(num_sample - num_train - num_val)
    train_data = flow_data[:num_train]
    val_data = flow_data[num_train: num_train + num_val]
    test_data = flow_data[-num_test:]

    percentiles = [5, 10, 20, 30, 40, 50, 80, 90, 95, 98, 99, 100]
    results = np.percentile(test_data, percentiles, axis=None)
    for i in range(len(percentiles)):
        print(percentiles[i], ':', results[i])

    # 0 is teh latest observed sample
    x_offsets = np.sort(np.concatenate((np.arange(-(args.x_offsets * 24 - 1), 1, 1),)))  # x_/y_offset单位是天，原始数据是小时
    y_offsets = np.sort(np.arange(args.y_start, (args.y_offsets * 24 + 1), 1))
    x_train, y_train = flow_sequene_to_instance(train_data, x_offsets, y_offsets)
    x_val, y_val = flow_sequene_to_instance(val_data, x_offsets, y_offsets)
    x_test, y_test = flow_sequene_to_instance(test_data, x_offsets, y_offsets)

    _, ts_fea, node_num = x_train.shape

    if args.data_type == '4D':
        x_train = torch.from_numpy(x_train).reshape(x_train.shape[0], args.x_offsets, -1, node_num).permute(0, 1, 3, 2)  # [num_sample, his ts, node_num, feature_dim]
        y_train = torch.from_numpy(y_train).reshape(y_train.shape[0], args.y_offsets, -1, node_num).permute(0, 1, 3, 2)  # [num_sample, his ts, node_num, feature_dim]
        x_val = torch.from_numpy(x_val).reshape(x_val.shape[0], args.x_offsets, -1, node_num).permute(0, 1, 3, 2)  # [num_sample, his ts, node_num, feature_dim]
        y_val = torch.from_numpy(y_val).reshape(y_val.shape[0], args.y_offsets, -1, node_num).permute(0, 1, 3, 2)  # [num_sample, his ts, node_num, feature_dim]
        x_test = torch.from_numpy(x_test).reshape(x_test.shape[0], args.x_offsets, -1, node_num).permute(0, 1, 3, 2)  # [num_sample, his ts, node_num, feature_dim]
        y_test = torch.from_numpy(y_test).reshape(y_test.shape[0], args.y_offsets, -1, node_num).permute(0, 1, 3, 2)  # [num_sample, his ts, node_num, feature_dim]

    return x_train.numpy(), y_train.numpy(), x_val.numpy(), y_val.numpy(), x_test.numpy(), y_test.numpy()

