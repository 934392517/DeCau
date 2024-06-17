from causal_model_trainer import *
from util import *
from my_logging import get_logger
import time
import copy
import os
from Low_Freq.load_POI import poi_sequene_to_instance
from earlystopping import EarlyStopping
from load_train_data import load_flow_data
from configs.args_BJ import get_parameters

def main(args):
    x_train, y_train, x_val, y_val, x_test, y_test = load_flow_data(args)
    poi_train, poi_val, poi_test = poi_sequene_to_instance(args)

    max_min = np.percentile(x_train, 85), 0
    scaler = StandardScale(mean=np.mean(x_train), std=np.std(x_train))
    x_train, x_val, x_test = scaler.transform(x_train), scaler.transform(x_val), scaler.transform(x_test)

    dataloader_train = DataLoader(MyDataset(x_train, poi_train, y_train), args.batch, shuffle=True, drop_last=True)
    dataloader_val = DataLoader(MyDataset(x_val, poi_val, y_val), args.batch, shuffle=False, drop_last=True)
    dataloader_test = DataLoader(MyDataset(x_test, poi_test, y_test), args.batch, shuffle=False, drop_last=True)

    geo_adj = get_norm_adj(args.static_adj)
    adjs = [geo_adj]
    static_norm_adjs = [torch.tensor(adj).to(torch.float32).to(args.device) for adj in adjs]

    # 设置log
    logger, log_dir = get_logger(args.logger_dir, 'DeCau', False)
    logger.info(args)
    logger.info('start training...')

    Ko = args.pre_ts - (args.Kt - 1) * 2 * args.stblock_num
    blocks = []
    blocks.append([64])
    for l in range(args.stblock_num):
        blocks.append([128, 64, 64])
    if Ko == 0:
        blocks.append([128])
    elif Ko > 0:
        blocks.append([128, 128])
    blocks.append([1])

    model = Model(args.decomp_kernel, args.num_layer, args.input_dim, args.hidden_dim_low, args.hidden_dim_high,
                  args.num_context, args.node_num, args.num_pseudo,
                  args.dropout, args.his_ts, args.pre_ts, args.pre_num, args.device, blocks, static_norm_adjs,
                  args.gcn_depth, args.dropout_type)


    engine = Trainer(model, args.lr, max_min, geo_adj, args.kl_ratio, args.weight_decay, args.milestones,
                     args.lr_decay_ratio, args.min_lr, args.loss_type,
                     args.max_grad_norm, scaler, args.device)

    es = EarlyStopping(mode='min', min_delta=0.0, patience=args.patience)


    his_loss, val_time, train_time = [], [], []
    best_epoch, best_loss = float('inf'), float('inf')
    for i in range(1, args.epochs + 1):
        train_loss, loss_kl, train_mae, train_mape, train_rmse = [], [], [], [], []
        t1 = time.time()
        for iter, item in enumerate(dataloader_train):
            trainx, train_poi, trainy = item['flow'].to(torch.float32).to(args.device), item['poi'].to(
                torch.float32).to(args.device), item['label'].to(torch.float32).to(args.device)
            metrics = engine.train(trainx, train_poi, trainy)
            train_loss.append(metrics[0])
            loss_kl.append(metrics[1])
            train_mae.append(metrics[2])
            train_mape.append(metrics[3])
            train_rmse.append(metrics[4])

        engine.scheduler.step()

        t2 = time.time()
        train_time.append(t2 - t1)

        valid_loss, valid_mae, valid_mape, valid_rmse = [], [], [], []
        s1 = time.time()
        for iter, item in enumerate(dataloader_val):
            valx, val_poi, valy = item['flow'].to(torch.float32).to(args.device), item['poi'].to(torch.float32).to(
                args.device), item['label'].to(torch.float32).to(args.device)
            metrics = engine.eval(valx, val_poi, valy)
            valid_loss.append(metrics[0])
            valid_mae.append(metrics[1])
            valid_mape.append(metrics[2])
            valid_rmse.append(metrics[3])

        s2 = time.time()
        val_time.append(s2 - s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_kl = np.mean(loss_kl)
        mtrain_mae = np.mean(train_mae)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mae = np.mean(valid_mae)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        if mvalid_loss < best_loss:
            best_loss = mvalid_loss
            best_mae, best_mape, best_rmse = mvalid_mae, mvalid_mape, mvalid_rmse
            best_epoch = i
            best_state = copy.deepcopy(engine.model.state_dict())

            # 保存模型
            ckpt_name = "exp{:s}_epoch{:d}_Val_mae:{:.2f}_mape:{:.2f}_rmse:{:.2f}.pth". \
                format(args.model_name, best_epoch, mvalid_mae, mvalid_mape, mvalid_rmse)
            best_mode_path = os.path.join(args.save_path_ckpt, ckpt_name)
            torch.save(best_state, best_mode_path)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, KL_Loss: {:.4f}, Train MAE: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        # print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)
        logger.info(
            log.format(i, mtrain_loss, mtrain_kl, mtrain_mae, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mae,
                       mvalid_mape, mvalid_rmse, (t2 - t1)))

        if es.step(torch.tensor(mvalid_loss)):
            logger.info('Early stopping.')
            break

    ckpt_name = "exp{:s}_epoch{:d}_Val_mae:{:.2f}_mape:{:.2f}_rmse:{:.2f}.pth". \
        format(args.model_name, best_epoch, best_mae, best_mape, best_rmse)
    best_mode_path = os.path.join(args.save_path_ckpt, ckpt_name)
    torch.save(best_state, best_mode_path)

    logger.info("...............Training finished...............")
    logger.info('The best epoch is {}, the best loss is {}'.format(best_epoch, best_loss))
    logger.info('......................................')

    if args.is_test:
        logger.info('......................................')
        logger.info('..................Testing....................')
        logger.info("loading {}".format(best_mode_path))
        save_dict = torch.load(best_mode_path)
        engine.model.load_state_dict(save_dict)
        logger.info('model load success! {}'.format(best_mode_path))

        logger.info('Start testing phase.....')
        engine.model.eval()

        test_loss_list, test_mae_list, test_mape_list, test_rmse_list = [], [], [], []

        for iter, item in enumerate(dataloader_test):
            valx, val_poi, valy = item['flow'].to(torch.float32).to(args.device), item['poi'].to(torch.float32).to(
                args.device), item['label'].to(torch.float32).to(args.device)
            metrics = engine.test(valx, val_poi, valy)
            test_loss_list.append(metrics[0])
            test_mae_list.append(metrics[1])
            test_mape_list.append(metrics[2])
            test_rmse_list.append(metrics[3])

        mtest_loss = np.mean(test_loss_list)
        mtest_mae = np.mean(test_mae_list)
        mtest_mape = np.mean(test_mape_list)
        mtest_rmse = np.mean(test_rmse_list)

        logger.info('Evaluate best model on test data, Test Loss:{:.4f}'.format(mtest_loss))
        log = 'Test MAE:{:.4f}, Test MAPE:{:.4f}, Test RMSE:{:.4f}'
        logger.info(log.format(mtest_mae, mtest_mape, mtest_rmse))


if __name__ == '__main__':
    args = get_parameters()
    main(args)

