import os
import csv
import time
import torch
import numpy as np
from opt import opter
from world import config
from pprint import pprint
from data_loader import load_data
from utils import set_seed, evaluate_and_log


if __name__ == '__main__':
    print('explicit feedback')
    print('cross entropy loss')
    print('no batch')
    print('o_emb_ex_restrict_dann_var')
    res_per_se = []
    for se in range(5):
        config['seed'] = se

        set_seed(config['seed'])
        dataset = load_data(config)
        file_name = 'u_' + str(config['data_name']) + 't' + str(config['thres']) + '_' \
                    + f"o_{config['emb_dim']}{config['dis_epoch']}{config['thres_epoch']}" \
                      f"{config['emb_model_name']}_{config['dc_model_name']}_ex_" \
                      f"DIRL_seed{config['seed']}_ex1"

        config['file_name'] = file_name
        config['num_users'] = dataset.num_users
        config['num_items'] = dataset.num_items
        config['biased_train_shape'] = dataset.all_biased_train.shape
        config['uni_val_shape'] = dataset.uni_val.shape
        config['uni_test_shape'] = dataset.uni_test.shape
        config['test_train_ratio'] = dataset.test_train_ratio
        print('===============config====================')
        pprint(config)
        print('===============config====================')

        file_name = config['file_name']
        hyper_param = str(config['alpha1']) + '_' + str(config['alpha2']) + '_' \
                      + str(config['eta']) + '_' \
                      + str(config['dc_lr']) + '_' + str(config['dc_decay']) + '_' \
                      + str(config['emb_lr']) + '_' + str(config['emb_decay'])


        op = opter(config)

        # train procedure
        train_user_tensor = torch.from_numpy(dataset.all_biased_train[:, 0]).long().to(config['device'])
        train_item_tensor = torch.from_numpy(dataset.all_biased_train[:, 1]).long().to(config['device'])
        train_label_tensor = torch.from_numpy(dataset.all_biased_train[:, 2]).float().to(config['device'])

        temp_test_metric_list1 = []
        res_list1 = [0, 0]

        for epoch in range(1, config['max_epochs'] + 1):
            epoch_start = time.time()
            op.update_and_log(train_user_tensor, train_item_tensor, train_label_tensor,
                              epoch)
            train_over = time.time()

            if epoch % 25 == 0 or epoch == 1:
                eval_start = time.time()
                uni_val_auc, uni_val_ndcg, uni_test_metric_list = evaluate_and_log(
                    config, epoch, op.emb_model, uni_val=dataset.uni_val, uni_test=dataset.uni_test)

                if uni_val_auc > res_list1[1]:
                    res_list1[0] = epoch
                    res_list1[1] = uni_val_auc
                    temp_test_metric_list1 = uni_test_metric_list

            if epoch % 1000 == 0:
                print(f'{epoch} train_cost, eval_cost, epoch_cost', train_over - epoch_start,
                      time.time() - eval_start, time.time() - epoch_start)

        """
        res_list1.extend(temp_test_metric_list1)
        res_list1.append(hyper_param)
        with open('../metric/' + file_name + '1.csv', mode='a', newline='') as tar_file1:
            writer1 = csv.writer(tar_file1)
            writer1.writerow(res_list1)
        """
        res_list1.extend(temp_test_metric_list1)
        res_per_se.append(res_list1)
        print(f'seed{se} val auc, test ndcg@5, precision@5, recall@5', res_list1[1:])

    res_per_se = np.array(res_per_se)
    average_res = res_per_se.mean(axis=0)
    print('avg val auc, test ndcg@5, precision@5, recall@5', average_res[1:])















































































