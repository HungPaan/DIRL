import numpy as np
from os.path import join

class load_data(object):
    def __init__(self, config):
        data_path = config['data_path']
        data_name = config['data_name']
        thres = config['thres']
        self.all_biased_train = np.loadtxt(join(data_path + data_name + '/', f'{data_name}t{thres}p5_train.txt')).astype(int)
        self.uni_val = np.loadtxt(join(data_path + data_name + '/', f'uni_{data_name}t{thres}p5_val.txt')).astype(int)
        self.uni_test = np.loadtxt(join(data_path + data_name + '/', f'uni_{data_name}t{thres}p5_test.txt')).astype(int)

        self.num_users = int(np.max(self.all_biased_train[:, 0])) + 1
        self.num_items = int(np.max(self.all_biased_train[:, 1])) + 1


        print(f'{data_name} num_users', self.num_users)
        print(f'{data_name} num_items', self.num_items)

        num_train_pos = float((self.all_biased_train[:, 2] == 1).sum())
        num_train_neg = float((self.all_biased_train[:, 2] == 0).sum())

        num_test_pos = float((self.uni_test[:, 2] == 1).sum())
        num_test_neg = float((self.uni_test[:, 2] == 0).sum())

        self.test_train_ratio = np.array([num_test_neg / num_train_neg, num_test_pos / num_train_pos])
        print('test_train_ratio', self.test_train_ratio)



        # train & validation & test
        print(f'{data_name} all_biased_train', self.all_biased_train, self.all_biased_train.shape)
        print(f'{data_name} all_biased_train_pos', (self.all_biased_train[:, 2] == 1).sum())
        print(f'{data_name} all_biased_train_neg', (self.all_biased_train[:, 2] == 0).sum())

        print(f'{data_name} uni_val', self.uni_val, self.uni_val.shape)
        print(f'{data_name} uni_val_pos', (self.uni_val[:, 2] == 1).sum())
        print(f'{data_name} uni_val_neg', (self.uni_val[:, 2] == 0).sum())
        print(f'{data_name} uni_test', self.uni_test, self.uni_test.shape)
        print(f'{data_name} uni_test_pos', (self.uni_test[:, 2] == 1).sum())
        print(f'{data_name} uni_test_neg', (self.uni_test[:, 2] == 0).sum())





if __name__ == '__main__':
    print('=================data_loader=====================')






