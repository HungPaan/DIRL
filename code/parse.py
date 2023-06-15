import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="DIRL")
    parser.add_argument('--data_path', type=str, default='../datasets/', help="data path")
    parser.add_argument('--data_name', type=str, default='yahooR3', help="data name")
    parser.add_argument('--thres', type=int, default=4, help="threshold for splitting data")

    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    parser.add_argument('--max_epochs', type=int, default=2000, help='the number of iterations')

    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--top_k', type=int, default=5, help='the number of items recommended to each user')

    parser.add_argument('--emb_model_name', type=str, default='mf', help='mf')
    parser.add_argument('--dc_model_name', type=str, default='sp_dc_mlp_bottle', help='sp_dc_mlp_bottle')

    parser.add_argument('--emb_dim', type=int, default=64, help='dimension of all embeddings')
    parser.add_argument('--emb_lr', type=float, default=0.1, help='learning rate for learning emb_model')
    parser.add_argument('--emb_decay', type=float, default=1e-5, help='weight decay for learning emb_model')
    parser.add_argument('--dc_lr', type=float, default=0.05, help='learning rate for learning dc_model')
    parser.add_argument('--dc_decay', type=float, default=0.0001, help='weight decay for learning dc_model')

    parser.add_argument('--eta', type=float, default=0.1, help='adv')
    parser.add_argument('--alpha1', type=float, default=0.01, help='cluster')
    parser.add_argument('--alpha2', type=float, default=1.0, help='contrast')

    parser.add_argument('--dis_epoch', type=int, default=1, help='5, 10, ...')
    parser.add_argument('--thres_epoch', type=int, default=0, help='5, 10, ...')

    parser.add_argument('--val_ratio', type=float, default=0.05, help='the proportion of validation set')

    return parser.parse_args()

