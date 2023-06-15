from parse import parse_args
args = parse_args()
config = {}


config['data_path'] = args.data_path
config['data_name'] = args.data_name
config['thres'] = args.thres

config['device'] = args.device
config['max_epochs'] = args.max_epochs

config['seed'] = args.seed
config['top_k'] = args.top_k

config['emb_model_name'] = args.emb_model_name
config['dc_model_name'] = args.dc_model_name

config['emb_dim'] = args.emb_dim
config['emb_lr'] = args.emb_lr
config['emb_decay'] = args.emb_decay
config['dc_lr'] = args.dc_lr
config['dc_decay'] = args.dc_decay

config['eta'] = args.eta
config['alpha1'] = args.alpha1
config['alpha2'] = args.alpha2

config['dis_epoch'] = args.dis_epoch
config['thres_epoch'] = args.thres_epoch

config['val_ratio'] = args.val_ratio

