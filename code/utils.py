import cppimport
ex = cppimport.imp("ex")
import torch
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def com_auc(rank_pos, len_data):
    len_pos = len(rank_pos)
    auc_norm = len_pos * (len_data - len_pos)
    auc_numerator = torch.sum(rank_pos) - (len_pos + 1.0) * len_pos / 2.0
    res = auc_numerator / float(auc_norm)
    return res.item()

def com_rec_metric(ranked_user_label, top_k, num_users):
    ans = ex.gaotest(ranked_user_label, np.array([top_k]), np.array([num_users]))

    return ans[0], ans[1], ans[2]

def com_pre_cl(label_tensor, rating):
    sig_rating = torch.sigmoid(rating)
    pre_label = torch.round(sig_rating)

    out_loss = torch.nn.functional.binary_cross_entropy(sig_rating, label_tensor.float())
    error = (pre_label - label_tensor.float()).abs().sum() / float(len(label_tensor))

    return out_loss, error

def evaluate(topk, device, num_users, user_item_label, model):
    user_array = user_item_label[:, 0]
    item_array = user_item_label[:, 1]
    label_array = user_item_label[:, 2]

    user_tensor = torch.from_numpy(user_array).long().to(device)
    item_tensor = torch.from_numpy(item_array).long().to(device)
    label_tensor = torch.from_numpy(label_array).bool().to(device)
    rating = model.get_rating(user_tensor, item_tensor)

    out_loss, error = com_pre_cl(label_tensor, rating)

    _, sorted_element = torch.sort(rating)
    _, rank = torch.sort(sorted_element)

    rank = rank + 1
    rank_pos = rank[label_tensor]
    auc = com_auc(rank_pos, len(rank))

    sorted_element = sorted_element.cpu().numpy()
    ranked_user_label = user_item_label[:, [0, 2]][sorted_element]
    ranked_user_label = ranked_user_label[::-1, :]

    precision, recall, ndcg = com_rec_metric(ranked_user_label, topk, num_users)

    return out_loss, error, auc, precision, recall, ndcg



def evaluate_and_log(config, epoch, emb_model, uni_val=None, uni_test=None):

    uni_val_out_loss, uni_val_error, uni_val_auc, uni_val_pre, uni_val_recall, \
    uni_val_ndcg = evaluate(config['top_k'], config['device'], config['num_users'], uni_val, emb_model)

    uni_test_out_loss, uni_test_error, uni_test_auc, uni_test_pre, uni_test_recall, \
    uni_test_ndcg = evaluate(config['top_k'], config['device'], config['num_users'], uni_test, emb_model)


    return uni_val_auc, uni_val_ndcg, [uni_test_ndcg, uni_test_pre, uni_test_recall]