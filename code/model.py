import torch
from torch import nn
import numpy as np

class sp_dc_mlp_bottle(nn.Module):
    def __init__(self, config):
        super(sp_dc_mlp_bottle, self).__init__()
        self.emb_dim = config['emb_dim']
        self.input_dim = config['emb_dim']

        self.bottle_trans = nn.Parameter(torch.zeros(self.input_dim, self.emb_dim))
        self.bottle_bias = nn.Parameter(torch.zeros(1, self.emb_dim))

        self.classifier_trans1 = nn.Parameter(torch.zeros(self.emb_dim, self.emb_dim))
        self.classifier_bias1 = nn.Parameter(torch.zeros(1, self.emb_dim))
        self.classifier_trans2 = nn.Parameter(torch.zeros(self.emb_dim, 1))
        self.classifier_bias2 = nn.Parameter(torch.zeros(1))

        self.non_linear = nn.Softplus()
        nn.init.xavier_normal_(self.bottle_trans)
        nn.init.xavier_normal_(self.classifier_trans1)
        nn.init.xavier_normal_(self.classifier_trans2)


        print('---------------------------------sp_dc_mlp_bottle---------------------------------------')
        print('sp_dc_mlp_bottle init')
        print('input_dim', self.input_dim)
        print('emb_dim', self.emb_dim)
        for p in self.parameters():
            print('sp_dc_mlp_bottle para', p, p.size())
        print('---------------------------------sp_dc_mlp_bottle---------------------------------------')

    def out_forward(self, event_emb, dt):
        prob = self.mlp(event_emb, dt).squeeze()

        return prob

    def mlp(self, event_emb, dt):
        if dt:
            bt_emb = torch.matmul(event_emb, self.bottle_trans.detach()) + self.bottle_bias.detach()
            bt_emb = self.non_linear(bt_emb)

            ct_emb1 = torch.matmul(bt_emb, self.classifier_trans1.detach()) + self.classifier_bias1.detach()
            ct_emb1 = self.non_linear(ct_emb1)
            rating = torch.matmul(ct_emb1, self.classifier_trans2.detach()) + self.classifier_bias2.detach()
        else:
            bt_emb = torch.matmul(event_emb, self.bottle_trans) + self.bottle_bias
            bt_emb = self.non_linear(bt_emb)

            ct_emb1 = torch.matmul(bt_emb, self.classifier_trans1) + self.classifier_bias1
            ct_emb1 = self.non_linear(ct_emb1)
            rating = torch.matmul(ct_emb1, self.classifier_trans2) + self.classifier_bias2

        prob = torch.sigmoid(rating)

        return prob


class mf(nn.Module):
    def __init__(self, config):
        super(mf, self).__init__()
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.emb_dim = config['emb_dim']
        self.user_embedding = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.emb_dim)
        self.item_embedding = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.emb_dim)
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)

        print('---------------------------------mf---------------------------------------')
        print('mf init')
        print('num_users num_items emb_dim', self.num_users, self.num_items,  self.emb_dim)
        for p in self.parameters():
            print('mf para', p, p.size())
        print('---------------------------------mf---------------------------------------')


    def out_forward(self, user_index, item_index):
        user_emb = self.user_embedding(user_index)
        item_emb = self.item_embedding(item_index)
        rating = (user_emb * item_emb).sum(dim=1)

        return rating


    def get_event_emb(self, user_index, item_index):
        user_emb = self.user_embedding(user_index)
        item_emb = self.item_embedding(item_index)
        event_emb = user_emb * item_emb

        return event_emb


    def get_rating(self, user_index, item_index):
        with torch.no_grad():
            rating = self.out_forward(user_index, item_index)

            return rating

