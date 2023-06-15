import torch
from torch import nn
from model import mf
from model import sp_dc_mlp_bottle

class opter(object):
    def __init__(self, config):
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.topk = config['top_k']
        self.device = config['device']

        self.len_train = config['biased_train_shape'][0]
        self.len_all = self.num_users * self.num_items

        print('=================================opter=======================================')
        print('opter init')

        print('num_users', self.num_users)
        print('num_items', self.num_items)
        print('topk', self.topk)
        print('device', self.device)

        print('len_train', self.len_train)
        print('len_all', self.len_all)

        self.emb_model_name = config['emb_model_name']
        print('emb_model_name', self.emb_model_name)
        self.emb_model = mf(config)

        self.dc_model_name = config['dc_model_name']
        print('dc_model_name', self.dc_model_name)
        self.dc_model = sp_dc_mlp_bottle(config)

        self.thres_epoch = config['thres_epoch']
        self.dis_epoch = config['dis_epoch']
        self.eta = config['eta']
        self.alpha1 = config['alpha1']
        self.alpha2 = config['alpha2']

        self.emb_lr = config['emb_lr']
        self.emb_decay = config['emb_decay']
        self.dc_lr = config['dc_lr']
        self.dc_decay = config['dc_decay']
        self.emb_model.to(self.device)
        self.dc_model.to(self.device)
        self.emb_optimizer = torch.optim.Adam(self.emb_model.parameters(), self.emb_lr, weight_decay=self.emb_decay)
        self.dc_optimizer = torch.optim.Adam(self.dc_model.parameters(), lr=self.dc_lr, weight_decay=self.dc_decay)

        print('thres_epoch', self.thres_epoch)
        print('dis_epoch', self.dis_epoch)
        print('emb_lr', self.emb_lr)
        print('emb_decay', self.emb_decay)
        print('dc_lr', self.dc_lr)
        print('dc_decay', self.dc_decay)

        print('eta', self.eta)
        print('alpha1', self.alpha1)
        print('alpha2', self.alpha2)



        print('=================================opter=======================================')


    def update_and_log(self, train_user_tensor, train_item_tensor, train_label_tensor,
                       epoch):
        batch_uni_index = torch.randint(0, self.len_all, [self.len_train], device=self.device)
        batch_uni_user_tensor = torch.div(batch_uni_index, self.num_items, rounding_mode='floor').long()
        batch_uni_item_tensor = batch_uni_index % self.num_items


        emb_loss = self.update_emb_w_adpt(train_user_tensor, train_item_tensor, train_label_tensor,
                                          batch_uni_user_tensor, batch_uni_item_tensor, epoch)



    def update_emb_w_adpt(self, train_user_tensor, train_item_tensor, train_label_tensor,
                          batch_uni_user_tensor, batch_uni_item_tensor, epoch):

        # embedding
        train_event_emb = self.emb_model.get_event_emb(train_user_tensor, train_item_tensor)
        batch_uni_event_emb = self.emb_model.get_event_emb(batch_uni_user_tensor, batch_uni_item_tensor)

        # d
        for i in range(self.dis_epoch):
            out_dc_train_prob = self.dc_model.out_forward(train_event_emb.detach(), dt=False)
            out_dc_batch_uni_prob = self.dc_model.out_forward(batch_uni_event_emb.detach(), dt=False)

            dc_train_loss = - torch.log(out_dc_train_prob + 1e-12).mean()
            dc_batch_uni_loss = - torch.log((1.0 - out_dc_batch_uni_prob) + 1e-12).mean()
            dis_dc_loss = 0.5 * (dc_train_loss + dc_batch_uni_loss)

            self.dc_optimizer.zero_grad()
            dis_dc_loss.backward()
            self.dc_optimizer.step()


        # biased error
        out_train_rating = self.emb_model.out_forward(train_user_tensor, train_item_tensor)
        out_train_prob = torch.sigmoid(out_train_rating)
        out_train_loss = nn.functional.binary_cross_entropy(out_train_prob, train_label_tensor)


        # ada
        out_dc_train_prob = self.dc_model.out_forward(train_event_emb, dt=True)
        dc_train_loss = (0.5 * torch.log(out_dc_train_prob + 1e-12)
                         + 0.5 * torch.log((1.0 - out_dc_train_prob) + 1e-12)).mean()

        out_dc_batch_uni_prob = self.dc_model.out_forward(batch_uni_event_emb, dt=True)
        dc_batch_uni_loss = (0.5 * torch.log(out_dc_batch_uni_prob + 1e-12)
                             + 0.5 * torch.log((1.0 - out_dc_batch_uni_prob) + 1e-12)).mean()

        gen_dc_loss = 0.5 * (dc_train_loss + dc_batch_uni_loss)


        # index
        train_pos_bool_index = train_label_tensor.bool()
        train_neg_bool_index = (1.0 - train_label_tensor).bool()

        # train + batch_uni cluster
        # train cluster
        train_pos_prototype = train_event_emb[train_pos_bool_index].mean(dim=0, keepdim=True)
        train_neg_prototype = train_event_emb[train_neg_bool_index].mean(dim=0, keepdim=True)
        train_intra_loss = ((train_event_emb[train_pos_bool_index]
                             - train_pos_prototype.detach()).pow(2).sum(dim=1).sum()
                            + (train_event_emb[train_neg_bool_index]
                               - train_neg_prototype.detach()).pow(2).sum(dim=1).sum()) / float(self.len_train)
        train_inter_loss = - (train_neg_prototype - train_pos_prototype.detach()).pow(2).sum()
        train_cluster_reg = train_intra_loss + train_inter_loss


        # batch_uni cluster
        with torch.no_grad():
            batch_uni_rating = self.emb_model.out_forward(batch_uni_user_tensor, batch_uni_item_tensor)
            batch_uni_prob = torch.sigmoid(batch_uni_rating)

        if epoch > self.thres_epoch:
            batch_uni_label_tensor = torch.round(batch_uni_prob)
            pos_num = batch_uni_label_tensor.sum()

            if pos_num.bool() and (self.len_train - pos_num).bool():
                batch_uni_pos_prototype = batch_uni_event_emb[batch_uni_label_tensor.bool()].mean(dim=0, keepdim=True)
                batch_uni_neg_prototype = batch_uni_event_emb[(1.0 - batch_uni_label_tensor).bool()].mean(dim=0,
                                                                                                          keepdim=True)
                batch_uni_intra_loss = ((batch_uni_event_emb[batch_uni_label_tensor.bool()]
                                         - batch_uni_pos_prototype.detach()).pow(2).sum(dim=1).sum()
                                        + (batch_uni_event_emb[(1.0 - batch_uni_label_tensor).bool()]
                                           - batch_uni_neg_prototype.detach()).pow(2).sum(dim=1).sum()) / float(
                    self.len_train)
                batch_uni_inter_loss = - (batch_uni_neg_prototype - batch_uni_pos_prototype.detach()).pow(2).sum()
                batch_uni_cluster_reg = batch_uni_intra_loss + batch_uni_inter_loss
            else:
                batch_uni_cluster_reg = torch.zeros(1, device=self.device)
        else:
            batch_uni_cluster_reg = torch.zeros(1, device=self.device)




        # contrasting
        out_batch_uni_rating = self.emb_model.out_forward(batch_uni_user_tensor, batch_uni_item_tensor)
        st_rating_reg = - torch.log(torch.sigmoid(out_train_rating.detach().mean() - out_batch_uni_rating.mean()) + 1e-12)


        emb_loss = out_train_loss - self.eta * gen_dc_loss \
                   + self.alpha1 * train_cluster_reg + self.alpha1 * batch_uni_cluster_reg \
                   + self.alpha2 * st_rating_reg \

        self.emb_optimizer.zero_grad()
        emb_loss.backward()
        self.emb_optimizer.step()


        emb_loss_scalar = emb_loss.detach()

        return emb_loss_scalar



