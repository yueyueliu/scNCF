import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable


class NCF(nn.Module):
    def __init__(self,num_users, num_items, MF_embedding_dim, MLP_embedding_dim):
        super(NCF,self).__init__()
        num_items +=1
        num_users +=1
        self.MF_userembed = MF_userEmbedding(num_users, MF_embedding_dim)
        self.MF_itemembed = MF_itemEmbedding(num_items, MF_embedding_dim)
        self.MLP_userembed = MLP_userEmbedding(num_users, MLP_embedding_dim)
        self.MLP_itemembed = MLP_itemEmbedding(num_items, MLP_embedding_dim)
        self.Linear = LinearLayer(2*MLP_embedding_dim, drop_ratio=0)
        self.Predict = PredictLayer(MF_embedding_dim + int(MLP_embedding_dim/2))

        #initial model
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal(m.weight)
            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal(m.weight)





        # forward
    def forward(self, user_inputs, item_inputs):
        MF_out = self.MF_forward(user_inputs, item_inputs)
        MLP_out = self.MLP_forward(user_inputs, item_inputs)
        flatten = torch.cat((MF_out, MLP_out), 1)
        # print(flatten.size())
        y = F.sigmoid(self.Predict(flatten))
        return y

    # MLP_forward
    def MLP_forward(self, user_inputs, item_inputs):
        user_inputs_var, item_inputs_var = Variable(user_inputs), Variable(item_inputs)
        user_embeds = self.MLP_userembed(user_inputs_var)
        item_embeds = self.MLP_itemembed(item_inputs_var)
        flatten = torch.cat((user_embeds, item_embeds), 1)
        # print(flatten.size())
        MLP = self.Linear(flatten)
        return MLP

    # MF_forward
    def MF_forward(self, user_inputs, item_inputs):
        user_inputs_var, item_inputs_var = Variable(user_inputs), Variable(item_inputs)
        user_embeds = self.MF_userembed(user_inputs_var)
        item_embeds = self.MF_itemembed(item_inputs_var)
        MF = torch.mul(user_embeds, item_embeds)  # Element-wise product
        return MF


class MF_userEmbedding(nn.Module):
    def __init__(self, num_users, embedding_dim):
        super(MF_userEmbedding, self).__init__()
        self.userEmbedding = nn.Embedding(num_users, embedding_dim)

    def forward(self, user_inputs):
        user_embeds = self.userEmbedding(user_inputs)
        return user_embeds

class MF_itemEmbedding(nn.Module):
    def __init__(self, num_items, embedding_dim):
        super(MF_itemEmbedding, self).__init__()
        self.itemEmbedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, item_inputs):
        item_embeds = self.itemEmbedding(item_inputs)
        return item_embeds

class MLP_userEmbedding(nn.Module):
    def __init__(self, num_users, embedding_dim):
        super(MLP_userEmbedding, self).__init__()
        self.userEmbedding = nn.Embedding(num_users, embedding_dim)

    def forward(self, user_inputs):
        user_embeds = self.userEmbedding(user_inputs)
        return user_embeds

class MLP_itemEmbedding(nn.Module):
    def __init__(self, num_items, embedding_dim):
        super(MLP_itemEmbedding, self).__init__()
        self.itemEmbedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, item_inputs):
        item_embeds = self.itemEmbedding(item_inputs)
        return item_embeds

class LinearLayer(nn.Module):
    def __init__(self,linear_dim, drop_ratio=0):
        super(LinearLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(linear_dim, int(linear_dim/2)),
            nn.Dropout(drop_ratio),
            nn.ReLU(),
            nn.Linear(int(linear_dim/2), int(linear_dim/4)),
            nn.ReLU(),
            # nn.Linear(linear_dim/4, int(linear_dim/8)),
            # nn.ReLU(),
        )

    def forward(self, x):
        out = self.linear(x)
        return out

class PredictLayer(nn.Module):
    def __init__(self,linear_dim):
        super(PredictLayer, self).__init__()
        self.predict = nn.Linear(linear_dim, 1)

    def forward(self, x):
        out = self.predict(x)
        return out