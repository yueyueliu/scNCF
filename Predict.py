# -*- coding: utf-8 -*-
import argparse
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from NCF import NCF
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, scale
import os
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

def parse_args():
    parser = argparse.ArgumentParser(description="Run NeuMF.")
    parser.add_argument('--dataset', type=str, default="Leukemia",
                        help='dataset name.')
    parser.add_argument('--num_item', type=int, default=7602,
                        help='Number of item.')
    parser.add_argument('--num_user', type=int, default=391,
                        help='Number of user.')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size.')
    parser.add_argument('--mf_dim', type=int, default=32,
                        help='Embedding size of MF model.')
    parser.add_argument('--mlp_dim', type=int, default=32,
                        help='Embedding size of MLP model.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16]',
                        help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_mf', type=float, default=0,
                        help='Regularization for MF embeddings.')
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each MLP layer. reg_layers[0] is the regularization for embeddings.")
    parser.add_argument('--num_neg', type=int, default=16,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=[0.001, 0.0005, 0.0001],
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    return parser.parse_args()

def load_rating_file_as_matrix_pos(filename):
    num_users, num_items = 0, 0
    with open(filename, "r") as f: 
        line = f.readline() 
        while line != None and line != "":
            arr = line.split("\t")  
            u, i = int(arr[0]), int(arr[1])
            num_users = max(num_users, u) 
            num_items = max(num_items, i)
            line = f.readline()

    # 以下构建矩阵
    mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)  # 初始化矩阵，矩阵第0行列是索引
    with open(filename, "r") as f:  # 读文件为f
        line = f.readline()  # 先读一行
        while line != None and line != "":  # 只要不为空
            arr = line.split("\t")  # 用\t分隔字符串
            user, item = int(arr[0]), int(arr[1])
            mat[user, item] = 1.0  # 有关系置为1
            line = f.readline()  # 再读一行
    return mat  # 返回矩阵（0-1邻接矩阵）

def get_train_instances(num_negatives, train):
    user_input, pos_item_input, neg_item_input = [], [], []
    num_users, num_items = train.shape
    for (u, i) in train.keys():
        # positive instance
        for _ in range(num_negatives):
            user_input.append(u)
            pos_item_input.append(i)
        # negative instances
        for _ in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train:
                j = np.random.randint(num_items)
            while j==0:
                j = np.random.randint(num_items)
            neg_item_input.append(j)
    pi_ni = [[pi, ni] for pi, ni in zip(pos_item_input, neg_item_input)]
    return user_input,pi_ni

def get_train_instances_all(mat):
    user_input, item_input = [], []
    num_users, num_items = mat.shape
    for u in range(1,num_users):
        for i in range(1,num_items):
                user_input.append(u)
                item_input.append(i)
    return user_input, item_input

def get_train_dataloader(user, pi_ni, batch_size):
    train_data = TensorDataset(torch.LongTensor(user), torch.LongTensor(pi_ni))
    user_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return user_train_loader

def get_test_dataloader(user, item):
    train_data = TensorDataset(torch.LongTensor(user), torch.LongTensor(item))
    user_train_loader = DataLoader(train_data)
    return user_train_loader

def training(model, train_loader, epoch_id,learning_rates):
    # learning rate decay
    lr = learning_rates[0]
    if epoch_id >= 15 and epoch_id < 25:
        lr = learning_rates[1]
    elif epoch_id >=20:
        lr = learning_rates[2]
    # lr decay
    if epoch_id % 5 == 0:
        lr /= 2

    # optimizer
    optimizer = optim.RMSprop(model.parameters(), lr)

    losses = []
    for batch_id, (u, pi_ni) in enumerate(train_loader):
        # Data Load
        user_input = u
        pos_item_input = pi_ni[:, 0]
        neg_item_input = pi_ni[:, 1]
        # Forward
        pos_prediction = model(user_input, pos_item_input)
        neg_prediction = model(user_input, neg_item_input)

        # Zero_grad
        model.zero_grad()
        loss = torch.mean((pos_prediction - neg_prediction - 1) ** 2)
        # record loss history
        losses.append(loss.item())
        # Backward
        loss.backward()
        optimizer.step()

    print('Iteration %d, loss is [%.4f ]' % (epoch_id, np.mean(losses)))

def read_labels(ref, return_enc=False):
    ref = pd.read_csv(ref, sep='\t', index_col=0, header=None)

    encode = LabelEncoder()
    ref = encode.fit_transform(ref.values.squeeze())
    classes = encode.classes_
    if return_enc:
        return ref, classes, encode
    else:
        return ref, classes

def roc(actual, predictions):

    false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    print(false_positive_rate, true_positive_rate, thresholds)
    print(roc_auc)

    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b',label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    return roc_auc

def plot_embedding(X, labels, classname=None, classes=None, method='tSNE', cmap='tab20', figsize=(4, 4), markersize=4,
                   marker=None,
                   return_emb=False, save=False, save_emb=False, show_legend=True, show_axis_label=False,
                   **legend_params):
    if marker is not None:
        X = np.concatenate([X, marker], axis=0)
    N = len(labels)
    if X.shape[1] != 2:
        if method == 'tSNE':
            from sklearn.manifold import TSNE
            X = TSNE(n_components=2, random_state=124).fit_transform(X)

    plt.figure(figsize=figsize)
    if classes is None:
        classes = np.unique(labels)

    if cmap is not None:
        cmap = cmap
    elif len(classes) <= 10:
        cmap = 'tab10'
    elif len(classes) <= 20:
        cmap = 'tab20'
    else:
        cmap = 'husl'
    colors = sns.color_palette(cmap, n_colors=len(classes))

    for i, c in enumerate(classes):
        plt.scatter(X[:N][labels == c, 0], X[:N][labels == c, 1], s=markersize, color=colors[i], label=classname[c])
    if marker is not None:
        plt.scatter(X[N:, 0], X[N:, 1], s=10 * markersize, color='black', marker='*')
    #     plt.axis("off")

    legend_params_ = {'loc': 'center left',
                      'bbox_to_anchor': (1.0, 0.45),
                      'fontsize': 10,
                      'ncol': 1,
                      'frameon': False,
                      'markerscale': 1.5
                      }
    legend_params_.update(**legend_params)
    if show_legend:
        plt.legend(**legend_params_)
    sns.despine(offset=10, trim=True)
    if show_axis_label:
        plt.xlabel(method + ' dim 1', fontsize=12)
        plt.ylabel(method + ' dim 2', fontsize=12)

    plt.savefig(save, bbox_inches='tight', dpi = 600)
    plt.show()

    if save_emb:
        np.savetxt(save_emb, X)
    if return_emb:
        return X

def reassign_cluster_with_ref(Y_pred, Y):
    def reassign_cluster(y_pred, index):
        y_ = np.zeros_like(y_pred)
        for i in range(y_pred.size):
            for j in range(index[1].size):
                if y_pred[i]==index[0][j]:
                    y_[i] = index[1][j]

        return y_

    from scipy.optimize import linear_sum_assignment as linear_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    return reassign_cluster(Y_pred, ind)

args = parse_args()
a = args.dataset
filename = './data_RNA/{}/cell-gene-pos.txt'.format(a)
mat = load_rating_file_as_matrix_pos(filename)
num_negatives = args.num_neg
num_items = args.num_item
num_users = args.num_user
mf_dim = args.mf_dim
mlp_dim = args.mlp_dim
batch_size = args.batch_size
num_epochs = args.epochs
learning_rates = args.lr
user_input, pi_ni = get_train_instances(num_negatives,mat)
ncf = NCF(num_users, num_items, mf_dim, mlp_dim)

for epoch in range(num_epochs):
    ncf.train()
    training(ncf, get_train_dataloader(user_input, pi_ni,batch_size), epoch,learning_rates)

ncf.eval()
user_input,item_input = get_train_instances_all(mat)
test_loader = get_test_dataloader(user_input, item_input)
result = []
for batch_id, (u, i) in enumerate(test_loader):
    result.append(ncf(u, i).item())
    # print(batch_id)

pre_mat = np.zeros((num_users,num_items))
for i in range(len(user_input)):
    pre_mat[int(float(user_input[i]-1)),int(float(item_input[i]-1))] = result[i]
pre_mat1 = pd.DataFrame(pre_mat)
pre_mat1.to_csv('{}_imputed.txt'.format(a), sep='\t')

a = args.dataset
indir = "./data_RNA/{}/".format(a)
label, classes = read_labels(os.path.join(indir, "labels.txt"))
k = len(classes)
print("聚类数量=",k)

from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
label_pre = kmeans.fit_predict(pre_mat1)
pred = reassign_cluster_with_ref(label_pre, label)
data = pd.read_csv(os.path.join(indir, "data.txt"), sep='\t')
pd.Series(label_pre, index=data.columns).to_csv('{}_label.txt'.format(a), sep='\t', header=False)
