import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils

def transform_token2index(sequences):
    token2index = {'A': 1, 'G': 2, 'U': 3, 'C': 4, 'X': 0}
    max_len = 41
    token_list = []
    # 遍历序列，生成token索引
    padded_seqs = []
    for seq in sequences:
        if len(seq) < max_len:
            padded_seq = seq + 'X' * (max_len - len(seq))  # 不足70补X，X对应全0特征
        else:
            padded_seq = seq[:max_len]  # 超过70截断
        padded_seqs.append(padded_seq)
    for seq in padded_seqs:
        seq_id = [token2index[aa] for aa in seq]
        token_list.append(torch.tensor(seq_id))
    return token_list
def construct_dataset(seqs, labels, train=True, batch_size=64):
    seqs, labels = list(seqs), list(labels)
    token_list = transform_token2index(seqs)
    seqs_data = rnn_utils.pad_sequence(token_list, batch_first=True)  # Fill the sequence to the same length
    data_loader = Data.DataLoader(Data.TensorDataset(seqs_data, torch.LongTensor(labels)),
                                  batch_size=batch_size,
                                  shuffle=train,
                                  drop_last=False)
    return data_loader

def load_bench_data(file):
    #将训练集划分：训练和验证集(非交叉验证划分)
    tmp = pd.read_csv(file)
    seqs, labels = tmp["seq"].values.tolist(), tmp["label"].values.tolist() # tmp[0].values：第0列（序列列）的numpy数组（如array(["ARND", "CEFG", ...])）；tolist()：将numpy数组转为Python列表，方便后续处理；
    train_seqs, test_seqs, train_labels, test_labels = train_test_split(seqs, labels, test_size=0.2, random_state=42)
    train_iter = construct_dataset(train_seqs,train_labels, train =True)
    valid_iter = construct_dataset(test_seqs,test_labels, train = False)
    return train_iter, valid_iter

def load_ind_data(file):
    # 独立测试集
    tmp = pd.read_csv(file)
    seqs, labels = tmp["seq"].values.tolist(), tmp["label"].values.tolist()
    data_iter = construct_dataset(seqs, labels, train=False)
    return data_iter
