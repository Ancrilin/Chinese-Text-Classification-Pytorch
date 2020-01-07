#!/user/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'TextCNN'
        self.train_path_x = dataset + '/data/SMP_train_x.txt'                                # 训练集
        self.train_path_y = dataset + '/data/SMP_train_intent_y.txt'
        self.dev_path_x = dataset + '/data/SMP_dev_x.txt'                                    # 验证集
        self.dev_path_y = dataset + '/data/SMP_dev_intent_y.txt'
        self.test_path_x = dataset + '/data/SMP_test_x.txt'                                  # 测试集
        self.test_path_y = dataset + '/data/SMP_test_intent_y.txt'
        # self.class_list = [x.strip() for x in open(
        #     dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
        self.vocab_path = dataset + '/data/vocab.pkl'                                # 词表
        self.save_path = dataset + '/saved_dict/SMP_' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/SMP_' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                       # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        # self.num_classes = len(self.class_list)                         # 类别数
        self.num_classes = 20
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256

    def setModeName(self, name):
        self.model_name = name

filepath = 'THUCNews/data/SMP_train_x.txt'
with open(filepath, 'r', encoding='utf-8') as fp:
    for line in fp:
        lin = line.strip()
        print(lin)