# coding=utf-8
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch


class InputFeatures(object):
    def __init__(self, input_id, domain_label_id, input_mask, feature_list):
        self.input_id = input_id
        self.domain_label_id = domain_label_id
        self.input_mask = input_mask
        self.feature = feature_list


def read_corpus(path, max_length, intent2idx, slot2idx, vocab, config, is_train=True):
    """

    :param path: 数据地址
    :param max_length:句子最大长度
    :param intent2idx: 意图标签字典
    :param slot2idx: 槽位标签字典
    :param vocab: word 字典

    :return:
    """
    file = open(path, encoding='utf-8')
    content = file.readlines()
    file.close()
    token_lists, slot_lists, intent_lists, mask_lists = [], [], [], []
    token_list, slot_list = [], []
    over_length = 0
    max_len_word = 0
    for line in content:
        line = line.strip()
        if line != "":
            line = line.split(" ")
            if len(line) == 1:
                intent = line[0]
                intent_lists.append(intent2idx[intent])
            if len(line) == 2:
                token, slot = line[0], line[1]
                max_len_word = max(max_len_word, len(token))
                if token not in vocab:
                    token_list.append(vocab["<unk>"])
                else:
                    token_list.append(vocab[token])
                slot_list.append(slot2idx[slot])
        else:
            if len(token_list) > max_length - 2:
                token_list = token_list[0: (max_length - 2)]
                over_length += 1
            slot_list = slot_list[0: (max_length - 2)]
            token_list = [vocab["</s>"]] + token_list + [vocab["</e>"]]
            slot_list = [slot2idx["<start>"]] + slot_list + [slot2idx["<end>"]]
            mask_list = [1] * len(token_list)
            while len(token_list) < max_length:
                token_list.append(0)
                slot_list.append(slot2idx["<PAD>"])
                mask_list.append(0)
            assert len(token_list) == max_length and len(slot_list) == max_length and len(mask_list) == max_length
            token_lists.append(token_list)
            slot_lists.append(slot_list)
            mask_lists.append(mask_list)
            token_list, slot_list = [], []

    dataset = TensorDataset(torch.LongTensor(token_lists), torch.LongTensor(slot_lists), torch.LongTensor(intent_lists),
                            torch.LongTensor(mask_lists))
    data_loader = DataLoader(dataset, shuffle=is_train, batch_size=config.batch_size)

    print("超过最大长度的样本数量为：", over_length)
    print("样本最大长度量为：", max_len_word)
    return data_loader


def process_emb(embedding, emb_dim):
    embeddings = {}
    embeddings["<pad>"] = np.zeros(emb_dim)
    embeddings["<unk>"] = np.random.uniform(-0.01, 0.01, size=emb_dim)
    embeddings["</s>"] = np.random.uniform(-0.01, 0.01, size=emb_dim)
    embeddings["</e>"] = np.random.uniform(-0.01, 0.01, size=emb_dim)

    for emb in embedding:
        line = emb.strip().split()
        word = line[0]
        word_emb = np.array([float(_) for _ in line[1:]])
        embeddings[word] = word_emb

    vocab_list = list(embeddings.keys())
    word2id = {vocab_list[i]: i for i in range(len(vocab_list))}
    embedding_matrix = np.array(list(embeddings.values()))

    return embedding_matrix, word2id


def lord_label_dict(path):
    label2id = {}
    id2label = {}
    f = open(path, "r", encoding="utf-8")
    for item in f:
        id, label = item.strip().split("\t")
        label2id[label] = int(id)
        id2label[int(id)] = label
    f.close()
    return id2label, label2id
