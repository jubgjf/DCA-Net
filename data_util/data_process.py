# coding=utf-8
import numpy as np
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, path, max_length, intent2idx, slot2idx, vocab):
        file = open(path, encoding='utf-8')
        content = file.readlines()
        file.close()
        token_lists, slot_lists, intent_lists, mask_lists, user_lists = [], [], [], [], []
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

        print("超过最大长度的样本数量为：", over_length)
        print("样本最大长度量为：", max_len_word)

        # TODO
        user_lists = [0] * len(intent_lists)  # only one user, called `user0`

        self.token_lists, self.slot_lists, self.intent_lists, self.mask_lists, self.user_lists, self.len = \
            token_lists, slot_lists, intent_lists, mask_lists, user_lists, len(token_lists)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return torch.LongTensor(self.token_lists[idx]), \
            torch.LongTensor(self.slot_lists[idx]), \
            torch.LongTensor(self.intent_lists)[idx], \
            torch.LongTensor(self.mask_lists[idx]), \
            torch.LongTensor(self.user_lists)[idx], \
            idx


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


def load_label_dict(path):
    label2id = {}
    id2label = {}
    f = open(path, "r", encoding="utf-8")
    for item in f:
        id, label = item.strip().split("\t")
        label2id[label] = int(id)
        id2label[int(id)] = label
    f.close()
    return id2label, label2id
