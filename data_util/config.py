import argparse
import random

import torch.nn

import model.loss as loss

import os


class Config:
    def __init__(self):
        # path
        self.data_path: str
        self.vocab_path: str
        self.model_save_dir: str
        self.model_path: str

        # model hyperparameters
        self.hidden_dim: int
        self.emb_dim: int
        self.emb_dropout: float
        self.lstm_dropout: float
        self.attention_dropout: float
        self.num_attention_heads: int

        # hyperparameters
        self.max_len: int
        self.lr_scheduler_gama: float
        self.batch_size: int
        self.epoch: int
        self.seed: int
        self.lr: float
        self.eps: float
        self.use_gpu: bool

        # noise
        self.user_mean: float
        self.user_std: float
        self.user_count: int
        self.user_noise_p: list

        # loss
        self.loss_type: list
        self.alpha: float
        self.beta: float
        self.scale: float
        self.q: float
        self.loss_fn = None

    def parse_cli(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--data_path", default="./data/atis/", type=str)
        parser.add_argument("--vocab_file", default="vocab.txt", type=str)
        parser.add_argument("--model_save_dir", default="./ckpt/", type=str)
        parser.add_argument("--model_path", default="atis_model.bin", type=str)
        parser.add_argument("--hidden_dim", default=128, type=int)
        parser.add_argument("--emb_dim", default=300, type=int)
        parser.add_argument("--emb_dropout", default=0.8, type=float)
        parser.add_argument("--lstm_dropout", default=0.5, type=float)
        parser.add_argument("--attention_dropout", default=0.1, type=float)
        parser.add_argument("--num_attention_heads", default=8, type=int)
        parser.add_argument("--max_len", default=32, type=int)
        parser.add_argument("--lr_scheduler_gama", default=0.5, type=float)
        parser.add_argument("--batch_size", default=32, type=int)
        parser.add_argument("--epoch", default=100, type=int)
        parser.add_argument("--seed", default=9, type=int)
        parser.add_argument("--lr", default=0.001, type=float)
        parser.add_argument("--eps", default=1e-12, type=float)
        parser.add_argument("--use_gpu", default=True, type=bool)
        parser.add_argument("--user_mean", nargs="+", default=[0.2] * 10, type=float)
        parser.add_argument("--user_std", nargs="+", default=[0.2] * 10, type=float)
        parser.add_argument("--loss", nargs="+", default=["ce"],
                            choices=["ce", "nce", "sce", "rce", "nrce", "gce", "ngce", "mae", "nmae", "nlnl", "fl",
                                     "nfl", "dmi"], type=str)
        parser.add_argument("--alpha", default=1.0, type=float)
        parser.add_argument("--beta", default=1.0, type=float)
        parser.add_argument("--scale", default=1.0, type=float)
        parser.add_argument("--q", default=0.7, type=float)
        args = parser.parse_args()

        self.data_path = args.data_path
        self.vocab_path = self.data_path + args.vocab_file
        self.model_save_dir = args.model_save_dir
        self.model_path = args.model_path
        self.hidden_dim = args.hidden_dim
        self.emb_dim = args.emb_dim
        self.emb_dropout = args.emb_dropout
        self.lstm_dropout = args.lstm_dropout
        self.attention_dropout = args.attention_dropout
        self.num_attention_heads = args.num_attention_heads
        self.max_len = args.max_len
        self.lr_scheduler_gama = args.lr_scheduler_gama
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.seed = args.seed
        self.lr = args.lr
        self.eps = args.eps
        self.use_gpu = args.use_gpu
        self.user_mean = args.user_mean
        self.user_std = args.user_std
        assert type(self.user_mean) == list and type(self.user_std) == list
        assert len(self.user_mean) == len(self.user_std), "Length of `user_mean` != `user_std`"
        self.user_count = len(self.user_mean)
        self.user_noise_p = [random.gauss(mean, std) for mean, std in zip(config.user_mean, config.user_std)]
        self.loss_type = args.loss
        self.alpha = args.alpha
        self.beta = args.beta
        self.scale = args.scale
        self.q = args.q
        n_class = 26  # ATIS
        self.loss_fn = self.get_loss_fn(set(self.loss_type), n_class, self.alpha, self.beta, self.scale, self.q)

    def get_loss_fn(self, loss_type: set, n_class: int, alpha: float, beta: float, scale: float, q: float):
        if loss_type == {"ce"}:
            return torch.nn.CrossEntropyLoss()
        elif loss_type == {"nce"}:
            return loss.NormalizedCrossEntropy(n_class, scale)
        elif loss_type == {"sce"}:
            return loss.SCELoss(alpha, beta, n_class)
        elif loss_type == {"rce"}:
            return loss.ReverseCrossEntropy(n_class, scale)
        elif loss_type == {"nrce"}:
            return loss.NormalizedReverseCrossEntropy(n_class, scale)
        elif loss_type == {"gce"}:
            return loss.GeneralizedCrossEntropy(n_class, q)
        elif loss_type == {"ngce"}:
            return loss.NormalizedGeneralizedCrossEntropy(n_class, scale, q)
        elif loss_type == {"mae"}:
            return loss.MeanAbsoluteError(n_class, scale)
        elif loss_type == {"nmae"}:
            return loss.NormalizedMeanAbsoluteError(n_class, scale)
        elif loss_type == {"nlnl"}:
            raise NotImplementedError
        elif loss_type == {"fl"}:
            raise NotImplementedError
        elif loss_type == {"nfl"}:
            raise NotImplementedError
        elif loss_type == {"dmi"}:
            raise NotImplementedError
        elif loss_type == {"nce", "rce"}:
            return loss.NCEandRCE(alpha, beta, n_class)
        elif loss_type == {"nce", "mae"}:
            return loss.NCEandMAE(alpha, beta, n_class)
        elif loss_type == {"gce", "nce"}:
            return loss.GCEandNCE(alpha, beta, n_class, q)
        elif loss_type == {"gce", "rce"}:
            return loss.GCEandRCE(alpha, beta, n_class, q)
        elif loss_type == {"gce", "mae"}:
            return loss.GCEandMAE(alpha, beta, n_class, q)
        elif loss_type == {"ngce", "nce"}:
            return loss.NGCEandNCE(alpha, beta, n_class, q)
        elif loss_type == {"ngce", "rce"}:
            return loss.NGCEandRCE(alpha, beta, n_class, q)
        elif loss_type == {"ngce", "mae"}:
            return loss.NGCEandMAE(alpha, beta, n_class, q)
        elif loss_type == {"mae", "rce"}:
            return loss.MAEandRCE(alpha, beta, n_class)
        elif loss_type == {"nfl", "nce"}:
            raise NotImplementedError
        elif loss_type == {"nfl", "rce"}:
            raise NotImplementedError
        elif loss_type == {"nfl", "mae"}:
            raise NotImplementedError
        else:
            raise ValueError

    def dump(self):
        print(
            f"==========================================\n"
            f"data_path: {self.data_path}\n"
            f"vocab_path: {self.vocab_path}\n"
            f"model_save_dir: {self.model_save_dir}\n"
            f"model_path: {self.model_path}\n"
            f"hidden_dim: {self.hidden_dim}\n"
            f"emb_dim: {self.emb_dim}\n"
            f"emb_dropout: {self.emb_dropout}\n"
            f"lstm_dropout: {self.lstm_dropout}\n"
            f"attention_dropout: {self.attention_dropout}\n"
            f"num_attention_heads: {self.num_attention_heads}\n"
            f"max_len: {self.max_len}\n"
            f"lr_scheduler_gama: {self.lr_scheduler_gama}\n"
            f"batch_size: {self.batch_size}\n"
            f"epoch: {self.epoch}\n"
            f"seed: {self.seed}\n"
            f"lr: {self.lr}\n"
            f"eps: {self.eps}\n"
            f"use_gpu: {self.use_gpu}\n"
            f"user_count: {self.user_count}\n"
            f"user_mean: {self.user_mean}\n"
            f"user_std: {self.user_std}\n"
            f"user_noise_p: {self.user_noise_p}\n"
            f"loss: {'+'.join(self.loss_type)}\n"
            f"alpha: {self.alpha}\n"
            f"beta: {self.beta}\n"
            f"scale: {self.scale}\n"
            f"q: {self.q}\n"
            f"==========================================\n"
        )


config = Config()
