# -*- coding: utf-8 -*-
import os
import random
import warnings

import wandb
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange

from data_util.Metrics import IntentMetrics, SlotMetrics, semantic_acc
from data_util.cluster import cluster_draw
from data_util.data_process import *
from data_util.noise import make_noise
from model.Radam import RAdam
from model.joint_model_trans import Joint_model

warnings.filterwarnings("ignore")
assert torch.cuda.is_available()
device = torch.device("cuda", torch.cuda.current_device())


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dev(model, dev_loader, idx2slot, config):
    model.eval()
    eval_loss_intent = 0
    eval_loss_slot = 0
    pred_intents = []
    true_intents = []
    pred_slots = []
    true_slots = []
    for batch in tqdm(dev_loader, desc="Evaluating"):
        inputs, slot_labels, intent_labels, masks, _, _ = batch
        inputs, masks, intent_labels, slot_labels = inputs.cuda(), masks.cuda(), intent_labels.cuda(), slot_labels.cuda()
        logits_intent, logits_slot = model.forward_logit(inputs, masks)
        loss_intent, loss_slot = model.loss1(logits_intent, logits_slot, intent_labels, slot_labels, masks)

        pred_intent, pred_slot = model.pred_intent_slot(logits_intent, logits_slot, masks)
        pred_intents.extend(pred_intent.cpu().numpy().tolist())
        true_intents.extend(intent_labels.cpu().numpy().tolist())
        eval_loss_intent += loss_intent.item()
        eval_loss_slot += loss_slot.item()
        slot_labels = slot_labels.cpu().numpy().tolist()

        for i in range(len(pred_slot)):
            pred = []
            true = []
            for j in range(len(pred_slot[i])):
                pred.append(idx2slot[pred_slot[i][j].item()])
                true.append(idx2slot[slot_labels[i][j]])
            pred_slots.append(pred[1:-1])
            true_slots.append(true[1:-1])
    # slot f1, p, r
    slot_metrics = SlotMetrics(true_slots, pred_slots)
    slot_f1, slot_p, slot_r = slot_metrics.get_slot_metrics()
    # intent f1, p, r
    Metrics_intent = IntentMetrics(pred_intents, true_intents)
    intent_acc = Metrics_intent.accuracy
    data_nums = len(dev_loader.dataset)
    ave_loss_intent = eval_loss_intent * config.batch_size / data_nums
    ave_loss_slot = eval_loss_slot * config.batch_size / data_nums

    sent_acc = semantic_acc(pred_slots, true_slots, pred_intents, true_intents)
    tqdm.write(f"Evaluation - intent_loss: {ave_loss_intent} slot_loss: {ave_loss_slot} "
               f"acc: {intent_acc} slot f1: {slot_f1} sent acc: {sent_acc}")

    return intent_acc, slot_f1, sent_acc


def run_warmup(train_data_file, config):
    print("============================ WARMUP ============================")
    embedding_file = open(config.data_path + "emb_word.txt", "r", encoding="utf-8")
    embeddings = [emb.strip() for emb in embedding_file]
    embedding_word, vocab = process_emb(embeddings, emb_dim=config.emb_dim)

    idx2intent, intent2idx = load_label_dict(config.data_path + "intent_label.txt")
    idx2slot, slot2idx = load_label_dict(config.data_path + "slot_label.txt")
    n_slot_tag = len(idx2slot.items())
    n_intent_class = len(idx2intent.items())

    train_dir = os.path.join(config.data_path, train_data_file)
    train_dataset = MyDataset(train_dir, config.max_len, intent2idx, slot2idx, vocab)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size)
    model = Joint_model(config, config.hidden_dim, config.batch_size, config.max_len, n_intent_class, n_slot_tag,
                        embedding_word)

    model.cuda()
    model.train()
    optimizer = RAdam(model.parameters(), lr=config.lr, weight_decay=0.000001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [40, 70], gamma=config.lr_scheduler_gamma,
                                                     last_epoch=-1)

    for _ in trange(config.warmup_epoch, desc="Epoch"):
        for batch in tqdm(train_loader, desc="Batch"):
            model.zero_grad()
            inputs, slot_labels, intent_labels, masks, _, _ = batch
            inputs, masks, intent_labels, slot_labels = inputs.cuda(), masks.cuda(), intent_labels.cuda(), slot_labels.cuda()
            logits_intent, logits_slot = model.forward_logit(inputs, masks)
            loss_intent, loss_slot, = model.loss1(logits_intent, logits_slot, intent_labels, slot_labels, masks)
            loss = loss_slot + loss_intent
            loss.backward()
            optimizer.step()
        scheduler.step()

    torch.save(model, config.model_save_dir + config.model_path)


def run_train(train_data_file, dev_data_file, config):
    print("============================ TRAIN ============================")
    embedding_file = open(config.data_path + "emb_word.txt", "r", encoding="utf-8")
    embeddings = [emb.strip() for emb in embedding_file]
    embedding_word, vocab = process_emb(embeddings, emb_dim=config.emb_dim)

    idx2intent, intent2idx = load_label_dict(config.data_path + "intent_label.txt")
    idx2slot, slot2idx = load_label_dict(config.data_path + "slot_label.txt")

    train_dir = os.path.join(config.data_path, train_data_file)
    dev_dir = os.path.join(config.data_path, dev_data_file)
    train_dataset = MyDataset(train_dir, config.max_len, intent2idx, slot2idx, vocab)
    dev_dataset = MyDataset(dev_dir, config.max_len, intent2idx, slot2idx, vocab)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size)
    dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=config.batch_size)
    print("Load model from warmup")
    model = torch.load(config.model_save_dir + config.model_path, map_location=device)  # load from warmup

    model.cuda()
    model.train()
    optimizer = RAdam(model.parameters(), lr=config.lr, weight_decay=0.000001)
    best_slot_f1 = [0.0, 0.0, 0.0]
    best_intent_acc = [0.0, 0.0, 0.0]
    best_sent_acc = [0.0, 0.0, 0.0]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [40, 70], gamma=config.lr_scheduler_gamma,
                                                     last_epoch=-1)

    for epoch in trange(config.epoch, desc="Epoch"):
        # ===== gmm score =====
        # model.eval()
        # losses = np.zeros(len(train_loader.dataset))
        # with torch.no_grad():
        #     for batch in tqdm(train_loader, desc="GMM eval"):
        #         inputs, slot_labels, intent_labels, masks, users, index = batch
        #         inputs, masks, intent_labels, slot_labels = inputs.cuda(), masks.cuda(), intent_labels.cuda(), slot_labels.cuda()
        #         logits_intent, logits_slot = model.forward_logit(inputs, masks)
        #         loss_intent, loss_slot, = model.loss1(logits_intent, logits_slot, intent_labels, slot_labels, masks)
        #
        #         if epoch < 40:
        #             loss = loss_slot + loss_intent
        #         else:
        #             loss = 0.8 * loss_intent + 0.2 * loss_slot
        #         losses[index] = loss.cpu().detach().numpy()
        #         optimizer.step()

        # losses = (losses - losses.min()) / (losses.max() - losses.min())
        # losses = losses.reshape(-1, 1)  # 一个 epoch 内各个 sample 的 loss
        # gmm = GaussianMixture(
        #     n_components=2,
        #     max_iter=10,
        #     tol=1e-2,
        #     reg_covar=5e-4,
        # )
        # gmm.fit(losses)
        # prob = gmm.predict_proba(losses)  # 各个 sample 属于两个高斯分布的概率
        # prob = prob[:, gmm.means_.argmin()]  # 各个 sample 属于 mean 较小的那个高斯分布的概率

        # cluster_draw(
        #     f"mean = {config.user_mean}, std = {config.user_std}\n"
        #     f"epoch = {epoch}\n"
        #     f"batch_size = {config.batch_size}",
        #     losses,
        #     gmm.means_,
        #     gmm.covariances_,
        # )

        # ===== user score =====
        # TODO

        # ===== final score, filter noisy examples =====
        # pred = prob > 0.5  # 全是 bool，判定各个 sample 是否为无噪音样本
        # noisy_sample_indices = pred.nonzero()[0]

        # ===== update user score ====
        # TODO

        # ===== creates labels with sharpen =====
        # with torch.no_grad():
        #     for i in tqdm(noisy_sample_indices, desc="Fix noise"):
        #         inputs_u, _, _, masks_u, _, index = train_loader.dataset[i]
        #         inputs_u, masks_u = inputs_u.cuda(), masks_u.cuda()
        #         inputs_u, masks_u = torch.unsqueeze(inputs_u, dim=0), torch.unsqueeze(masks_u, dim=0)  # as a batch
        #         logits_intent, _ = model.forward_logit(inputs_u, masks_u)
        #         p = torch.softmax(logits_intent, dim=1)
        #         pt = p ** (1 / config.temp)
        #         targets_u = pt / pt.sum(dim=1, keepdim=True)
        #         targets_u = targets_u.detach()
        #         assert index.item() == i
        #         train_dataset.intent_lists[i] = torch.argmax(targets_u).cpu()  # replace noisy label

        # ===== train with mixup =====
        model.train()
        for batch in tqdm(train_loader, desc="Batch", leave=False):
            if len(batch[0]) != config.batch_size:  # last batch
                break

            model.zero_grad()
            inputs, slot_labels, intent_labels, masks, users, _ = batch
            inputs, masks, intent_labels, slot_labels = inputs.cuda(), masks.cuda(), intent_labels.cuda(), slot_labels.cuda()

            logits_intent, logits_slot = model.forward_logit(inputs, masks)

            # ===== mixup =====
            l = np.random.beta(config.b, config.b)
            l = max(l, 1 - l)
            index = torch.randperm(config.batch_size).cuda()
            logits_intent_pred = l * logits_intent + (1 - l) * logits_intent[index]
            logits_slot_pred = l * logits_slot + (1 - l) * logits_slot[index]
            loss1_intent, loss1_slot = model.loss1(logits_intent_pred, logits_slot_pred, intent_labels, slot_labels,
                                                   masks)
            loss2_intent, loss2_slot = model.loss1(logits_intent_pred, logits_slot_pred, intent_labels[index],
                                                   slot_labels[index], masks)
            loss1_intent, loss1_slot = l * loss1_intent, l * loss1_slot
            loss2_intent, loss2_slot = (1 - l) * loss2_intent, (1 - l) * loss2_slot
            if epoch < 40:
                loss = (loss1_slot + loss2_slot) + (loss1_intent + loss2_intent)
            else:
                loss = 0.8 * (loss1_intent + loss2_intent) + 0.2 * (loss1_slot + loss2_slot)
            # ===== origin =====
            # loss_intent, loss_slot = model.loss1(logits_intent, logits_slot, intent_labels, slot_labels, masks)
            # if epoch < 40:
            #     loss = loss_slot + loss_intent
            # else:
            #     loss = 0.8 * loss_intent + 0.2 * loss_slot
            loss.backward()
            optimizer.step()
            wandb.log({"loss": loss.item()})

        intent_acc, slot_f1, sent_acc = dev(model, dev_loader, idx2slot, config)
        wandb.log({"dev intent_acc": intent_acc, "dev slot_f1": slot_f1, "dev sent_acc": sent_acc})

        if slot_f1 > best_slot_f1[1]:
            best_slot_f1 = [sent_acc, slot_f1, intent_acc, epoch]
            torch.save(model, config.model_save_dir + config.model_path)
        if intent_acc > best_intent_acc[2]:
            torch.save(model, config.model_save_dir + config.model_path)
            best_intent_acc = [sent_acc, slot_f1, intent_acc, epoch]
        if sent_acc > best_sent_acc[0]:
            torch.save(model, config.model_save_dir + config.model_path)
            best_sent_acc = [sent_acc, slot_f1, intent_acc, epoch]
        scheduler.step()
    print("best_slot_f1:", best_slot_f1)
    print("best_intent_acc:", best_intent_acc)
    print("best_sent_acc:", best_sent_acc)


def run_test(test_data_file, config):
    print("============================ TEST ============================")
    # load dict
    idx2intent, intent2idx = load_label_dict(config.data_path + "intent_label.txt")
    idx2slot, slot2idx = load_label_dict(config.data_path + "slot_label.txt")

    embedding_file = open(config.data_path + "emb_word.txt", "r", encoding="utf-8")
    embeddings = [emb.strip() for emb in embedding_file]
    embedding_word, vocab = process_emb(embeddings, emb_dim=config.emb_dim)

    test_dir = os.path.join(config.data_path, test_data_file)
    test_dataset = MyDataset(test_dir, config.max_len, intent2idx, slot2idx, vocab)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=config.batch_size)
    model = torch.load(config.model_save_dir + config.model_path, map_location=device)
    model.eval()
    pred_intents = []
    true_intents = []
    pred_slots = []
    true_slots = []

    for batch in tqdm(test_loader, desc="Evaluating"):
        inputs, slot_labels, intent_labels, masks, _, _ = batch
        inputs, masks, intent_labels, slot_labels = inputs.cuda(), masks.cuda(), intent_labels.cuda(), slot_labels.cuda()
        logits_intent, logits_slot = model.forward_logit(inputs, masks)
        pred_intent, pred_slot = model.pred_intent_slot(logits_intent, logits_slot, masks)
        pred_intents.extend(pred_intent.cpu().numpy().tolist())
        true_intents.extend(intent_labels.cpu().numpy().tolist())

        slot_labels = slot_labels.cpu().numpy().tolist()
        for i in range(len(pred_slot)):
            pred = []
            true = []
            for j in range(len(pred_slot[i])):
                pred.append(idx2slot[pred_slot[i][j].item()])
                true.append(idx2slot[slot_labels[i][j]])
            pred_slots.append(pred[1:-1])
            true_slots.append(true[1:-1])
    slot_metrics = SlotMetrics(true_slots, pred_slots)
    slot_f1, _, _ = slot_metrics.get_slot_metrics()

    Metrics_intent = IntentMetrics(pred_intents, true_intents)
    print(Metrics_intent.classification_report)
    intent_acc = Metrics_intent.accuracy
    sent_acc = semantic_acc(pred_slots, true_slots, pred_intents, true_intents)
    tqdm.write(f"Evaluation - acc: {intent_acc} slot f1: {slot_f1} sent_acc: {sent_acc}")
    wandb.log({"test intent_acc": intent_acc, "test slot_f1": slot_f1, "test sent_acc": sent_acc})

    return sent_acc


if __name__ == "__main__":
    wandb.init(project="robust-slu-selfmix", config="config.yaml")
    wandb.run.name += "__mixup"

    assert type(wandb.config.user_mean) == list and type(wandb.config.user_std) == list
    assert len(wandb.config.user_mean) == len(wandb.config.user_std), "Length of `user_mean` != `user_std`"
    user_count = len(wandb.config.user_mean)
    user_noise_p = [random.gauss(mean, std) for mean, std in zip(wandb.config.user_mean, wandb.config.user_std)]
    if "atis" in wandb.config.data_path:
        n_class = 26  # ATIS
    else:
        raise ValueError
    wandb.config.update({
        "user_count": user_count,
        "user_noise_p": user_noise_p,
        "n_class": n_class,
    })
    config = wandb.config

    train_file = "train-noisy.txt"
    dev_file = "dev.txt"
    test_file = "test.txt"

    set_seed(config.seed)
    make_noise(config)

    # selfmix warmup
    run_warmup(train_file, config)
    # train model
    run_train(train_file, dev_file, config)
    # test model
    run_test(test_file, config)
