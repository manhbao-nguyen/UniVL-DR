import json
from visual import TSVFile
import logging
import sys
import base64
import os
from typing import Optional, List
import numpy as np
from torch import nn
from torch.nn import LayerNorm
from tqdm import tqdm
import torch
import argparse
import os.path as op
import time
import pickle
import math
import clip
from torch import optim
from custom_loss import loss_on_every_positive_source, loss_on_all_positive_sources, loss_multihop
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from custom_loss import loss_on_every_positive_source, loss_on_all_positive_sources, loss_multihop
from data_2 import WebQADataset_2, load_caps, load_docs, load_file
from contextlib import suppress
logger = logging.getLogger()
import random
import torch.nn.functional as F
from train import set_seed, convert_models_to_fp32, cosine_lr


def plot_tr_and_val_metrics(
    tr_steps:List[int],
    tr_loss_trajectory: List[float],
    tr_acc_trajectory: List[float],
    val_steps:List[int],
    val_loss_trajectory: List[float],
    val_acc_trajectory: List[float],
    epoch = None
) -> None:
    """
    Args:
        tr_acc_trajectory: list of mean accuracies across epochs
        tr_loss_trajectory: list of training batch loss across epochs
        val_loss_trajectory: list of validation mean accuracies across epochs
        val_acc_trajectory:  list of validation batch loss across epochs
    """
    #epochs = list(range(1, len(tr_loss_trajectory) + 1))

    fig, ax = plt.subplots()
    ax.plot(tr_steps, tr_loss_trajectory, label="tr_loss", color="blue")
    ax.plot(val_steps, val_loss_trajectory, label="val_loss", color="red")
    ax.set_xlabel("grad steps")
    ax.set_ylabel("loss")
    ax.set_title(f"Evolution of the train and val losses at grad step {epoch}")
    ax.legend()
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(tr_steps, tr_acc_trajectory, label="tr_acc", color="blue")
    ax.plot(val_steps, val_acc_trajectory, label="val_acc", color="red")
    ax.set_xlabel("grad steps")
    ax.set_ylabel("acc")
    ax.set_title(f"Evolution of the train and val accuracies at grad step {epoch}")
    ax.legend()
    plt.show()


def eval_loss(model, valid_reader):
    model.eval()
    total_loss = 0.0
    total_corr = 0.0
    counter = 0.0
    for step, batch in tqdm(enumerate(valid_reader)):
        with torch.no_grad():
            convert_models_to_fp32(model)
            query_embedding = model.encode_text(batch['queries'])#.cuda()
            # bsz, emb_dim = query_embedding.size()
            # query_embedding = query_embedding.view(bsz, 1, emb_dim)
            candidate_embeddings = []
            if 'img_inputs' in batch:
                img_embeddings = model.encode_image(batch['img_inputs'])#.cuda()
                if 'img_caps' in batch:
                    cap_embeddings = model.encode_text(batch['img_caps'])#.cuda(
                    img_embeddings = img_embeddings + cap_embeddings
                candidate_embeddings.append(img_embeddings)
            if 'txt_inputs' in batch:
                txt_embeddings = model.encode_text(batch['txt_inputs'])#.cuda()
                candidate_embeddings.append(txt_embeddings)
            candidate_embeddings = torch.cat(candidate_embeddings, dim=0)
            query_embedding = F.normalize(query_embedding, dim=-1)
            candidate_embeddings = F.normalize(candidate_embeddings, dim=-1)

            # pos_embeddings = torch.index_select(candidate_embeddings, 0, torch.LongTensor(batch['pos_idx']))#.cuda()
            # neg_embeddings = torch.index_select(candidate_embeddings, 0, torch.LongTensor(batch['neg_idx']))#.cuda()
            # pos_embeddings = pos_embeddings.view(bsz, 1, emb_dim)
            # neg_embeddings = neg_embeddings.view(bsz, -1, emb_dim)
            # logit_scale = model.logit_scale.exp()
            # pos_scores = (query_embedding * pos_embeddings).sum(-1) * logit_scale
            # neg_scores = (query_embedding * neg_embeddings).sum(-1) * logit_scale
            # logit_matrix = torch.cat([pos_scores, neg_scores], 1)
            #
            # lsm = F.log_softmax(logit_matrix, dim=1)
            #loss = torch.mean(-1.0 * lsm[:, 0])


            logit_matrix, loss = loss_on_every_positive_source(model, query_embedding, candidate_embeddings, batch['pos_idx'], batch['neg_idx'])

            max_score, max_idxs = torch.max(logit_matrix, 1)
            correct_predictions_count = (max_idxs == 0).sum() / logit_matrix.size(0)
            total_corr += correct_predictions_count.item()
            total_loss += loss.item()
            counter += 1
    if counter == 0:
        return 0.0, 0.0
    return total_loss / counter, total_corr / counter


def train(train_reader, valid_reader, model, output_dir = None, model_name = None, individual = False, together = False, multihop = False, ponderate = None):

    t_total = len(train_reader) // args.gradient_accumulation_steps * args.num_train_epochs
    exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n, p: not exclude(n, p)

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

    optimizer = optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.},
            {"params": rest_params, "weight_decay": 0.2},
        ],
        lr=args.learning_rate,
        betas=(0.9,  0.98),
        eps=1.0e-6,
    )
    scheduler = cosine_lr(optimizer, args.learning_rate, args.warmup_steps, t_total)
    tag, global_step, global_loss, best_acc = 0, 0, 0.0, 0.0
    model.zero_grad()


    loss_tr = []
    acc_tr = []
    steps_tr = []
    loss_te = []
    acc_te = []
    steps_te = []

    for epoch in range(int(args.num_train_epochs)):
        for step, batch in enumerate(train_reader):
            model.train()
            convert_models_to_fp32(model)
            query_embedding = model.encode_text(batch['queries'])#.cuda()
            # bsz, emb_dim = query_embedding.size()
            # query_embedding = query_embedding.view(bsz, 1, emb_dim)
            candidate_embeddings = []
            if 'img_inputs' in batch:
                img_embeddings = model.encode_image(batch['img_inputs'])#.cuda()
                if 'img_caps' in batch:
                    cap_embeddings = model.encode_text(batch['img_caps'])#
                    img_embeddings = img_embeddings + cap_embeddings
                candidate_embeddings.append(img_embeddings)
            if 'txt_inputs' in batch:
                txt_embeddings = model.encode_text(batch['txt_inputs'])#.cuda()
                candidate_embeddings.append(txt_embeddings)
            candidate_embeddings = torch.cat(candidate_embeddings, dim=0)
            query_embedding = F.normalize(query_embedding, dim=-1)
            candidate_embeddings = F.normalize(candidate_embeddings, dim=-1)
            # pos_embeddings = torch.index_select(candidate_embeddings, 0, torch.LongTensor(batch['pos_idx']))#.cuda()
            # neg_embeddings = torch.index_select(candidate_embeddings, 0, torch.LongTensor(batch['neg_idx']))#.cuda()
            # pos_embeddings = pos_embeddings.view(bsz, 1, emb_dim)
            # neg_embeddings = neg_embeddings.view(bsz, -1, emb_dim)
            # logit_scale = model.logit_scale.exp()
            # pos_scores = (query_embedding * pos_embeddings).sum(-1) * logit_scale
            # neg_scores = (query_embedding * neg_embeddings).sum(-1) * logit_scale
            # logit_matrix = torch.cat([pos_scores, neg_scores], 1)
            #
            # lsm = F.log_softmax(logit_matrix, dim=1)
            # loss = torch.mean(-1.0 * lsm[:, 0])
            # logit_matrix, loss = loss_multihop(model, query_embedding, candidate_embeddings,
            #                                                      batch['pos_idx'], batch['neg_idx'])


            if individual:
                logit_matrix, loss = loss_on_every_positive_source(model, query_embedding, candidate_embeddings,
                                                   batch['pos_idx'], batch['neg_idx'])
            elif together:
                logit_matrix, loss = loss_on_all_positive_sources(model, query_embedding, candidate_embeddings,
                                                   batch['pos_idx'], batch['neg_idx'])
            elif multihop:
                logit_matrix, loss = loss_multihop(model, query_embedding, candidate_embeddings,
                                                   batch['pos_idx'], batch['neg_idx'])
            elif  ponderate is not None and len(ponderate) == 3:
                logit_matrix_individual, loss_individual = loss_on_every_positive_source(model, query_embedding, candidate_embeddings,
                                                                   batch['pos_idx'], batch['neg_idx'])
                logit_matrix_together, loss_together = loss_on_all_positive_sources(model, query_embedding, candidate_embeddings,
                                                   batch['pos_idx'], batch['neg_idx'])
                logit_matrix_multi, loss_multi = loss_multihop(model, query_embedding, candidate_embeddings,
                                                   batch['pos_idx'], batch['neg_idx'])
                logit_matrix = torch.cat([logit_matrix_individual, logit_matrix_together, logit_matrix_multi], dim=0)
                loss = ponderate[0] * loss_individual + ponderate[1] * loss_together + ponderate[2] * loss_multi

            else:
                raise AssertionError

            max_score, max_idxs = torch.max(logit_matrix, 1)
            correct_predictions_count = (max_idxs == 0).sum() / logit_matrix.size(0)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            global_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
                scheduler(global_step)
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)
                model.zero_grad()
                logger.info("Epoch: {}, global_step: {}, lr: {:.6f}, acc: {:.4f} ({:.4f}), ".format(
                    epoch, global_step,
                    optimizer.param_groups[0]["lr"], correct_predictions_count,
                    global_loss / global_step,
                ))
                acc_tr.append(correct_predictions_count)
                loss_tr.append(global_loss / global_step)
                steps_tr.append(global_step)
                # print('*************', global_loss, '****************')
                if global_step % args.eval_steps == 0 and global_step > 0:
                    logger.info('*********Start eval loss**********')
                    dev_loss, dev_acc = eval_loss(model, valid_reader)
                    acc_te.append(dev_acc)
                    loss_te.append(dev_loss)
                    steps_te.append(global_step)
                    logger.info("Evaluation at global step {}, average dev loss: {:.4f}, average dev acc: {:.4f}".format(
                        global_step, dev_loss, dev_acc))
                    if best_acc <= dev_acc:
                        best_acc = dev_acc
                        tag = 0
                    else:
                        tag += 1
                    torch.save({'epoch': epoch,
                                    'model': model.state_dict()}, os.path.join(output_dir or args.out_path, model_name or f"model_step_{global_step}.pt"))
                    logger.info("Saved epoch {0}, best acc {1}".format(epoch, dev_acc))
                    if tag >= args.early_stop:
                        logger.info('*********early stop**********')
                        plot_tr_and_val_metrics(steps_tr, loss_tr, acc_tr, steps_te, loss_te, acc_te)
                        return

        plot_tr_and_val_metrics(steps_tr, loss_tr, acc_tr, steps_te, loss_te, acc_te, epoch)




if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser("")

    parser.add_argument("--out_path", type=str, default='./retrained_model/')
    parser.add_argument("--train_path", type=str, default=  '../CLIP-DPR/checkpoint_multi_inb/train_all.json')
    parser.add_argument("--valid_path", type=str, default = '../CLIP-DPR/checkpoint_multi_inb/dev_all.json')
    parser.add_argument("--pretrained_model_path", type=str, default = '../CLIP-DPR/checkpoint_multi_inb/model.best.pt')
    parser.add_argument("--doc_path", type=str, default = '../data/all_docs.json')
    parser.add_argument("--cap_path", type=str, default = '../data/all_imgs_exp.json')
    parser.add_argument("--img_feat_path", type=str, default = "../../WebQA/imgs.tsv") #TODO: change this location
    parser.add_argument("--img_linelist_path", type=str, default = '../data/imgs.lineidx.new')

    parser.add_argument('--only_txt', action='store_true', default=False)
    parser.add_argument('--only_img', action='store_true', default=False)


    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--early_stop", type=int, default=5)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--valid_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--eval_steps", type=int, default=2)
    parser.add_argument("--img_neg_num", type=int, default=4)
    parser.add_argument("--txt_neg_num", type=int, default=4)
    parser.add_argument("--neg_num", type=int, default=0)

    args = parser.parse_args()
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    handlers = [logging.FileHandler(os.path.join(args.out_path, 'train_log.txt')), logging.StreamHandler()]
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG,
                        datefmt='%d-%m-%Y %H:%M:%S', handlers=handlers)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logger.info(args)
    set_seed(args)

    if args.only_txt:
        train_data = load_file(args.train_path, img=False)
        valid_data = load_file(args.valid_path, img=False)
    elif args.only_img:
        train_data = load_file(args.train_path, txt=False)
        valid_data = load_file(args.valid_path, txt=False)
    else:
        train_data = load_file(args.train_path)
        valid_data = load_file(args.valid_path)
    docs = load_docs(args.doc_path)
    captions = None
    if args.cap_path:
        captions = load_caps(args.cap_path)
    args_dev = args
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)  # Must set jit=False for training
    clip.model.convert_weights(model)



    # COMMON DATASETS

    train_data = WebQADataset_2(args, preprocess, train_data, docs, captions=captions, shuffle=True, max_queries=5000)
    train_sampler = RandomSampler(train_data)
    traindata_reader = DataLoader(dataset=train_data, sampler=train_sampler, num_workers=args.num_workers,
                                  batch_size=args.train_batch_size, collate_fn=train_data.Collector, drop_last=True)
    args_dev.neg_num = 8
    valid_data = WebQADataset_2(args_dev, preprocess, valid_data, docs, captions=captions, shuffle=False, max_queries=300)
    valid_sampler = SequentialSampler(valid_data)
    validdata_reader = DataLoader(dataset=valid_data, sampler=valid_sampler, num_workers=args.num_workers,
                                batch_size=args.valid_batch_size, collate_fn=valid_data.Collector, drop_last=False)

    path_last_checkpoint = "./checkpoint_multi_hn/model.best.pt"

    # # EXPERIMENT 1: training on each positive source individually
    # if args.pretrained_model_path != None:
    #     logger.info('loading checkpoint from {}'.format(args.pretrained_model_path))
    #     model.load_state_dict(torch.load(args.pretrained_model_path, map_location=torch.device('cpu'))['model'])

    # logger.info('loading checkpoint from {}'.format(path_last_checkpoint))
    # model.load_state_dict(torch.load(path_last_checkpoint, map_location=torch.device('cpu'))['model'])
    #
    # train(traindata_reader, validdata_reader, model, output_dir="./exp_1/", model_name="model_best.pt", individual=True)

    path_checkpoint_exp_1 = "./exp_1/best_model.pt"
    # EXPERIMENT 2: training on all positive sources as a batch together (from experiment 1 checkpoint)

    # logger.info('loading checkpoint from {}'.format(path_last_checkpoint))
    # model.load_state_dict(torch.load(path_last_checkpoint, map_location=torch.device('cpu'))['model'])
    # train(traindata_reader, validdata_reader, model, output_dir="./exp_2/", model_name="model_best.pt", together=True)

    # EXPERIMENT 3: multihop training (from experiment 1 checkpoint)
    logger.info('loading checkpoint from {}'.format(path_last_checkpoint))
    model.load_state_dict(torch.load(path_last_checkpoint, map_location=torch.device('cpu'))['model'])
    train(traindata_reader, validdata_reader, model, output_dir="./exp_3/", model_name=None, multihop=True)


    # EXPERIMENT 4: ponderated loss (from initial checkpoint)
    # if args.pretrained_model_path != None:
    #     logger.info('loading checkpoint from {}'.format(path_last_checkpoint))
    #     model.load_state_dict(torch.load(path_last_checkpoint, map_location=torch.device('cpu'))['model'])
    # logger.info('loading checkpoint from {}'.format(path_last_checkpoint))
    # model.load_state_dict(torch.load(path_last_checkpoint, map_location=torch.device('cpu'))['model'])
    # train(traindata_reader, validdata_reader, model, output_dir="./exp_4/", model_name="model_best.pt", ponderate=[0.7, 0.1, 0.2])





