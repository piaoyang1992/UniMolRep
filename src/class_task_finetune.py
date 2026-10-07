#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/7/21 14:05
# @Author : piaoyang cao
# @File : run_class.py


import argparse

import torch
from torch import nn

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.optim.lr_scheduler import StepLR

import numpy as np

import torch.optim as optim

from tqdm import tqdm

from modules.group_vae_model import PSVAEModel
#from data.bpe_dataset import *
from data.group_graph_dataset import MolDataset1
#from data.mol_bpe import Tokenizer
from data.mol_group_graph import Tokenizer, get_vocab
from data.logger import print_log
from utils.nn_utils import VAEEarlyStopping
#from utils.group_nn_utils import common_config, predictor_config, encoder_config
from utils.nn_utils import common_config, predictor_config, encoder_config
from utils.nn_utils import ps_vae_config
from torch_geometric.loader import DataLoader
import time

import os
import pandas as pd

from modules.predictor import Predictor

from sklearn.model_selection import KFold
from sklearn import metrics

from train_group import load_model
import random
from utils.nn_utils import  encoder_config
from modules.encoder  import GroupEncoder


def train(vocab_datas, model,  device, loader, optimizer, loss_criterion):

    model.train()
    # train_meter = Meter()
    loss_accum = 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        atom_batch, group_batch = batch
        atom_batch,group_batch = atom_batch.to(device), group_batch.to(device)
        #batch = atom_batch, group_batch

        label = atom_batch.prop
        #optimized_latents = model.get_z_from_batch((atom_batch, group_batch))
        #print(label)
        pred = model(vocab_datas,group_batch)
        #batch = batch.to(device)


        # pred, _ = model(batch)
        # print(pred.size)
        # print(batch.y)
        loss = (loss_criterion(pred, label.view(pred.shape).to(device)).float()).mean()
        # print(loss)
        loss_accum += loss.detach().cpu().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # train_meter.update(pred, batch.y.to(args.device).float(), None)
        torch.cuda.empty_cache()

        # print(train_meter.compute_metric('return_pred_true'))
        # train_score = np.mean(train_meter.compute_metric('roc_auc'))
    return loss_accum / (step + 1)


def eval(vocab_datas, model, device, loader):
    model.eval()
    # eval_meter = Meter()
    y_pred_list = []
    y_true_list = []

    with torch.no_grad():
        for step, batch in enumerate(tqdm(loader, desc="Iteration")):

            atom_batch, group_batch = batch
            atom_batch,group_batch = atom_batch.to(device), group_batch.to(device)
            batch = atom_batch, group_batch

            label = atom_batch.prop
            # optimized_latents = model.get_z_from_batch((atom_batch, group_batch))

            pred = model(vocab_datas,group_batch)
            #optimized_latents = model.get_z_from_batch((atom_batch, group_batch))


            y_pred_list.append(pred.detach().cpu())
            y_true_list.append(label.view(pred.shape).detach().cpu())
            # eval_meter.update(pred, batch.y.view(pred.shape).to(device), mask=None)
            torch.cuda.empty_cache()

    # y_pred, y_true = eval_meter.compute_metric('return_pred_true')
    y_pred = torch.cat(y_pred_list, dim=0)
    y_true = torch.cat(y_true_list, dim=0)
    # print(y_pred)
    # print(y_pred.size())
    y_true = torch.nan_to_num(y_true.squeeze())
    y_pred = torch.nan_to_num(y_pred.squeeze())

    y_pred_list = torch.sigmoid(y_pred)
    y_pred_label = torch.where(y_pred_list >= 0.5, 1, 0)

    auc = metrics.roc_auc_score(y_true.numpy(), y_pred.numpy())
    prc_auc = metrics.average_precision_score(y_true.numpy(), y_pred.numpy())
    accuracy = metrics.accuracy_score(y_true.numpy(), y_pred_label.numpy())
    # se, sp = sesp_score(y_true.numpy(),y_pred_label.numpy())
    pre, rec, f1, sup = metrics.precision_recall_fscore_support(y_true.numpy(), y_pred_label.numpy())
    # mcc = metrics.matthews_corrcoef(y_true.numpy(), y_pred_label.numpy())
    f1 = f1[1]
    result = {'roc_auc': auc, 'accuracy': accuracy, 'prc_auc': prc_auc, 'f1': f1}
    return result


def parse():
    """parse command"""
    parser = argparse.ArgumentParser(description='training overall model for molecule generation')
    parser.add_argument('--times', type=int, default=5,
                        help='repeat times')
    parser.add_argument('--task_name_list', type=list,
                        default=['hiv'],
                        # default=['bace','bbbp','hiv','clintox',
                        help='task_name_list')

    parser.add_argument('--shuffle', action='store_true', help='shuffle data')
    parser.add_argument('--save_dir', type=str, default='/home/pycao/PS-VAE-main/model/np_gg.txt',
                        help='path to store the model')
    parser.add_argument('--batch_size', type=int, default=32, help='size of mini-batch')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='balancing reconstruct loss and predictor loss')
    # vae training
    parser.add_argument('--beta', type=float, default=0,
                        help='balancing kl loss and other loss')
    parser.add_argument('--step_beta', type=float, default=0.002,
                        help='value of beta increasing step')
    parser.add_argument('--max_beta', type=float, default=0.02)
    parser.add_argument('--kl_warmup', type=int, default=0,
                        help='Within these steps beta is set to 0')
    parser.add_argument('--kl_anneal_iter', type=int, default=1000)

    parser.add_argument('--num_workers', type=int, default=1, help='number of cpus to load data')
    parser.add_argument('--gpus', default=-1, help='gpus to use')
    parser.add_argument('--epochs', type=int, default=100, help='max epochs')
    parser.add_argument('--patience', type=int, default=3, help='early stopping number of epochs')
    parser.add_argument('--grad_clip', type=float, default=10,
                        help='clip large gradient to prevent gradient boom')
    parser.add_argument('--monitor', type=str, default='val_loss',
                        help='Value to monitor in early stopping')

    parser.add_argument('--graph_embedding_dim', type=int, default=100,
                        help='dim of graph embedding by encoder and also condition for ae decoders')
    parser.add_argument('--latent_dim', type=int, default=56,
                        help='dim of latent z for vae decoders')

    # ps-vae decoder only
    parser.add_argument('--num_pos_type', type=int, default=100,
                        help='Max number of pieces')
    parser.add_argument('--t', type=int, default=4,
                        help='Max number of gnn layer')
    parser.add_argument('--atom_embedding_dim', type=int, default=50,
                        help='Embedding dim for a single atom')
    parser.add_argument('--piece_embedding_dim', type=int, default=100,
                        help='Embedding dim for piece')
    parser.add_argument('--pos_embedding_dim', type=int, default=50,
                        help='Position embedding of piece')
    parser.add_argument('--predictor_hidden_dim', type=int, default=200,
                        help='hidden dim of predictor (MLP)')
    parser.add_argument('--node_hidden_dim', type=int, default=200,
                        help='dim of node hidden embedding in encoder and decoder')
    parser.add_argument('--piece_in_dim', type=int, default=210,
                        help='Hidden dim for rnn used in piece generation')
    parser.add_argument('--piece_link_dim', type=int, default=270,
                        help='Hidden dim for rnn used in piece generation')
    parser.add_argument('--graph_pooling', type=str, default='sum',
                        help='Hidden dim for rnn used in piece generation')

    return parser.parse_args()


def main():
    # Training settings
    args = parse()
    print_log(args)
    print_log('loading data ...')


    if args.gpus == -1:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.gpus}')
    args.device = device


    root = '/home/pycao/PS-VAE-main-1/data/finetune/class_task'
    check_path = '/home/pycao/PS-VAE-main-1/data/check/'
    result_path = '/home/pycao/PS-VAE-main-1/data/result'
    model_path = '/home/pycao/PS-VAE-main-1/data/zinc250k/train.txt'
    '''
    root = '/work/home/zhy_hzau/cpy/PS-VAE-main-1/data/finetune/class_task'
    check_path = '/work/home/zhy_hzau/cpy/PS-VAE-main-1/data/check/'
    result_path = '/work/home/zhy_hzau/cpy/PS-VAE-main-1/data/result'
    model_path = '/work/home/zhy_hzau/cpy//PS-VAE-main-1/data/zinc250k/train.txt'
    '''
    # result_path = '../result/class_task'
    test_pd = pd.DataFrame(columns=args.task_name_list)
    time_task = {}

    #print("loading pretrain model")

    #model = load_model(model_path, device)



    for task in args.task_name_list:

        time_start = time.time()

        path = os.path.join(root, task)
        df = pd.read_csv(os.path.join(path, task + '.csv'),header=0)
        label = list(df.columns).index('smiles') + 1
        tokenizer = Tokenizer(os.path.join(path, task + '.csv'),label=label)
        #vocab = tokenizer.chem_vocab


        dataset = MolDataset1(os.path.join(path, task + '.csv'), tokenizer)
        vocab_datas = tokenizer.get_vocab_data()


        raw_smiles_datas = []

        if task == "tox21":
            num_tasks = 12
        elif task == "pcba":
            num_tasks = 128
        elif task == "muv":
            num_tasks = 17
        elif task == "toxcast":
            num_tasks = 617
        elif task == "sider":
            num_tasks = 27
        elif task == "clintox":
            num_tasks = 2
        else:
            num_tasks = 1

        pretrain_path = '/home/pycao/PS-VAE-main-1/data/zinc250k/encoder_checkpoint.pt'

        pretrain_state_dict = torch.load(pretrain_path, map_location='cpu')


        result_pd = pd.DataFrame()
        # result_pd['index'] = ['roc_auc']
        result_pd['index'] = ['roc_auc', 'accuracy', 'prc_auc', 'f1']
        test_result = []
        # loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        split = []
        kf = KFold(n_splits=args.times, shuffle=True, random_state=42)
        # split = torch.load(path + '/split.pt')
        for k, (train_index, test_index) in enumerate(kf.split(dataset)):
            # for k, (train_index, test_index) in enumerate(split):
            split.append((train_index, test_index))
            train_dataset = [dataset[i] for i in train_index]
            test_dataset = [dataset[i] for i in test_index]

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            # valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

            encoder = GroupEncoder(**encoder_config(args))  # tokenizer 不影响编码器
            encoder.load_state_dict(pretrain_state_dict)
            #encoder.eval()
            model = Predictor(args.node_hidden_dim, num_tasks, encoder=encoder)
            best_valid_loss = 0

            loss_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

            params = model.parameters()
            optimizer = optim.Adam(params, lr=args.lr)
            scheduler = StepLR(optimizer, step_size=10, gamma=0.8)

            for name, param in model.named_parameters():
                if "encoder" in name:
                    param.requires_grad = False


            for epoch in range(1, args.epochs + 1):
                model.to(device)
                l = train(vocab_datas, model, device, train_loader, optimizer, loss_criterion)
                # print(l)
                scheduler.step()

                valid_loss = eval(vocab_datas, model, device, test_loader)['roc_auc']
                # test_loss = eval(vocab_datas, raw_smiles_datas, model, device, test_loader)['auc']

                if valid_loss > best_valid_loss:
                    best_valid_loss = valid_loss

                    checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                                  'optimizer_state_dict': optimizer.state_dict(),
                                  'scheduler_state_dict': scheduler.state_dict(), 'best_val_mae': best_valid_loss,
                                  }
                    torch.save(checkpoint, check_path + task + '_checkpoint.pt')

                print(
                    '***************************************************************************************************')
                print('{}, {}/{} time {}  valid_loss{}'.format(task, k + 1, args.times, epoch,
                                                               valid_loss))
                print(
                    '***************************************************************************************************')

            # 回归任务计算R2打分

            state_dict = torch.load(check_path + task + '_checkpoint.pt')
            model.load_state_dict(state_dict['model_state_dict'])
            stop_train_list = eval(vocab_datas, model,  device, train_loader)
            # stop_val_list = eval(vocab_datas, raw_smiles_datas, model, device, valid_loader)
            stop_test_list = eval(vocab_datas, model, device, test_loader)
            test_result.append(stop_test_list['roc_auc'])

            result_pd['train_' + str(k + 1)] = list(stop_train_list.values())
            # result_pd['val_' + str(k + 1)] = list(stop_val_list.values())
            result_pd['test_' + str(k + 1)] = list(stop_test_list.values())

            print(result_pd)

        time_end = time.time()
        time_task[task] = (time_end - time_start) / args.times

        test_pd[task] = test_result
        result_pd.to_csv(result_path + '/' + task + '_result.csv', index=False)
        torch.save(split,path+'/split.pt')
    # test_pd = test_pd.append(time_task,ignore_index=True)
    test_pd.loc[len(test_pd)] = list(time_task.values())
    print(test_pd)
    test_pd.to_csv(result_path + '/' + 'class_result.csv', index=False)




if __name__ == "__main__":
    main()

