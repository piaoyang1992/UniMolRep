#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import argparse
import numpy as np
import torch
#import pytorch_lightning as pl
#from pytorch_lightning.callbacks import ModelCheckpoint

from modules.group_vae_model import PSVAEModel
#from data.bpe_dataset import *
from data.group_graph_dataset import MolDataset1
#from data.mol_bpe import Tokenizer
from data.mol_group_graph import Tokenizer, get_vocab
from src.data.logger import print_log
from utils.nn_utils import VAEEarlyStopping
#from utils.group_nn_utils import common_config, predictor_config, encoder_config
from utils.nn_utils import common_config, predictor_config, encoder_config,decoder_config
from utils.nn_utils import ps_vae_config
from torch_geometric.loader import DataLoader

from modules.encoder  import GroupEncoder
from modules.vae_acylic_decoder import VAEPieceDecoder
from modules.predictor import Predictor
import pandas as pd

def setup_seed(seed):
     torch.manual_seed(seed)
     #torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     #random.seed(42)
     #torch.backends.cudnn.deterministic = True
     #pl.utilities.seed.seed_everything(seed=seed)




def train(model, vocab_data, train_loader, valid_loader, test_loader, args):
    vocab_data = vocab_data.to(args.device)

    model.to(args.device)
    [opt], [sche] = model.configure_optimizers()
    loss_accum = 0
    vocab_data.to(args.device)
    for epoch in range(args.epochs):
        for b in train_loader:
            atom_batch, group_batch = b
            b = atom_batch.to(args.device),group_batch.to(args.device)
            loss = model.training_step(vocab_data, b)
            #print(loss)
            #loss_accum += loss.detach().cpu().item()
            opt.zero_grad()
            loss.backward()
            opt.step()


    torch.save(model.state_dict(), args.path + '/checkpoint.pt')
    torch.save(model.encoder.state_dict(), args.path + '/encoder_checkpoint.pt')

    #trainer.fit(model, train_loader, valid_loader)
    # test
    #trainer.test(model, test_dataloaders=test_loader)


def get_smiles_set(path):
    train_path,valid_path,test_path = os.path.join(path,'train.txt'),\
                                                os.path.join(path,'valid.txt'),os.path.join(path,'test.txt')

    with open(train_path, 'r') as fin:
        lines = fin.readlines()
        train_smiles = [s.strip('\n') for s in lines]
    with open(valid_path, 'r') as fin:
        lines = fin.readlines()
        valid_smiles = [s.strip('\n') for s in lines]
    with open(test_path, 'r') as fin:
        lines = fin.readlines()
        test_smiles = [s.strip('\n') for s in lines]

    all_smiles = train_smiles + valid_smiles + test_smiles

    train_set = [all_smiles.index(s) for s in train_smiles ]
    valid_set = [all_smiles.index(s) for s in valid_smiles]
    test_set = [all_smiles.index(s) for s in test_smiles]
    return all_smiles, train_set, valid_set, test_set

def parse():
    """parse command"""
    parser = argparse.ArgumentParser(description='training overall model for molecule generation')
    parser.add_argument('--shuffle', action='store_true', help='shuffle data')
    parser.add_argument('--save_dir', type=str, default='/home/pycao/PS-VAE-main/model/np_gg.txt', help='path to store the model')
    parser.add_argument('--batch_size', type=int, default=128, help='size of mini-batch')
    parser.add_argument('--lr', type=float, default= 1e-3, help='Learning rate')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='balancing reconstruct loss and predictor loss')
    # vae training
    parser.add_argument('--beta', type=float, default=0,
                        help='balancing kl loss and other loss')
    parser.add_argument('--step_beta', type=float, default=0.002,
                        help='value of beta increasing step')
    parser.add_argument('--max_beta', type=float, default=0.01)
    parser.add_argument('--kl_warmup', type=int, default=0,
                        help='Within these steps beta is set to 0')
    parser.add_argument('--kl_anneal_iter', type=int, default=1000)

    parser.add_argument('--num_workers', type=int, default=1, help='number of cpus to load data')
    parser.add_argument('--gpus', default=0, help='gpus to use')
    parser.add_argument('--epochs', type=int, default=10, help='max epochs')
    parser.add_argument('--patience', type=int, default=3, help='early stopping number of epochs')
    parser.add_argument('--grad_clip', type=float, default=10,
                        help='clip large gradient to prevent gradient boom')
    parser.add_argument('--monitor', type=str, default='val_loss',
                        help='Value to monitor in early stopping')

    # model parameters
    parser.add_argument('--props', default=['qed', 'sa', 'logp'],choices=['qed', 'sa', 'logp'],
                        help='properties to predict')
    parser.add_argument('--predictor_hidden_dim', type=int, default=200,
                        help='hidden dim of predictor (MLP)')
    parser.add_argument('--node_hidden_dim', type=int, default=200,
                        help='dim of node hidden embedding in encoder and decoder')
    '''
    parser.add_argument('--graph_embedding_dim', type=int, default=100,
                        help='dim of graph embedding by encoder and also condition for ae decoders')
    parser.add_argument('--latent_dim', type=int, default=56,
                        help='dim of latent z for vae decoders')
    '''
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

    parser.add_argument('--piece_in_dim', type=int, default=208,
                        help='Hidden dim for rnn used in piece generation')
    parser.add_argument('--piece_link_dim', type=int, default=270,
                        help='Hidden dim for rnn used in piece generation')
    parser.add_argument('--graph_pooling', type=str, default='sum',
                        help='Hidden dim for rnn used in piece generation')
    return parser.parse_args()


def load_model(file_path, device):
    args = parse()

    tokenizer = Tokenizer(file_path)
    args.device = device
    encoder = GroupEncoder(**encoder_config(args))  # tokenizer 不影响编码器
    predictor = Predictor(**predictor_config(args), encoder=None)

    decoder = VAEPieceDecoder(**decoder_config(args, tokenizer))
    config = common_config(args)
    #print(config['selected_properties'])
    model = PSVAEModel(config, tokenizer, encoder, decoder, predictor)

    root, task = os.path.split(file_path)

    state_dict = torch.load(root+'/checkpoint.pt',map_location='cpu')
    model.load_state_dict(state_dict)
    model.to(device)
    return model



if __name__ == '__main__':
    args = parse()
    print_log(args)
    print_log('loading data ...')


    path = '/home/pycao/PS-VAE-main-1/data/zinc250k'
    train_set = '/home/pycao/PS-VAE-main-1/data/zinc250k/train.txt'
    '''
    path = '/work/home/zhy_hzau/cpy/PS-VAE-main-1/data/zinc250k'
    train_set = '/work/home/zhy_hzau/cpy/PS-VAE-main-1/data/zinc250k/train.txt'
    '''
    #path = '/home/pycao/PS-VAE-main-1/data/finetune/class_task/bbbp'
    #train_set = '/home/pycao/PS-VAE-main-1/data/finetune/class_task/bbbp/bbbp.csv'

    args.path = path
    df = pd.read_csv(train_set, header=0)
    label = list(df.columns).index('smiles') + 1
    label_list = list(range(label, len(df.columns)))
    tokenizer = Tokenizer(train_set, label=label_list)
    print('tokenizer.num_piece()', tokenizer.num_piece())

    vocab = tokenizer.chem_vocab
    #root = '/home/pycao/hgraph2graph-master/hgraph2graph-master/dataset/dataset/class_task'
    vocab_tensors = tokenizer.get_vocab_data()
    print(vocab_tensors.size())
    #torch.save(vocab_tensors,args.path + '/vocab_tensors
    # .pt')
    #vocab_tensors = torch.load(args.path + '/vocab_ tensors.pt')
    dataset = MolDataset1(train_set, tokenizer)


    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    #print(len(train_loader),len(valid_loader),len(test_loader))

    print('creating model ...')
    if args.gpus == -1:
        loc = torch.device('cpu')
    else:
        loc = torch.device(f'cuda:{args.gpus}')
    args.device = loc

    #config = {**common_config(args), **encoder_config(args, vocab), **decoder_config(args, tokenizer), **predictor_config(args)}
    #config = {**common_config(args,vocab_tensors), **encoder_config(args, vocab_tensors),**predictor_config(args)}
    encoder = GroupEncoder(**encoder_config(args))  # tokenizer 不影响编码器


    num_tasks = 3
    predictor = Predictor(args.node_hidden_dim,num_tasks,encoder=None)

    # print(config['encoder'])
    setup_seed(2021)

    # config['decoder']['tokenizer'] = tokenizer
    #pretrain_file = '/home/pycao/PS-VAE-main-1/data/zinc250k' + '/encoder_checkpoint.pt'

    pretrain_file = None
    if pretrain_file:
        state_dict = torch.load(pretrain_file, map_location='cpu')
        encoder.load_state_dict(state_dict)

    decoder = VAEPieceDecoder(**decoder_config(args, tokenizer))
    config = common_config(args)
    model = PSVAEModel(config, tokenizer, encoder, decoder, predictor)




    print('start training')
    train(model, vocab_tensors, train_loader, train_loader, train_loader, args)
