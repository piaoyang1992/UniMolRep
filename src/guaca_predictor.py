#!/usr/bin/python
# -*- coding:utf-8 -*-
from functools import partial
from tqdm import tqdm
import multiprocessing as mp

from rdkit import Chem
import torch
import torch.nn as nn

from modules.group_vae_model import PSVAEModel
from train_group import load_model
from modules.predictor import Predictor
from data.chem_utils import smiles2molecule
#from data.group_graph_dataset import get_graph
from time import time
import os

class Dataset(torch.utils.data.Dataset):
    def __init__(self, model_ckpt, fpath, scoring_funcs,vocab_tensors,device=-1,
                 process_batch_size=32,
                 num_workers=1):
        super().__init__()
        self.model_ckpt = model_ckpt
        self.device = device
        self.root_path, self.file_name = os.path.split(fpath)

        self.fpath = fpath
        self.scoring_funcs = scoring_funcs
        self.process_batch_size = process_batch_size
        self.num_workers = num_workers
        self.vocab_tensors = vocab_tensors


        self.save_path = os.path.join(self.root_path, self.file_name[:-4] +'smiles_embed.pkl')

        try:
            self.smis, self.embeds, self.scores = torch.load(self.save_path)

        except FileNotFoundError:
            self.smis, self.embeds, self.scores = self.process()


    def load_VAE_model(self):
        #model = PSVAEModel.load_from_checkpoint(self.model_ckpt)
        model = load_model(self.fpath,self.device)
        #model.to(self.device)
        model.eval()
        return model

    def get_embed_scores(self, seq, scoring_funcs, batch_size):
        model = self.load_VAE_model()
        if isinstance(seq, str):
            seq = [seq]
        origin_seq = seq
        embeds, scores = [], [[] for _ in scoring_funcs]
        for begin in tqdm(range(0, len(origin_seq), batch_size)):
            end = min(begin + batch_size, len(origin_seq))
            seq = origin_seq[begin:end]
            good_seq = []
            with torch.no_grad():
                mols = [smiles2molecule(smi) for smi in seq]
                good_mols = []
                for i, mol in enumerate(mols):
                    if mol is not None:
                        good_mols.append(mol)
                        good_seq.append(seq[i])


                zs = model.get_z_from_smis(self.vocab_tensors,good_seq)

                #print(zs.shape)
                embeds.extend(zs.cpu().numpy())
            for i, func in enumerate(scoring_funcs):
                scores[i].extend(func(good_seq))

        return embeds, scores

    def process(self):

        with open(self.fpath, 'r') as fin:
            self.smis = fin.read().strip().split('\n')
        self.embeds, self.scores = [], [[] for _ in self.scoring_funcs]

        if self.num_workers > 1:
            pool = mp.Pool(self.num_workers)
            spliced_smis, splice_size = [], (len(self.smis) + self.num_workers - 1) // self.num_workers
            for begin in range(0, len(self.smis), splice_size):
                end = min(begin + splice_size, len(self.smis))
                smis = self.smis[begin:end]
                spliced_smis.append(smis)

            # get embeds and scores
            res = pool.map(partial(self.get_embed_scores, scoring_funcs=self.scoring_funcs,
                                   batch_size=self.process_batch_size), tqdm(spliced_smis))
            pool.close()
        else:
            res = [self.get_embed_scores(self.smis, self.scoring_funcs, self.process_batch_size)]

        for embeds, scores in res:
            self.embeds.extend(embeds)
            for i, s in enumerate(scores):
                self.scores[i].extend(s)
        print(f'valid number of samples: {len(self.embeds)}')
        torch.save((self.smis, self.embeds, self.scores),self.save_path)
        return self.smis, self.embeds, self.scores

    def __len__(self):
        return len(self.embeds)
    
    def __getitem__(self, idx):
        scores = [mol_scores[idx] for mol_scores in self.scores]
        return self.embeds[idx], scores

    def get_embed_dim(self):
        return len(self.embeds[0])



class InferenceModel:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def emb_to_seq(self, embeddings):
        smis = []
        with torch.no_grad():
            embeddings = torch.tensor(embeddings, device=self.device)
            for z in embeddings:
                mol = self.model.inference_single_z(z, max_atom_num=60, add_edge_th=0.5, temperature=0.8)
                if not isinstance(mol, str) and mol is not None:
                    Chem.SanitizeMol(mol)
                    smis.append(Chem.MolToSmiles(mol))
                else:
                    smis.append(mol)
        return smis

    def seq_to_emb(self, seq):
        if isinstance(seq, str):
            seq = [seq]
        with torch.no_grad():
            mols = [smiles2molecule(smi) for smi in seq]
            ill_idxs, good_mols = [], []
            for i, mol in enumerate(mols):
                if mol is not None:
                    good_mols.append(mol)
                
                
            zs = self.model.get_z_from_batch_mol(mols)
            zs = zs.cpu().numpy()
        return zs


class GuacaPredictor(nn.Module):
    def __init__(self, embed_dim, func_names, scoring_funcs, init_smiles):
        super().__init__()
        self.predictor = Predictor(embed_dim, len(scoring_funcs),None)
        self.pred_loss = nn.MSELoss()
        self.init_smiles = init_smiles
        self.func2idx, self.func2name = {}, {}
        for i, func in enumerate(scoring_funcs):
            key = self.get_func_key(func)
            self.func2idx[key] = i
            self.func2name[key] = func_names[i]
            print(key, func_names[i])
        assert len(self.func2idx) == len(scoring_funcs)

    def get_func_key(self, func):
        scores = func(self.init_smiles)
        key = '\t'.join([str(round(s, 4)) for s in scores])
        return key

    def forward(self, embeds, targets):
        scores = self.predictor.predict_z(embeds)  # [batch_size, num_properties]
        #print(embeds.size(), targets.size())
        loss = self.pred_loss(scores, targets)
        return loss

    def predict(self, embeds, scoring_func=None):
        squeeze = False
        if len(embeds.shape) == 1:
            embeds = embeds.unsqueeze(0)
            squeeze = True
        scores = self.predictor.predict_z(embeds)
        if scoring_func is None:
            return scores
        key = self.get_func_key(scoring_func)
        res = scores[:, self.func2idx[key]]
        if squeeze:
            res = res.squeeze()
        return res