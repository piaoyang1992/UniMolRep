#!/usr/bin/python
# -*- coding:utf-8 -*-
from functools import partial
import os
import argparse
from typing import List, Optional
from tqdm import tqdm
import random
import joblib
from joblib import delayed

import torch
import torch.nn.functional as F
import numpy as np
from guacamol.assess_goal_directed_generation import assess_goal_directed_generation
from guacamol.goal_directed_generator import GoalDirectedGenerator
from guacamol.scoring_function import ScoringFunction
from guacamol.utils.helpers import setup_default_logger
from train_group import load_model
#from modules.group_vae_model import PSVAEModel
from src.data.chem_utils import molecule2smiles, smiles2molecule
from src.data.logger import print_log
from generate import gen
from src.data.mol_group_graph import Tokenizer
from guaca_predictor import GuacaPredictor

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


class PSVAEGenerator(GoalDirectedGenerator):
    def __init__(self, gpu, model_ckpt, pred_ckpt, vocab_tensors, num_workers=1):
        super().__init__()
        self.model_ckpt = model_ckpt
        self.pred_ckpt = pred_ckpt
        self.gpu = gpu
        self.root_path, self.file_name = os.path.split(model_ckpt)
        # generate related
        self.max_atom_num = 60
        self.add_edge_th = 0.5
        self.temperature = 0.8
        self.beam = 5
        self.num_workers = num_workers
        self.predictor = self.init_ckpt()
        self.model = self.load_vae_model()
        self.vocab_tensor = vocab_tensors

    def load_vae_model(self):

        vae_model = load_model(self.model_ckpt,self.gpu)
        #model.to(self.device)
        vae_model.eval()
        return vae_model

    def init_ckpt(self):
        from guaca_predict_train import load_predict_model
        predictor = load_predict_model(self.pred_ckpt,self.gpu)
        predictor.eval()
        '''
        if self.gpu == -1:
            loc = torch.device('cpu')
        else:
            loc = torch.device(f'cuda:{self.gpu}')
        predictor.to(loc)
        '''
        #self.device = loc
        #self.load_model()
        return predictor

    def top_k(self, smiles, scoring_function, k):
        if len(smiles) == 1:
            return smiles
        smiles = list(set(smiles))  # unique smiles
        scores = [scoring_function.score(s) for s in smiles]
        scored_smiles = list(zip(scores, smiles))
        scored_smiles = sorted(scored_smiles, key=lambda x: x[0], reverse=True)
        return [smile for _, smile in scored_smiles][:k]

    def generate_optimized_molecules(self, scoring_function: ScoringFunction, number_molecules: int, starting_population: Optional[List[str]] = None) -> List[str]:
        print_log(f'number of molecules: {number_molecules}')
        num_cand = number_molecules
        if self.num_workers > 1:
            func = partial(self.single_process_generate, scoring_function=scoring_function, starting_population=starting_population)
            splice_sizes, single_process_size = [], (num_cand + self.num_workers - 1) // self.num_workers
            while num_cand > 0:
                splice_sizes.append(min(num_cand, single_process_size))
                num_cand -= splice_sizes[-1]
            print_log(f'splices: {splice_sizes}')
            joblist = (delayed(func)(s) for s in splice_sizes)
            res = joblib.Parallel(n_jobs=self.num_workers)(joblist)
        else:
            res = [self.single_process_generate(num_cand,scoring_function)]
        smis = []
        for s in res:
            smis.extend(s)
        return self.top_k(smis, scoring_function, number_molecules)

    def single_process_generate(self, number_molecules:int, scoring_function: ScoringFunction, starting_population: Optional[List[str]] = None) -> List[str]:
        #self.load_vae_model()
        #self.init_ckpt()
        latents = self.model.sample_z(number_molecules)  # tensor or ndarray, [n, latent_size]
        target = 2
        config = {
            'lr': 0.01,
            'max_iter': 100,
            'patience': 3,
        }
        optimized_latents = [self.direct_optimize(z, scoring_function.score_list, target, **config) for z in tqdm(latents)]
        optimized_graphs = [self.beam_gen(z, self.beam, scoring_function.score_list)
                            for z in tqdm(optimized_latents)]
        smis = [molecule2smiles(g) for g in optimized_graphs]
        #print(smis)
        return smis

    def direct_optimize(self, z, scoring_func, target, lr, max_iter, patience):
        '''direct gradient optimization, return optimized z'''

        optimized = z.clone()
        optimized.requires_grad = True
        optimizer = torch.optim.Adam([optimized], lr=lr)
        best_loss, not_inc, best_z = 100, 0, optimized.clone()
        target = torch.tensor(target, device=optimized.device, dtype=torch.float)
        for i in range(max_iter):
            score = self.predictor.predict(optimized, scoring_func)
            loss = F.mse_loss(score, target)
            optimizer.zero_grad()
            self.predictor.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.item()
            print_log(f'step {i}, loss: {loss}', level='DEBUG')
            if loss < best_loss:
                best_loss, not_inc, best_z = loss, 0, optimized.clone()
            else:
                not_inc += 1
                if not_inc > patience:
                    break
        return best_z

    def beam_gen(self, z, beam, scoring_func, constraint_mol=None):
        gens = [gen(self.model, self.vocab_tensor, z, self.max_atom_num, self.add_edge_th, self.temperature, constraint_mol) for _ in range(beam)]
        mols = [g for g in gens if g != None]
        smis = [molecule2smiles(mol) for mol in mols]
        scores = scoring_func(smis)
        sorted_idx = sorted([i for i in range(len(scores))], key=lambda x: scores[x], reverse=True)
        best_idx = sorted_idx[0]
        if constraint_mol is not None:
            return mols
        return mols[best_idx]


def parse():
    parser = argparse.ArgumentParser(description='Goal-Directed learning benchmark for vae models',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    #parser.add_argument('--ckpt', type=str, required=True, help='Full path to vae model')
    parser.add_argument('--ckpt', type=str, default='/home/pycao/PS-VAE-main-1/data/zinc250k/checkpoint.pt')
    parser.add_argument('--train_set', type=str, default='/home/pycao/PS-VAE-main-1/data/zinc250k/train.txt',
                        help='path of training dataset')

    parser.add_argument('--vocab', type=str, help='HierG2G needs vocabulary file')
    #parser.add_argument('--pred_ckpt', type=str, required=True, help='Full path predictor model')
    parser.add_argument('--pred_ckpt', type=str, default='/home/pycao/PS-VAE-main-1/data/zinc250k/guaca_predict_checkpoint.pt')
    #parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--output_dir', type=str,
                        default='/home/pycao/PS-VAE-main-1/data/zinc250k')
    parser.add_argument('--suite', default='v2')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU to use, -1 for cpu')

    return parser.parse_args()


def main(args):
    if args.gpu == -1:
        loc = torch.device('cpu')
    else:
        loc = torch.device(f'cuda:{args.gpu}')
    args.device = loc
    setup_seed(args.seed)
    setup_default_logger()
    tokenizer = Tokenizer(args.train_set, label=None)
    vocab_tensors = tokenizer.get_vocab_data()
    json_file_path = os.path.join(args.output_dir, 'goal_directed_learning_results.json')
    model = PSVAEGenerator(args.device, args.train_set, args.pred_ckpt, vocab_tensors)
    assess_goal_directed_generation(
        model,
        json_output_file=json_file_path,
        benchmark_version=args.suite
    )


if __name__ == '__main__':
    main(parse())
