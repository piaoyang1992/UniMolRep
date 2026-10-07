#!/usr/bin/python
# -*- coding:utf-8 -*-
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

#from src.modules.subgraph_encoder  import Encoder
from modules.encoder  import Encoder
from modules.vae_piece_decoder import VAEPieceDecoder
from modules.predictor import Predictor
from data.group_graph_dataset import *
from evaluation.utils import get_normalized_property_scores, PROPS


class PSVAEModel(pl.LightningModule):
    def __init__(self, config, tokenizer):
        super(PSVAEModel, self).__init__()
        self.save_hyperparameters()
        self.config = config
        self.tokenizer = tokenizer
        self.chem_vocab = tokenizer.chem_vocab
        self.predictor = Predictor(**(config['predictor']))
        #print(config['encoder'])
        self.encoder = Encoder(**config['encoder'])
        self.decoder = VAEPieceDecoder(**(config['vae_piece_decoder']))
        # loss
        self.pred_loss = nn.MSELoss()
        # total time
        self.total_time = 0
        self.vocab_data = config['vocab']

    def forward(self, batch, return_accu=False):
        # see data.bpe_dataset for definition of batch
        x, edge_index, edge_attr = batch['x'], batch['edge_index'], batch['edge_attr']  # 原子图
        x_pieces, x_pos = batch['x_pieces'], batch['x_pos']  # 每个原子对应的piece ID
        #edge_mask = batch['edge_mask']
        #group_batch = batch['group_x'], batch['group_edge_index'],batch['group_edge_attr'],batch['group_inter_tensor'],\
                     # batch['group_batch']
        #print(edge_mask.size(), edge_index.size())
        in_piece_edge_idx = torch.tensor(batch['in_piece_edge_idx'])
        edge_attr = edge_attr.view(-1, 1)
        neg_edge_index = batch['neg_edge_index']
        '''
        none_id = torch.nonzero(in_piece_edge_idx == 0).squeeze()

        edge_attr[none_id] = -1000
        

        #_, graph_embedding = self.encoder(self.vocab_data,group_batch)
        # print(group_embedding.size())

        
        pos = x_pos.view(-1)
        group_x = self.vocab_data.index_select(index=batch['group_x'].view(-1), dim=0)
        x_pos = group_x.index_select(index=pos, dim=0).view(x.shape[0], x.shape[1], -1)
        '''
        #print('x_pieces',x_pieces)
        x = x[:,:,0].long()
        x = self.decoder.embed_atom(x, x_pieces, x_pos)  # cat
        #print('embed_x', x)
        x = torch.where(torch.isnan(x), torch.full_like(x, 0), x)

        batch_size, node_num, node_dim = x.shape  # [batch_size, 最大原子数，最大原子种类]
        graph_ids = torch.repeat_interleave(torch.arange(0, batch_size, device=x.device), node_num)
        _, all_x = self.encoder.embed_node(x.view(-1, node_dim), edge_index, edge_attr)
        #print('all_x', all_x)
        # [batch_size, dim_graph_feature]
        #print(torch.any(torch.isnan(x.view(-1, node_dim))), torch.any(torch.isnan(all_x.view(-1, node_dim))))
        graph_embedding = self.encoder.embed_graph(all_x, graph_ids, batch['atom_mask'].flatten())

        pred_prop = self.predictor(graph_embedding)  # [batch_size, num_properties]
        golden = batch['props'].reshape(batch_size, -1)[:,self.config['selected_properties']]
        golden = golden.float()
        pred_loss = self.pred_loss(pred_prop, golden)
        #pred_loss = 0
        return pred_loss

    def cal_beta(self):
        step = self.global_step
        warmup = self.config['kl_warmup']
        if step < warmup:
            beta = 0
        else:
            step_beta, anneal_iter = self.config['step_beta'], self.config['kl_anneal_iter']
            beta = min(self.config['max_beta'], ((step - warmup) // anneal_iter + 1) * step_beta)
        beta += self.config['beta']
        return beta

    def weighted_loss(self, pred_loss, rec_loss, kl_loss):
        alpha = self.config['alpha']
        #alpha = 1
        beta = self.cal_beta()
        #print(beta)
        return alpha * rec_loss + (1 - alpha) * pred_loss + beta * kl_loss

    def training_step(self, batch):
        #print(batch)
        st = time.time()
        pred_loss= self.forward(batch)
        print('pred_loss', pred_loss)

        '''
        self.log('predictor loss', pred_loss)
        self.log('reconstruct loss', rec_losses[0])
        self.log('piece loss', rec_losses[1])
        self.log('edge loss', rec_losses[2])
        self.log('kl loss', rec_losses[3])
        self.log('beta', self.cal_beta())
        self.total_time += time.time() - st
        self.log('total time', self.total_time)
        '''
        return pred_loss
    
    def validation_step(self, batch):
        pred_loss = self.forward(batch, return_accu=True)
        self.log('test_loss', pred_loss)
        return pred_loss



    def validation_epoch_end(self, outputs):
        return
        device = next(self.predictor.parameters()).device
        # reconstruct molecule with predicted property
        st = time.time()
        latents = self.sample_z(500, device=device)  # tensor, [n, latent_size]
        with torch.no_grad():
            pred_y = self.predictor(latents)  # [batch_size, num_properties]
            res_data = [self.inference_single_z(z,
                                                max_atom_num=80,
                                                add_edge_th=0.5,
                                                temperature=0.8)\
                        for z in latents]
            mols = [self.return_data_to_mol(d) for d in res_data]
            true_y = torch.tensor([get_normalized_property_scores(mol) for mol in mols], device=device)
            true_y = true_y[:,self.config['selected_properties']]
            rec_prop_loss = self.pred_loss(pred_y, true_y)
        self.log('rec_prop_loss', rec_prop_loss)
        print(f'Sampling rec_prop test elapsed {round(time.time() - st, 2)}s')

    def test_step(self, batch):
        pred_loss = self.forward(batch)
        self.log('test_loss', pred_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'])
        lam = lambda epoch: 1 / (epoch + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lam)
        return [optimizer], [scheduler]

    # interface
    def sample_z(self, n, device=None):
        if device is not None:
            return torch.randn(n, self.decoder.latent_dim, device=device)
        return torch.randn(n, self.decoder.latent_dim, device=self.device)


    def inference_single_z(self, z, max_atom_num, add_edge_th, temperature):
        # max_atom_num and temperature has no use here. Directly generate a molecule
        return self.decoder.inference(z, max_atom_num, add_edge_th, temperature)
    
    def inference_batch_z(self, zs, max_atom_num, add_edge_th, temperature, constraint_mol=None):
        # not accelerating, just a loop
        return [self.inference_single_z(z, max_atom_num, add_edge_th, temperature) for z in zs]

    def inference_single_z_constraint(self, z, max_atom_num, add_edge_th, temperature, constraint_mol):
        return self.decoder.inference_constraint(z, max_atom_num, add_edge_th, temperature, constraint_mol)

    def predict_props(self, z):
        props = self.predictor(z)
        # padding to given length in evaluation library
        if len(props.shape) == 1:
            padded_props = [torch.tensor(0, device=props.device) for _ in PROPS]
            for i, val in zip(self.config['selected_properties'], props):
                padded_props[i] = val
            return torch.stack(padded_props)
        else:
            padded_props = torch.zeros(props.shape[0], len(PROPS), device=props.device)
            prop_ids = self.config['selected_properties']
            padded_props[:, prop_ids] = props
            return padded_props
    
    def get_z(self, batch):
        '''
        x, edge_index, edge_attr = batch['x'], batch['edge_index'], batch['edge_attr']
        x_pieces, x_pos = batch['x_pieces'], batch['x_pos']
        x = self.decoder.embed_atom(x, x_pieces, x_pos)
        batch_size, node_num, node_dim = x.shape
        graph_ids = torch.repeat_interleave(torch.arange(0, batch_size, device=x.device), node_num)
        _, all_x = self.encoder.embed_node(x.view(-1, node_dim), edge_index, edge_attr)
        # [batch_size, dim_graph_feature]
        graph_embedding = self.encoder.embed_graph(all_x, graph_ids, batch['atom_mask'].flatten())
        z_vecs, _ = self.decoder.rsample(graph_embedding)
        '''
        x, edge_index, edge_attr = batch['x'], batch['edge_index'], batch['edge_attr']  # 原子图
        x_pieces, x_pos = batch['x_pieces'], batch['x_pos']  # 每个原子对应的piece ID
        edge_attr = edge_attr.view(-1, 1)
        #group_batch = batch['group_x'], batch['group_edge_index'],batch['group_edge_attr'],batch['group_inter_tensor'],\
                      #batch['group_batch']
        x = x[:, :, 0].long()
        x = self.decoder.embed_atom(x, x_pieces, x_pos)  # cat
        batch_size, node_num, node_dim = x.shape  # [batch_size, 最大原子数，最大原子种类]
        graph_ids = torch.repeat_interleave(torch.arange(0, batch_size, device=x.device), node_num)
        _, all_x = self.encoder.embed_node(x.view(-1, node_dim), edge_index, edge_attr)
        graph_embedding = self.encoder.embed_graph(all_x, graph_ids, batch['atom_mask'].flatten())
        z_vecs, _ = self.decoder.rsample(graph_embedding)
        return z_vecs
    
    def get_z_from_smi(self, s):
        return self.get_z_from_return_data(s)

    def get_z_from_return_data(self, s):
        '''reprocess generated data and encode its latent variable
           gen is an instance of Mol of rdkit.Chem'''
        step1_res = MolDataset.process_step1(s, self.tokenizer)
        step2_res = MolDataset.process_step2(step1_res, self.tokenizer)
        res = MolDataset.process_step3([step2_res], self.tokenizer, device=self.device)
        return self.get_z(res).squeeze()
    
    def return_data_to_mol(self, data):
        return data

    def get_dataloader(self, fname, **kwargs):
        return get_dataloader(fname, self.tokenizer, **kwargs)

    def get_from_batch(self, batch, key):
        if key == 'batch_size':
            return batch['x'].shape[0]
        elif key == 'props':
            return batch['props']