#!/usr/bin/python
# -*- coding:utf-8 -*-
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

#import pytorch_lightning as pl

from data.chem_utils import smiles2molecule, valence_check, cnt_atom, cycle_check, molecule2smiles, compare_similar
from rdkit import Chem
from data.group_graph_dataset import atom_features, bond_features,get_graph
from data.mol_group_graph import TreeCodec
from data.molecule_frags_no_dot import get_sub_mol
from evaluation.utils import get_normalized_property_scores, PROPS, map_prop_to_idx
import networkx as nx
from rdkit.Chem.MolStandardize import rdMolStandardize
from torch_geometric.data import Batch
from torch_geometric.data import Data
from rdkit.Chem import MACCSkeys
import math




class PSVAEModel(nn.Module):
    def __init__(self,config, tokenizer,encoder,decoder,predictor):
        super(PSVAEModel, self).__init__()
        #self.save_hyperparameters()
        self.config = config
        self.tokenizer = tokenizer
        self.chem_vocab = tokenizer.chem_vocab
        self.predictor = predictor
        self.encoder = encoder
        self.decoder = decoder
        #self.predictor = Predictor(**config['predictor'])
        #print(config['encoder'])
        #self.encoder = MolEncoder(**config['encoder']) # tokenizer 不影响编码器

        #config['decoder']['tokenizer'] = tokenizer
        #self.decoder = VAEPieceDecoder(**config['decoder'])
        self.pred_loss = nn.MSELoss()
        #self.pred_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.total_time = 0


    def forward(self, vocab_data, batch):
        # see data.bpe_dataset for definition of batch
        #graph_embedding, group_x = self.encoder(batch)
        #print(batch)
        atom_batch, group_batch = batch

        z_vecs, kl_loss = self.encoder(vocab_data, group_batch)
        #piece_encoder = self.encoder.encoder_node(vocab_data, group_batch.x, group_batch.edge_index_set)

        batch_size = group_batch.num_graphs
        piece_loss,  edge_link_loss, edge_loss = self.decoder(z_vecs=z_vecs, piece_encoder=None, batch=batch)
        mol_loss = piece_loss + edge_link_loss + edge_loss
        res = mol_loss, piece_loss, edge_link_loss, edge_loss, kl_loss
        #print(res)
        # res (rec_loss, piece_loss, edge_loss, kl_loss)
        # predict props using z
        #print(z_vecs.size()) #[128, 200])
        #print(z_vecs)
        pred_prop = self.predictor.predict_z(z_vecs)  # [batch_size, num_properties]
        #print(pred_prop)

        golden = atom_batch.prop.reshape(batch_size, -1)[:,self.config['selected_properties']]
        #golden = atom_batch.prop.reshape(batch_size, -1)
        golden = golden.float()
        #print(pred_prop.size(), golden.size())
        pred_loss = self.pred_loss(pred_prop, golden).mean()
        #pred_loss = 0
        return pred_loss, res

    def cal_beta(self):
        step = 0
        warmup = self.config['kl_warmup']
        if step < warmup:
            beta = 0
        else:
            step_beta, anneal_iter = self.config['step_beta'], self.config['kl_anneal_iter']
            beta = min(self.config['max_beta'], ((step - warmup) // anneal_iter + 1) * step_beta)
        beta += self.config['beta']
        return beta

    def weighted_loss(self, pred_loss, rec_loss, kl_loss):
        #print(pred_loss, rec_loss, kl_loss)
        alpha = self.config['alpha']
        #alpha = 1
        beta = self.cal_beta()
        #print(beta)
        #alpha * rec_loss + (1 - alpha) * pred_loss + beta * kl_loss
        return rec_loss + pred_loss + beta * kl_loss

    def training_step(self, vocab_data, batch):
        #print(batch)
        st = time.time()
        pred_loss, rec_losses = self.forward(vocab_data, batch)
        # (rec_loss, piece_loss, edge_loss, kl_loss)
        # rec_loss, piece_loss, piece_link_loss, edge_loss, kl_loss

        loss = self.weighted_loss(pred_loss, rec_losses[0], rec_losses[-1])
        print('piece loss', rec_losses[1], 'atom link loss', rec_losses[2],'link contribution loss',rec_losses[3])

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
        return loss
    
    def validation_step(self, vocab_data, batch):
        pred_loss, rec_losses = self.forward(vocab_data, batch)
        rec_losses, accus = rec_losses
        loss = self.weighted_loss(pred_loss, rec_losses[0], rec_losses[-1]) # first is overall rec loss, last is kl loss
        self.log('val_loss', loss)
        for i, name in enumerate(['piece accu', 'edge accu']):
            self.log(name, accus[i])

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

    def test_step(self, vocab_data, batch):
        pred_loss, rec_losses = self.forward(vocab_data, batch)
        loss = self.weighted_loss(pred_loss, rec_losses[0], rec_losses[-1])
        self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'])
        lam = lambda epoch: 1 / (epoch + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lam)
        return [optimizer], [scheduler]

    # interface
    def sample_z(self, n, device=None):
        if device is not None:
            return torch.randn(n, self.decoder.dim_hidden, device=device)
        return torch.randn(n, self.decoder.dim_hidden, device=device)

    def get_link_prediction(self,i,j,cls_i,cls_j,cur_fp,atom_embedding,pieces_embed,z):

        embedding_i = torch.cat([atom_embedding[i].unsqueeze(0), pieces_embed[cls_i].unsqueeze(0), z], dim=1)
        embedding_j = torch.cat([atom_embedding[j].unsqueeze(0), pieces_embed[cls_j].unsqueeze(0), z], dim=1)
        embedding_i = self.decoder.atom_embedding_predictor(embedding_i)
        embedding_j = self.decoder.atom_embedding_predictor(embedding_j)
        link_logits = embedding_i * embedding_j
        cur_fp = torch.tensor(cur_fp, dtype=torch.float).unsqueeze(0)
        link_logits = torch.cat([cur_fp, link_logits], dim=1)
        link_logits = self.decoder.atom_pieces_pred(link_logits)[:,1]  # 预测连接概率
        return link_logits.squeeze()

    def inference(self,vocab_data,z, max_atom_num, add_edge_th, temperature):
        # z: [latent_dim]
        z = z.unsqueeze(0)  # [1, latent_dim] # Z包含整个原子图的信息
        batch_size = z.shape[0]  # 1
        # predict piece
        # [1, 1, embedding_dim]


        hidden = self.decoder.z_to_hidden(z).unsqueeze(0)  # #1,batch_size, -1 unsqueeze for 1 layer of rnn
        # hidden = torch.zeros(self.rnn.num_layers, z.shape[0], self.rnn.hidden_size, device=z.device)

        inter_start_token = self.tokenizer.get_smiles_location('<inter_start>')
        piece_start_token = self.tokenizer.get_smiles_location('<piece_start>')
        inter_pad_token = self.tokenizer.inter_pad_idx()
        pieces_pad_token = self.tokenizer.piece_pad_idx()
        piece_ids, cur_piece_id = [], None
        inter_num_list, cur_inter_num = [inter_start_token,1],None
        cur_atom_num = 0
        #pieces_embed = []

        cur_piece = torch.tensor([piece_start_token], dtype=torch.long, device=z.device) # 1,1
        cur_inter = torch.tensor([inter_start_token,1], dtype=torch.long, device=z.device)  # 1,2

        mask_tensor = self.tokenizer.get_mask_tensor().T.to(z.device)
        sort_vocab_df = self.tokenizer.sort_df
        h1, h2 = z.clone(), z.clone()

        while cur_piece_id != piece_start_token:
            # rnn_in = torch.cat([cur_piece, z.unsqueeze(0)], dim=-1)

            piece_in = cur_piece
            inter_in = cur_inter
            #print(inter_in)
            #print(rnn_in.size(), hidden.size())
            #piece_embed, hidden = self.decoder.rnn(rnn_in, hidden) #[1, 1, hidden_dim]
            piece_out, inter_out, h1,h2 = self.decoder.DRNN.decode(piece_in,inter_in, h1,h2)
            piece_out, inter_out = piece_out.squeeze(), inter_out.squeeze() # [piece_out_size]  [2, inter_out_size]
            #print(inter_out)
            #output = self.decoder.to_vocab(piece_embed)  # [1, 1, num_piece_type]

            if len(inter_num_list) == 2:
                #inter_out[0, 1] = float('inf')  # at least output one piece
                inter_out[1, inter_pad_token] = float('-inf')
                inter_out[1, inter_start_token] = float('-inf')
            #inter_probs = torch.softmax(inter_out / temperature, dim=-1)  # [num_piece_type]
            #piece_probs = torch.softmax(piece_out / temperature, dim=-1)  # [num_piece_type]
            cur_inter_id = torch.argmax(inter_out, dim=-1)
            #print(cur_inter_id)
            #cur_inter_id[0] = inter_in.squeeze()[1]
            #print(cur_inter_id)

            mask_select = mask_tensor[inter_in[1],:]
            piece_out[mask_select==1] = float('-inf')
            #piece_out[pieces_pad_token] = float('-inf')


            cur_piece_id = torch.argmax(piece_out, dim=-1)

            #cur_piece_id = torch.multinomial(piece_probs, num_samples=1)  # 对input的每一行做n_samples次取值，输出的张量是每一次取值时input张量对应行的下标
            #cur_inter_id = torch.multinomial(inter_probs, num_samples=1)

            #cur_piece = self.decoder.piece_embedding(cur_piece_id).unsqueeze(0)
            cur_piece =  torch.tensor(cur_piece_id, dtype=torch.long, device=z.device)
            cur_inter = torch.tensor(cur_inter_id, dtype=torch.long, device=z.device)

            cur_smiles = sort_vocab_df.iloc[cur_piece_id.item(),inter_in[1].item()]
            inter_num_list.append(cur_inter_id[1].item())
            #cur_atom_num += cnt_atom(self.tokenizer.idx_to_piece(cur_piece_id))
            cur_atom_num += cnt_atom(cur_smiles)
            piece_ids.append(cur_smiles)
            #pieces_embed.append(piece_embed.squeeze())
            if cur_atom_num > max_atom_num:  # the last one will still be dropped
                break
            if len(piece_ids) >= 100:  # 0 is padding, the last one dropped
                break

        mol_seq = piece_ids[:-1]  # get rid of end token

        mol_tree = TreeCodec()

        #mol_seq = [self.tokenizer.idx_to_piece(i) for i in piece_ids]
        print(inter_num_list)
        print(mol_seq)
        piece_nodes, pieces_edges = mol_tree.decode_from_sequence(mol_seq)
        print(piece_nodes, pieces_edges)
        #piece_nodes = [piece_nodes[i] for i in range(len(piece_nodes))]
        vocab_df = self.tokenizer.vocab_df
        group_tree = nx.Graph()
        for i,sub_smiles in piece_nodes.items():
            #pid = self.tokenizer.piece2idx[sub_smiles]
            sub_mol = self.tokenizer.piece_to_mol(sub_smiles, marker='*')
            inter_num = sub_smiles.count('[*]')
            group_tree.add_node(i,sub_mol=sub_mol,sub_smiles=sub_smiles,inter_num=inter_num)

        for a,b in pieces_edges:
            group_tree.add_edge(a,b)


        for node, degree in group_tree.degree():
            if group_tree.nodes[node]['inter_num'] == degree:
                pass
            else:
                sub_smiles = group_tree.nodes[node]['sub_smiles']
                print(sub_smiles, degree)
                change_df = vocab_df[vocab_df['inter_num'] == degree]
                change_df['sim'] = change_df['smiles'].apply(lambda x: compare_similar(x,sub_smiles))
                max_sim = change_df['sim'].max()
                change_smiles = change_df.loc[change_df['sim'] == max_sim, 'smiles']
                group_tree.nodes[node]['sub_smiles'] = change_smiles
                change_pid = self.tokenizer.piece_to_idx(change_smiles)
                piece_nodes[node] = change_pid

                group_tree.nodes[node]['sub_mol'] = self.tokenizer.idx_to_mol(change_pid)

                group_tree.nodes[node]['inter_num'] = change_smiles.count('[*]')
                print(degree,'changing',sub_smiles, change_smiles)


        merged_mol = Chem.RWMol()
        for pid, attr in group_tree.nodes(data=True):
            mol_i = attr['sub_mol']
            for atom in mol_i.GetAtoms():
                atom.SetUnsignedProp('piece_index', pid)
                atom.SetUnsignedProp('atom_id', atom.GetIdx())
            merged_mol = Chem.CombineMols(merged_mol, mol_i)

        merged_mol = Chem.RWMol(merged_mol)
        nodes_id = [self.tokenizer.piece2idx[piece_nodes[i]] for i in range(len(piece_nodes))]
        group_batch = Data(x=torch.tensor(nodes_id, dtype=torch.long),
                           edge_index=torch.tensor(pieces_edges, dtype=torch.long).T)
        vocab_data = self.tokenizer.get_vocab_data()  # 解码数据集的tokenizer
        pieces_embed = self.decoder.group_gnn.encoder_node(vocab_data, group_batch, edge_attr=False)

        cls_inter = {i:[] for i in group_tree.nodes()}
        inter_atoms_c = {}
        atom_cls = [0 for i in range(merged_mol.GetNumAtoms())]

        for atom in merged_mol.GetAtoms():
            piece_index = int(atom.GetProp('piece_index'))
            inter_num = int(atom.GetProp('inter'))
            atom_cls[atom.GetIdx()] = piece_index
            if inter_num > 0:
                cls_inter[piece_index] = cls_inter[piece_index] + [atom.GetIdx()]
                inter_atoms_c[atom.GetIdx()] = inter_num

        print(cls_inter, inter_atoms_c)
        x, edge_index, edge_attr = [],[],[]
        for i in range(merged_mol.GetNumAtoms()):
            atom = merged_mol.GetAtomWithIdx(i)
            x.append(atom_features(atom))

        for bond in merged_mol.GetBonds():
            begin, end = bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()
            edge_attr.append(bond_features(bond))
            edge_index.append(sorted([begin, end]))  # do not repeat for storage

        x = torch.tensor(x, device=z.device)


        if len(edge_index) == 0:
            edge_index = torch.randn(2, 0, device=z.device).long()

        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=z.device).t().contiguous()

        edge_attr = torch.tensor(edge_attr, dtype=torch.long, device=z.device)
        atom_embedding = self.decoder.atom_gnn.encoder_node(x, edge_index, edge_attr)  # [n, node_embeding_dim]
        merged_mol.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(merged_mol)

        #inter_pairs = []
        for id, (cls_i, cls_j) in enumerate(pieces_edges):
            cur_fp = list(MACCSkeys.GenMACCSKeys(merged_mol))
            inter_i, inter_j = cls_inter[cls_i], cls_inter[cls_j]
            link_prediction = []
            atom_link_id = []
            for a_i in inter_i:
                for a_j in inter_j:
                    atom_link_id.append((a_i, a_j))
                    link_logits = self.get_link_prediction(a_i,a_j,cls_i,cls_j,cur_fp,atom_embedding,pieces_embed,z)  # 预测连接概率
                    link_prediction.append(link_logits)
            #m = max(link_prediction)
            link = list(zip(link_prediction, atom_link_id))
            #inter_pair = atom_link_id[link_prediction.index(m)]
            sorted_link = sorted(link, key=lambda x: x[0], reverse=True)
            #print(sorted_link)
            #print(inter_atoms_c)
            for p, (a,b) in sorted_link:
                if inter_atoms_c[a] and inter_atoms_c[b]:
                    merged_mol.AddBond(a, b, Chem.BondType.SINGLE)
                    inter_atoms_c[a] -= 1
                    inter_atoms_c[b] -= 1
                    print(a,b)
                    break

            merged_mol.UpdatePropertyCache(strict=False)
            Chem.GetSymmSSSR(merged_mol)

        inter_atoms = [k for k,v in inter_atoms_c.items() if v > 0]
        from itertools import combinations

        while len(inter_atoms):
            cur_fp = list(MACCSkeys.GenMACCSKeys(merged_mol))
            pairs = list(combinations(inter_atoms, 2))

            link_prediction = [self.get_link_prediction(x,y,atom_cls[x],atom_cls[x],cur_fp,atom_embedding,pieces_embed,z) for x,y in pairs]
            link = list(zip(link_prediction, pairs))
            #inter_pair = atom_link_id[link_prediction.index(m)]
            sorted_link = sorted(link, key=lambda x: x[0])


            for p, (a,b)in sorted_link:
                if inter_atoms_c[a] and inter_atoms_c[b]:
                    merged_mol.AddBond(a, b, Chem.BondType.SINGLE)
                    inter_atoms_c[a] -= 1
                    inter_atoms_c[b] -= 1
                    print(a,b)
                    break

            merged_mol.UpdatePropertyCache(strict=False)
            Chem.GetSymmSSSR(merged_mol)
            inter_atoms = [k for k, v in inter_atoms_c.items() if v > 0]
        try:
            Chem.SanitizeMol(merged_mol)
        except:
            print('Sanitize error')
            merged_mol = None

        return merged_mol


    def decode(self, vocab_data,z, max_atom_num, add_edge_th, temperature):
        # z: [latent_dim]
        z = z.unsqueeze(0)  # [1, latent_dim] # Z包含整个原子图的信息
        batch_size = z.shape[0]  # 1
        # predict piece
        # [1, 1, embedding_dim]

        cur_piece = self.decoder.piece_embedding(torch.tensor([[self.tokenizer.end_idx()]], dtype=torch.long, device=z.device))
        hidden = self.decoder.latent_to_rnn_hidden(z).unsqueeze(0)  # #1,batch_size, -1 unsqueeze for 1 layer of rnn
        # hidden = torch.zeros(self.rnn.num_layers, z.shape[0], self.rnn.hidden_size, device=z.device)
        piece_ids, cur_piece_id = [], None
        cur_atom_num = 0
        pieces_embed = []
        while cur_piece_id != self.tokenizer.end_idx():
            # rnn_in = torch.cat([cur_piece, z.unsqueeze(0)], dim=-1)

            rnn_in = cur_piece
            #print(rnn_in.size(), hidden.size())
            piece_embed, hidden = self.decoder.rnn(rnn_in, hidden) #[1, 1, hidden_dim]
            output = self.decoder.to_vocab(piece_embed)  # [1, 1, num_piece_type]
            output = output.squeeze()
            output[self.tokenizer.pad_idx()] = float('-inf')
            if len(piece_ids) == 0:
                output[self.tokenizer.end_idx()] = float('-inf')  # at least output one piece
            probs = torch.softmax(output / temperature, dim=-1)  # [num_piece_type]
            cur_piece_id = torch.multinomial(probs, num_samples=1)  # 对input的每一行做n_samples次取值，输出的张量是每一次取值时input张量对应行的下标
            cur_piece = self.decoder.piece_embedding(cur_piece_id).unsqueeze(0)
            cur_piece_id = cur_piece_id.item()
            cur_atom_num += cnt_atom(self.tokenizer.idx_to_piece(cur_piece_id))
            piece_ids.append(cur_piece_id)
            #pieces_embed.append(piece_embed.squeeze())
            if cur_atom_num > max_atom_num:  # the last one will still be dropped
                break
            if len(piece_ids) >= 100:  # 0 is padding, the last one dropped
                break

        piece_ids = piece_ids[:-1]  # get rid of end token
        #pieces_embed = pieces_embed[:-1]
        #print(piece_ids)
        # vocab_data 表示从该分子集中抽样
        mol_seq = []
        mol_tree = TreeCodec()
        for i in piece_ids:
            if i == self.tokenizer.idx2piece[self.branch_start]:
                mol_seq.append('(')
            elif i == self.tokenizer.self.idx2piece(self.branch_end):
                mol_seq.append(')')
            else:
                mol_seq.append(i)

        piece_nodes, pieces_edges = mol_tree.decode_from_sequence(piece_ids)
        piece_nodes = [piece_nodes[i] for i in range(len(piece_nodes))]
        print(piece_nodes, pieces_edges)

        group_batch = Data(x=torch.tensor(piece_nodes,dtype=torch.long),edge_index=torch.tensor(pieces_edges,dtype=torch.long))
        vocab_data = self.tokenizer.get_vocab_data()  # 解码数据集的tokenizer
        pieces_embed = self.decoder.group_gnn.encoder_node(vocab_data, group_batch, edge_attr=False)



        # print(piece_matrix)

        # graph embedding and start to construct a molecule
        x, edge_index, edge_attr = [], [], []
        groups = [[]] * len(piece_ids)
        aid2gid = {}  # map atom idx to group idx
        aid2bid = {}  # map atom idx to connected block (bid)
        block_atom_cnt = []
        #gen_mol = Chem.RWMol()  # generated mol
        edge_sets = []  # record each atom is connected to which kinds of bonds
        x_pieces, x_pos, x_type = [], [], []
        x_inter = []
        atoms = []
        # count_atoms = 0



        merged_mol = Chem.RWMol()

        for pos, pid in enumerate(piece_ids):
            mol_i = self.tokenizer.idx_to_mol(pid, marker='*')
            #uncharger = rdMolStandardize.Uncharger()
            #mol_i = uncharger.uncharge(mol_i)
            #print(Chem.MolToSmiles(mol_i))

            for atom in mol_i.GetAtoms():
                atom.SetUnsignedProp('piece_index', pos)
                atom.SetUnsignedProp('atom_id', atom.GetIdx())
            merged_mol = Chem.CombineMols(merged_mol, mol_i)



        for i in range(merged_mol.GetNumAtoms()):
            atom = merged_mol.GetAtomWithIdx(i)
            atoms.append(self.tokenizer.chem_vocab.atom_to_idx(atom.GetSymbol()))
            x.append(atom_features(atom))
            edge_sets.append([])
            piece_index = int(atom.GetProp('piece_index'))
            groups[piece_index] = groups[piece_index] + [i]
            x_pieces.append(piece_index)
            #x_pos.append(int(atom.GetProp('atom_id')))
            #x_type.append(pieces_type[piece_index])

            x_inter.append(int(atom.GetProp('inter')))

        for bond in merged_mol.GetBonds():
            begin, end = bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()
            edge_attr.append(bond_features(bond))
            edge_index.append(sorted([begin, end]))  # do not repeat for storage
            edge_sets[begin].append(bond.GetBondType())
            edge_sets[end].append(bond.GetBondType())

        '''
        x_inter_embedding = torch.tensor(x_inter, dtype=torch.long, device=z.device)
        x_inter_embedding = x_inter_embedding.reshape(-1, 1)
        x_type_embedding = torch.tensor(x_type, dtype=torch.long, device=z.device)
        x_type_embedding = self.decoder.pieces_type_embedding(x_type_embedding)
        '''
        # pos = x_pos.view(-1)
        x = torch.tensor(x, device=z.device)
        #x = x[:, 0].long()
        x_pieces_embed = torch.tensor(x_pieces, dtype=torch.long, device=z.device)
        x_pieces_embed = pieces_embed.index_select(index=x_pieces_embed, dim=0)
        #node_x = self.decoder.atom_gnn.embed_atom(x, x_pieces_embed, x_pos)

        if len(edge_index) == 0:
            edge_index = torch.randn(2, 0, device=z.device).long()

        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=z.device).t().contiguous()

        edge_attr = torch.tensor(edge_attr, dtype=torch.long, device=z.device)
        node_embedding = self.decoder.atom_gnn.encoder_node(x, edge_index, edge_attr)  # [n, node_embeding_dim]



        # construct edge select mat, only care about up triangle
        node_num = len(x)
        latent_repeat = torch.repeat_interleave(z, node_num, dim=0).view(node_num, -1)

        #node_embedding = torch.cat([node_embedding, x_pieces_embed, latent_repeat, x_type_embedding,x_inter_embedding], dim=-1)

        node_embedding = torch.cat([node_embedding, x_pieces_embed, latent_repeat], dim=-1)
        node_embedding = self.decoder.atom_embedding_predictor(node_embedding)




        edge_select = torch.triu(torch.ones(node_num, node_num, dtype=torch.long, device=z.device))
        
        for m in range(len(inters)):
            for n in range(len(inters) - 1, m, -1):
                edge_select[m][n] = edge_select[n][m] = 1
        
        # 找到所有piece间的边，看是否有link
        for group in groups:
            group_len = len(group)
            for i in range(group_len):
                for j in range(i, group_len):
                    m, n = group[i], group[j]
                    edge_select[m][n] = edge_select[n][m] = 0  # 如果两个原子在一个piece里，对应edge_select0
        '''
        #print(x_type)
        #print(x_inter)
        edge_select = torch.triu(torch.zeros(node_num, node_num, dtype=torch.long, device=z.device))
        edge_index_list = edge_index.t().tolist()
        for i in range(len(x)):
            for j in range(i, len(x)):
                #atom_i, atom_j = merged_mol.GetAtomWithIdx(i), merged_mol.GetAtomWithIdx(j)
                inter_i, inter_j = x_inter[i], x_inter[j]
                type_i, type_j = x_type[i], x_type[j]
                #print([i, j], inter_i, inter_j,  type_i, type_j)
                if i != j and ([i, j] not in edge_index_list) and not (type_i == type_j == 0):
                    edge_select[i,j] = 1
        '''

        possible_edge_index = torch.nonzero(edge_select)  # piece间的边
        #print(possible_edge_index)
        src, dst = possible_edge_index[:, 0].squeeze(), possible_edge_index[:, 1].squeeze()

        pred_edge = self.decoder.edge_predictor(node_embedding[src] * node_embedding[dst])

        #print(pred_edge)

        '''
        edge_select = edge_select.unsqueeze(0).bool()  # [1, node_num, node_num]
        # predict edge
        node_embedding = node_embedding.view(batch_size, node_num, -1) # [batch_size, N, node_dim] 获得每个原子的embed


        node_embedding = self.edge_predictor(node_embedding)
        edge_pred_in = torch.matmul(node_embedding, node_embedding.view(batch_size, -1, node_num))
        edge_pred_in = edge_pred_in[edge_select]

        node_embedding = node_embedding.unsqueeze(0)  # [1, n, embedding_dim]
        src_embedding = torch.repeat_interleave(node_embedding, node_num, dim=1).view(batch_size, node_num, node_num, -1)
        src_embedding = src_embedding[edge_select]
        dst_embedding = torch.repeat_interleave(node_embedding, node_num, dim=0).view(batch_size, node_num, node_num, -1)
        dst_embedding = dst_embedding[edge_select]
        latent_repeat = torch.repeat_interleave(z, node_num ** 2, dim=0).view(batch_size, node_num, node_num, -1)
        latent_repeat = latent_repeat[edge_select]
        edge_pred_in = torch.cat([src_embedding, dst_embedding, latent_repeat], dim=-1) #根据原子的表示，计算那两个原子可能成为piece间键的可能性
        pred_edge = self.edge_predictor(edge_pred_in) # [159, 1]
        print(pred_edge)

        '''

        pred_edge = pred_edge.reshape(-1, 3)  # 有可能只有一条边

        #print(pred_edge_index.size(), pred_edge.size())

        #Chem.Kekulize(merged_mol)

        merged_mol = Chem.RWMol(merged_mol)
        if pred_edge.shape[0]:  # maybe only one piece is generated -> no edges need to be predicted
            # pred_edge = self.edge_predictor(edge_pred_in) # [E, num_edge_type]  E edge_select为1的边
            #gg = nx.MultiGraph()
            pred_edge = torch.softmax(pred_edge, dim=-1)
            #print(pred_edge)
            # add edge to mol by confidence level
            # pred_edge_index = torch.nonzero(edge_select.squeeze())  # [E, 2]  edge_select 边对应的原子
            #none_bond = self.tokenizer.chem_vocab.bond_to_idx(None)

            confidence, edge_type = torch.max(pred_edge, dim=-1)

            # print(torch.max(pred_edge, dim=-1))

            possible_edge_idx = [i for i in range(len(pred_edge))
                                 if edge_type[i] == 1]
            
            '''
            possible_edge_idx = [i for i in range(len(pred_edge))
                                 if pred_edge[i, 1] >= 0.33]
            '''
            #print(node_num - 1)
            if len(possible_edge_idx) >= len(piece_ids)*2:
                sorted_idx = sorted(possible_edge_idx, key=lambda i: pred_edge[i, 1], reverse=True)[
                             : len(piece_ids)*2]  # 最多
            else:
                sorted_idx = sorted(possible_edge_idx, key=lambda i: pred_edge[i, 1], reverse=True)

            #ed_merged_mol = Chem.EditableMol(merged_mol)
            #print(sorted_idx)
            # original edge generation algorithm: add the edges with higher confidence
            # sorted_idx = sorted(range(len(pred_edge)), key=lambda i: confidence[i], reverse=True)

            '''
            for i in sorted_idx:
                begin, end = pred_edge_index[i]
                # begin, end = pred_edge_index[i]
                begin, end = begin.item(), end.item()
                # bond_type = edge_type[i]
                bond_type = 1  # edge_sets[begin].append(attr)
                # the cycle check is very important (only generate cycles with 5 or 6 atoms)

                if valence_check(atoms[begin], atoms[end], edge_sets[begin],
                                 edge_sets[end], bond_type, self.tokenizer.chem_vocab) and \
                        cycle_check(begin, end, gen_mol):
                
                    gen_mol.AddBond(begin, end, self.tokenizer.chem_vocab.idx_to_bond(bond_type))
                    edge_sets[begin].append(bond_type)
                    edge_sets[end].append(bond_type)
                    # update connected block
                    bid1, bid2 = aid2bid[begin], aid2bid[end]
                    if bid1 != bid2:
                        gid = aid2gid[begin]
                        for aid in aid2bid:  # redirect all atom in block1 to block2
                            if aid2bid[aid] == bid1:
                                aid2bid[aid] = bid2
                        block_atom_cnt[bid2] += block_atom_cnt[bid1]
            '''
            i = 0
            link_pieces = set()
            x_inter_copy = x_inter
            '''
            gg = nx.Graph()
            for i, group in enumerate(groups):
                gg.add_node(i, frag_idx=group)
            '''
            #print(sorted_idx)
            while i < len(sorted_idx):

                begin, end = possible_edge_index[sorted_idx[i]]
                # begin, end = pred_edge_index[i]
                begin, end = begin.item(), end.item()
                #print('atom', begin, end)
                #print( x_pieces[begin], x_pieces[end])
                atom_b, atom_e = merged_mol.GetAtomWithIdx(begin), merged_mol.GetAtomWithIdx(end)
                b_inter, e_inter = x_inter_copy[begin], x_inter_copy[end]
                b_charge, e_charge = atom_b.GetFormalCharge(), atom_e.GetFormalCharge()
                if (b_charge !=0 and b_inter == 0) or (e_charge !=0 and b_inter == 0):
                    continue

                if valence_check(atoms[begin], atoms[end], edge_sets[begin],
                                 edge_sets[end], 1, self.tokenizer.chem_vocab) and \
                        cycle_check(begin, end, merged_mol) :
                    #print(begin, end, 'bonded')
                    merged_mol.AddBond(begin, end, Chem.rdchem.BondType.SINGLE)
                    #gg.add_edge(x_pieces[begin], x_pieces[end])

                    link_pieces.add(x_pieces[begin])
                    link_pieces.add(x_pieces[end])

                    edge_sets[begin].append(Chem.rdchem.BondType.SINGLE)
                    edge_sets[end].append(Chem.rdchem.BondType.SINGLE)
                    x_inter_copy[begin] -= 1
                    x_inter_copy[end] -= 1

                i += 1

            #print(link_pieces, groups)
            retain_atoms = []
            leave_atoms = []
            link_pieces = list(link_pieces)
            for i, group in enumerate(groups):
                if i in link_pieces:
                    retain_atoms = retain_atoms + group
                else:
                    leave_atoms = leave_atoms + group
            #print(groups, retain_atoms)


            merged_mol = get_sub_mol(merged_mol, retain_atoms)


        else:
            merged_mol = get_sub_mol(merged_mol, groups[0])

        try:
            Chem.SanitizeMol(merged_mol)
        except:
            print('Sanitize error')
            merged_mol = None

        return merged_mol

    def inference_constraint(self, z, max_atom_num, add_edge_th, temperature, constraint_mol):
        # z: [latent_dim]
        z = z.unsqueeze(0)  # [1, latent_dim]
        batch_size = z.shape[0]  # 1
        # predict piece
        # [1, 1, embedding_dim]

        # constraint_mol = del_N_positive(constraint_mol)
        Chem.Kekulize(constraint_mol)
        init_pieces, init_groups = self.tokenizer.tokenize(constraint_mol, return_idx=True)
        piece_ids, cur_piece_id = list(init_pieces[1:-1]), None
        cur_atom_num = 0
        for pid in piece_ids:
            cur_atom_num += cnt_atom(self.tokenizer.idx_to_piece(pid))

        # max_atom_num = 1.2 * cur_atom_num

        cur_piece = self.piece_embedding(torch.tensor([[init_pieces[-2]]], dtype=torch.long, device=z.device))
        init_rnn_in = torch.tensor(init_pieces[:-2], device=z.device)  # include start <eos>
        init_embeddings = self.piece_embedding(init_rnn_in).unsqueeze(0)

        init_pieces, init_groups = list(init_pieces[1:-1]), list(init_groups[1:-1])
        init_mol = constraint_mol

        hidden = self.latent_to_rnn_hidden(z).unsqueeze(0)  # unsqueeze for 1 layer of rnn
        _, hidden = self.rnn(init_embeddings, hidden)
        while cur_piece_id != self.tokenizer.end_idx():
            # rnn_in = torch.cat([cur_piece, z.unsqueeze(0)], dim=-1)
            rnn_in = cur_piece
            output, hidden = self.rnn(rnn_in, hidden)
            output = self.to_vocab(output)  # [1, 1, num_piece_type]
            output = output.squeeze()
            output[self.tokenizer.pad_idx()] = float('-inf')  # mask pad
            if len(piece_ids) == 0:
                output[self.tokenizer.end_idx()] = float('-inf')  # at least output one piece
            probs = torch.softmax(output / temperature, dim=-1)  # [num_piece_type]
            cur_piece_id = torch.multinomial(probs, num_samples=1)
            cur_piece = self.piece_embedding(cur_piece_id).unsqueeze(0)
            cur_piece_id = cur_piece_id.item()
            cur_atom_num += cnt_atom(self.tokenizer.idx_to_piece(cur_piece_id))
            piece_ids.append(cur_piece_id)
            if cur_atom_num > max_atom_num:  # the last one will still be dropped
                break
            if len(piece_ids) == self.pos_embedding.num_embeddings:  # 0 is padding, the last one dropped
                break
        piece_ids = piece_ids[:-1]  # get rid of end token

        # graph embedding and start to construct a molecule
        x, edge_index, edge_attr, groups = [], [], [], []
        aid2gid = {}  # map atom idx to group idx
        aid2bid = {}  # map atom idx to connected block (bid)
        block_atom_cnt = []
        gen_mol = Chem.RWMol()  # generated mol
        edge_sets = []  # record each atom is connected to which kinds of bonds
        x_pieces, x_pos = [], []
        # add init mol
        inter_group_edges = []
        init_aid2aid = {}
        for pos, pid in enumerate(piece_ids):
            if pos == len(init_pieces):
                break
            group, atom_num = [], len(init_groups[pos])
            for aid in init_groups[pos]:
                atom = init_mol.GetAtomWithIdx(aid)
                group.append(len(x))
                init_aid2aid[aid] = len(x)
                aid2gid[len(x)], aid2bid[len(x)] = len(groups), len(groups)
                x.append(self.tokenizer.chem_vocab.atom_to_idx(atom.GetSymbol()))
                edge_sets.append([])
                x_pieces.append(pid)
                x_pos.append(pos + 1)  # position starts from 1

                atom_sym = atom.GetSymbol()
                new_atom = Chem.Atom(atom_sym)
                new_atom.SetFormalCharge(atom.GetFormalCharge())
                gen_mol.AddAtom(new_atom)  # add atom to generated mol

            groups.append(group)
            block_atom_cnt.append(atom_num)
            for bond in init_mol.GetBonds():
                begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if begin not in init_aid2aid or end not in init_aid2aid:
                    continue
                begin, end = init_aid2aid[begin], init_aid2aid[end]
                if begin not in group and end not in group:
                    continue
                if begin not in group or end not in group:
                    inter_group_edges.append((begin, end))
                attr = self.tokenizer.chem_vocab.bond_to_idx(bond.GetBondType())
                edge_index.append([begin, end])  # not appear in GNN (no! should appear)
                edge_index.append([end, begin])
                edge_attr.append(attr)
                edge_attr.append(attr)
                edge_sets[begin].append(attr)
                edge_sets[end].append(attr)

                gen_mol.AddBond(begin, end, bond.GetBondType())  # add bond to generated mol

        for pos, pid in enumerate(piece_ids):
            if pos < len(init_pieces):
                continue
            smi = self.tokenizer.idx_to_piece(pid)
            try:
                mol = smiles2molecule(smi, kekulize=True)
            except Exception:
                print(smi)
            offset = len(x)
            group, atom_num = [], mol.GetNumAtoms()
            for aid in range(atom_num):
                atom = mol.GetAtomWithIdx(aid)
                group.append(len(x))
                aid2gid[len(x)], aid2bid[len(x)] = len(groups), len(groups)
                x.append(self.tokenizer.chem_vocab.atom_to_idx(atom.GetSymbol()))
                edge_sets.append([])
                x_pieces.append(pid)
                x_pos.append(pos + 1)  # position starts from 1

                gen_mol.AddAtom(Chem.Atom(atom.GetSymbol()))  # add atom to generated mol

            groups.append(group)
            block_atom_cnt.append(atom_num)
            for bond in mol.GetBonds():
                begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                attr = self.tokenizer.chem_vocab.bond_to_idx(bond.GetBondType())
                begin, end = begin + offset, end + offset
                edge_index.append([begin, end])
                edge_index.append([end, begin])
                edge_attr.append(attr)
                edge_attr.append(attr)
                edge_sets[begin].append(attr)
                edge_sets[end].append(attr)

                gen_mol.AddBond(begin, end, bond.GetBondType())  # add bond to generated mol

        atoms, edges, edge_types = x, edge_index, edge_attr
        node_x = self.embed_atom(torch.tensor(x, dtype=torch.long, device=z.device),
                                 torch.tensor(x_pieces, dtype=torch.long, device=z.device),
                                 torch.tensor(x_pos, dtype=torch.long, device=z.device))
        if len(edge_index) == 0:
            edge_index = torch.randn(2, 0, device=z.device).long()
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=z.device).t().contiguous()
        edge_attr = F.one_hot(torch.tensor(edge_attr, dtype=torch.long, device=z.device),
                              num_classes=self.tokenizer.chem_vocab.num_bond_type())
        node_embedding, _ = self.graph_embedding.embed_node(node_x, edge_index, edge_attr)  # [n, node_embeding_dim]
        # construct edge select mat, only care about up triangle
        node_num = len(x)
        edge_select = torch.triu(torch.ones(node_num, node_num, dtype=torch.long, device=z.device))
        for group in groups:
            group_len = len(group)
            for i in range(group_len):
                for j in range(i, group_len):
                    m, n = group[i], group[j]
                    edge_select[m][n] = edge_select[n][m] = 0
        # inter-group (added by init mol)
        for begin, end in inter_group_edges:
            edge_select[begin][end] = edge_select[end][begin] = 0
            bid1, bid2 = aid2bid[begin], aid2bid[end]  # revise connect block
            if bid1 != bid2:
                gid = aid2gid[begin]
                for aid in aid2bid:  # redirect all atom in block1 to block2
                    if aid2bid[aid] == bid1:
                        aid2bid[aid] = bid2
                block_atom_cnt[bid2] += block_atom_cnt[bid1]
        edge_select = edge_select.unsqueeze(0).bool()  # [1, node_num, node_num]
        # predict edge
        node_embedding = node_embedding.unsqueeze(0)  # [1, n, embedding_dim]
        src_embedding = torch.repeat_interleave(node_embedding, node_num, dim=1).view(batch_size, node_num, node_num,
                                                                                      -1)
        src_embedding = src_embedding[edge_select]
        dst_embedding = torch.repeat_interleave(node_embedding, node_num, dim=0).view(batch_size, node_num, node_num,
                                                                                      -1)
        dst_embedding = dst_embedding[edge_select]
        latent_repeat = torch.repeat_interleave(z, node_num ** 2, dim=0).view(batch_size, node_num, node_num, -1)
        latent_repeat = latent_repeat[edge_select]
        edge_pred_in = torch.cat([src_embedding, dst_embedding, latent_repeat], dim=-1)
        if edge_pred_in.shape[0]:  # maybe only one piece is generated -> no edges need to be predicted
            pred_edge = self.edge_predictor(edge_pred_in)  # [E, num_edge_type]
            pred_edge = torch.softmax(pred_edge, dim=-1)
            # add edge to mol by confidence level
            pred_edge_index = torch.nonzero(edge_select.squeeze())  # [E, 2]
            none_bond = self.tokenizer.chem_vocab.bond_to_idx(None)
            confidence, edge_type = torch.max(pred_edge, dim=-1)  # [E], [E]
            possible_edge_idx = [i for i in range(len(pred_edge))
                                 if confidence[i] >= add_edge_th and edge_type[i] != none_bond]
            sorted_idx = sorted(possible_edge_idx, key=lambda i: confidence[i], reverse=True)

            # original edge generation algorithm: add the edges with higher confidence
            for i in sorted_idx:
                begin, end = pred_edge_index[i]
                begin, end = begin.item(), end.item()
                bond_type = edge_type[i]
                # the cycle check is very important (only generate cycles with 5 or 6 atoms)
                if valence_check(atoms[begin], atoms[end], edge_sets[begin],
                                 edge_sets[end], bond_type, self.tokenizer.chem_vocab,
                                 gen_mol.GetAtomWithIdx(begin).GetFormalCharge(),
                                 gen_mol.GetAtomWithIdx(end).GetFormalCharge()) and \
                        cycle_check(begin, end, gen_mol):
                    gen_mol.AddBond(begin, end, self.tokenizer.chem_vocab.idx_to_bond(bond_type))
                    edge_sets[begin].append(bond_type)
                    edge_sets[end].append(bond_type)
                    # update connected block
                    bid1, bid2 = aid2bid[begin], aid2bid[end]
                    if bid1 != bid2:
                        gid = aid2gid[begin]
                        for aid in aid2bid:  # redirect all atom in block1 to block2
                            if aid2bid[aid] == bid1:
                                aid2bid[aid] = bid2
                        block_atom_cnt[bid2] += block_atom_cnt[bid1]

        # delete isolated parts
        # find connect block with max atom num and must contain the original molecule
        sorted_bids = sorted(range(len(block_atom_cnt)), key=lambda i: block_atom_cnt[i], reverse=True)
        for i in sorted_bids:
            if i == aid2bid[0]:  # root must be in
                bid = i
                break
        # bid = max(range(len(block_atom_cnt)), key=lambda i: block_atom_cnt[i])
        atoms_to_remove = sorted([aid for aid in aid2bid.keys() if aid2bid[aid] != bid], reverse=True)
        for aid in atoms_to_remove:
            gen_mol.RemoveAtom(aid)
        gen_mol = gen_mol.GetMol()
        smis = Chem.MolToSmiles(gen_mol)
        gen_mol = Chem.MolFromSmiles(smis)
        return gen_mol

    def predict_props(self, z):
        props = self.predictor.predict_z(z)

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

    def inference_single_z(self, vocab_data,z, max_atom_num, add_edge_th, temperature):
        # max_atom_num and temperature has no use here. Directly generate a molecule
        return self.inference(vocab_data,z, max_atom_num, add_edge_th, temperature)


    def inference_single_z_constraint(self, z, max_atom_num, add_edge_th, temperature, constraint_mol):
        return self.inference_constraint(z, max_atom_num, add_edge_th, temperature, constraint_mol)



    def get_z_from_batch(self, vocab_data, group_batch):
        #atom_batch, group_batch = graph_batch

        z_vecs,_ = self.encoder(vocab_data, group_batch)


        return z_vecs



    def get_z_from_data(self, vocab_data, data):
        atom_data, group_data = data
        if isinstance(atom_data,list):
            group_data = group_data[0]
        print(group_data.inters)
        print(group_data.pieces)
        z_vecs,_ = self.encoder(vocab_data, group_data)
        z_vecs = z_vecs.squeeze()
        #print(z_vecs.size())

        return z_vecs

    def get_z_from_smis(self, vocab_data, smis):
        group_data_list = [get_graph(self.tokenizer, s)[1] for s in smis]
        group_batch = Batch.from_data_list(group_data_list)

        z_vecs = self.get_z_from_batch(vocab_data, group_batch)



        return z_vecs


