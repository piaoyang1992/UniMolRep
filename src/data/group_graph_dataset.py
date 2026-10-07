#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 2024/7/3 17:10 
# @Author : piaoyang cao 
# @File : group_graph_dataset.py

# !/usr/bin/python
# -*- coding:utf-8 -*-
import os
from tqdm import tqdm

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.data.mol_group_graph import Tokenizer,TreeCodec
#from src.data import chem_utils
# from  src.utils.nn_utils import to_one_hot
import src.evaluation as evaluation
from rdkit import  Chem
import pandas as pd
import numpy as np
import networkx as nx

from collections import Counter
from rdkit.Chem import MACCSkeys

MAX_VALENCE = {'B': 3, 'Br':1, 'C':4, 'Cl':1, 'F':1, 'I':1, 'N':3, 'O':2, 'P':5, 'S':6,
               'As':4, 'Fe':6, 'Sn':4, '#0':4, 'Be':2,'Mg':2, 'Ca':2,'Si':4, 'Se':4, 'Ge':6, 'Ni':6,
               'Cu':6, 'Li':1, 'Na':1,'K':1,'Ru':6, 'Co':6, 'Pt':6, 'Ir':6, 'Pd':6, 'W':6, 'misc':1}


allowable_features = {
    'possible_atomic_symbol_list' : list(MAX_VALENCE.keys()),
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_chirality_list' : [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
        'misc'
    ],
    'possible_hybridization_list' : [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED,
        'misc'
    ],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8,'misc'],
    'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6,'misc'],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,'misc'],
    'possible_bonds' : [
        None,
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        #Chem.rdchem.BondType.AROMATIC,
        'misc'
    ],
    'possible_bond_dirs' : [ # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}


def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1


def atom_features(atom):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms
    atom_feature = [safe_index(allowable_features['possible_atomic_symbol_list'],
        atom.GetSymbol())] + [safe_index(allowable_features[
        'possible_chirality_list'],atom.GetChiralTag())]
    return atom_feature



def onek_encoding_unk(value, choices):
    """
    Creates a one-hot encoding.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the value in a list of length len(choices) + 1.
    If value is not in the list of choices, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding

'''
def atom_features(atom: Chem.rdchem.Atom, functional_groups = None):
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    :return: A list containing the atom features.
    """
    features = onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
           onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
           onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
           onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
           onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
           onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
           [1 if atom.GetIsAromatic() else 0] + \
           [atom.GetMass() * 0.01]  # scaled to about the same range as other features
    if functional_groups is not None:
        features += functional_groups
    return features



def bond_features(bond: Chem.rdchem.Bond):
    """
    Builds a feature vector for a bond.

    :param bond: A RDKit bond.
    :return: A list containing the bond features.
    """
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond
'''

def bond_features(bond: Chem.rdchem.Bond):
    edge_feature = [safe_index(allowable_features['possible_bonds'],
                               bond.GetBondType())] + [safe_index(allowable_features[
                                                                      'possible_bond_dirs'],
                                                                  bond.GetBondDir())]
    return edge_feature

class MolDataset(Dataset):
    def __init__(self, fname, tokenizer):
        super(MolDataset, self).__init__()
        self.root_path, self.file_path = os.path.split(fname)
        self.save_path = os.path.join(self.root_path, 'processed_group_data.pkl')
        self.tokenizer = tokenizer
        try:
            self.data = torch.load(self.save_path)
        except FileNotFoundError:
            self.data = self.process()


    @staticmethod
    def process_step1(s, tokenizer):
        '''to raw representation (save storage space)'''
        # add nodes
        #print(smiles)
        pieces, groups, gg = tokenizer(s, return_idx=True)
        mol = gg.graph['mol']


        #x = [tokenizer.chem_vocab.atom_to_idx(mol.GetAtomWithIdx(i).GetSymbol())
             #for i in range(mol.GetNumAtoms())]  # [num_nodes]
        #AllChem.ComputeGasteigerCharges(mol)
        x = []
        for i in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(i)
            atom_f = atom_features(atom)
            #atom_f = atom_f + [atom.GetDoubleProp('_GasteigerCharge')]
            x.append(atom_f)
        # edge index and edge attr

        edge_index, edge_attr, mask_edge = [], [], []
        for bond in mol.GetBonds():
            begin, end = bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()
            attr = tokenizer.chem_vocab.bond_to_idx(bond.GetBondType())
            edge_index.append([begin, end])  # do not repeat for storage
            '''
            if tuple(sorted([begin, end])) in inter_pairs:
                mask_edge.append(0)
            else:
                mask_edge.append(1)
            '''
            edge_attr.append(attr)
        # add property scores
        properties = evaluation.utils.get_normalized_property_scores(mol)  # qed, sa, logD
        # piece breakdown
        #pieces, groups, data = tokenizer(s, return_idx=True)  # 基团的序号 基团的原子编号
        return {
            'x': x,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'props': properties,
            'pieces': pieces,
            'groups': groups,
            'group_graph': gg,
            #'group_data':data
        }

    @staticmethod
    def process_step2(data, tokenizer):
        '''to adjacent matrix representation'''
        x, edge_index, edge_attr = data['x'], data['edge_index'], data['edge_attr']

        ad_mat = [[tokenizer.chem_vocab.bond_to_idx(None) for _ in x] for _ in x]
        for i in range(len(edge_attr)):
            begin, end = edge_index[i]
            ad_mat[begin][end] = ad_mat[end][begin] = edge_attr[i]  # 邻接矩阵，只为键的类型

        x_pieces, x_pos, pieces = [0 for _ in x], [0 for _ in x], data[
            'pieces']  # x_piece 原子序号位置对应的piece编号，原子序号对应的piece 序号
        #edge_select = [[1 for _ in x] for _ in x]  # n*n  # 基团间的原子连接情况
        #print(x)
        #print(data['groups'])
        gg = data['group_graph']
        inter_pairs = gg.graph['inter_pair']
        inters = [j for i in inter_pairs for j in i]
        inters = list(sorted(set(inters)))

        edge_select = [[0 for _ in x] for _ in x]
        for m in range(len(inters)):
            for n in range(len(inters) - 1, m, -1):
                edge_select[m][n] = edge_select[n][m] = 1


        for pos, group in enumerate(data['groups']):
            group_len = len(group)  # 基团的原子个数
            for i in range(group_len):
                aid = group[i]
                x_pieces[aid] = pieces[pos]
                x_pos[aid] = i + 1

                '''
                if aid in inters:
                    x_pos[aid] = 2
                else:
                    x_pos[aid] = 1
                '''

                for j in range(i, group_len):
                    m, n = group[i], group[j]
                    edge_select[m][n] = edge_select[n][m] = 0  # self-loop is also excluded


        #group_graph = data['group_data']
        #group_data = Data(x=group_graph['x'],edge_index=group_graph['edge_index'],edge_attr=group_graph['edge_attr'],
                          #inter_tensor=group_graph['inter_tensor'])
        return {
            'x': x,
            'ad_mat': ad_mat,
            'props': data['props'],
            'pieces': pieces,
            'x_pieces': x_pieces,
            'x_pos': x_pos,
            'edge_select': edge_select,
            'group_graph':gg
        }

    @staticmethod
    def process_step3(data_list, tokenizer, device='cpu'):
        '''collate data from step2'''
        # pad atom
        # [batch_size, N]
        xs, lens, x_pieces, x_pos,group_data_list = [], [], [], [], []

        x_pos_1 = []

        pieces = pad_sequence([torch.tensor(data['pieces'], dtype=torch.long, device=device) for data in data_list],
                              batch_first=True, padding_value=tokenizer.pad_idx()) # [b,seq]


        for i,data in enumerate(data_list):
            x = torch.tensor(data['x'], device=device)
            xs.append(x)
            lens.append(len(x))

            x_pieces.append(torch.tensor(data['x_pieces'], device=device))
            #pos = [p + count_pieces for p in data['x_pos']]
            #n_piece = len(data['group_data'].x)
            #count_pieces = count_pieces + len(data['group_data'].x)

            #x_pos.append(torch.tensor(pos, device=device))
            x_pos_1.append(torch.tensor(data['x_pos'], device=device))
            #print(data['group_data'])
            #group_data = data['group_data']
            #group_data_list.append(group_data)

        #group_datas = Batch.from_data_list(group_data_list)

        #xs = pad_sequence(xs, batch_first=True, padding_value=tokenizer.atom_pad_idx())
        xs = pad_sequence(xs, batch_first=True, padding_value=0)

        x_pieces = pad_sequence(x_pieces, batch_first=True, padding_value=tokenizer.pad_idx())
        #x_pos = pad_sequence(x_pos, batch_first=True, padding_value=0)  # 0 is padding for position embedding
        x_pos_1 = pad_sequence(x_pos_1, batch_first=True, padding_value=0)
        atom_mask = torch.zeros(xs.shape[0], xs.shape[1] + 1, dtype=torch.long, device=device)
        atom_mask[torch.arange(xs.shape[0], device=device), lens] = 1
        atom_mask = atom_mask.cumsum(dim=1)[:, :-1]  # remove the superfluous column

        batch_size, node_num = xs.shape[0], xs.shape[1]
        edge_index, edge_attr, golden_edge, props = [], [], [], []
        neg_edge_index = []
        in_piece_edge_idx = []
        edge_select = torch.zeros(batch_size, node_num, node_num, device=device)  # [batch_size, N, N]
        none_bond = tokenizer.chem_vocab.bond_to_idx(None)
        for i, data in enumerate(data_list):
            ad_mat = data['ad_mat']
            offset = node_num * i
            props.append(data['props'])
            gg = data['group_graph']
            atom_cls = gg.graph['atom_cls']
            for m, row in enumerate(data['edge_select']):
                for n, val in enumerate(row):  # this is ad mat, 0-1 and 1-0 will both be added
                    edge_select[i][m][n] = val  # 这两个原子是否连接piece 1,0
                      # 键类型编号
                    cls_m, cls_n = atom_cls[m],atom_cls[n]
                    attr = ad_mat[m][n]
                    begin, end = m + offset, n + offset
                    if attr != none_bond:
                        edge_index.append([begin, end])
                        '''
                        if tuple(sorted([m, n])) in inter_pairs:
                            edge_mask.append(0)
                        else:
                            edge_mask.append(1)
                        '''

                        # edge_index.append([end, begin])
                        # edge_attr.append(attr)
                        edge_attr.append(attr)

                        if val == 0:  # to select in-piece bonds for decoder
                            in_piece_edge_idx.append(len(edge_index) - 1)  # 记录那一条边是piece的内部边
                            # in_piece_edge_idx.append(len(edge_index) - 2)
                            #in_piece_edge_idx.append(1)  # 边在piece内 存在
                        #else:
                            #in_piece_edge_idx.append(2)  # 边在piece间 存在

                    if val == 1: # piece 间
                        '''
                        edge_index.append([begin, end])
                        edge_attr.append(attr)
                        '''

                        #in_piece_edge_idx.append(len(edge_index) - 1)
                        if attr == none_bond:
                            golden_edge.append([1,0])
                            if cls_m in gg[cls_n]:
                                neg_edge_index.append([begin, end, 1])
                            else:
                                neg_edge_index.append([begin, end, 2])
                            #in_piece_edge_idx.append(0) # 边piece间 不存在
                        else:
                            golden_edge.append([0,1])
                            #in_piece_edge_idx.append(len(edge_index) - 1) #标记内部之间真正有边的
                            #in_piece_edge_idx.append(2) # 边piece 间 存在



        edge_attr = torch.tensor(edge_attr, dtype=torch.long, device=device)
        if len(edge_index):
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t().contiguous()  # [E, 2]
            neg_edge_index = torch.tensor(neg_edge_index, dtype=torch.long, device=device).t().contiguous()

        else:
            edge_index = torch.Tensor(2, 0, device=device).long()
            neg_edge_index = torch.Tensor(2, 0, device=device).long()

        return {'x': xs,  # [batch_size, N]
                'x_pieces': x_pieces,  # [batch_size, N]
                'x_pos': x_pos_1,  # [batch_size, N]
                'atom_mask': atom_mask.bool(),  # [batch_size, N], mask paddings
                'pieces': pieces,  # [batch_size, seq_len]
                'edge_index': edge_index,
                #'edge_attr': F.one_hot(edge_attr, num_classes=tokenizer.chem_vocab.num_bond_type()),  # [E, edge_dim]
                'edge_attr': edge_attr,
                'edge_select': edge_select.bool(),  # [batch_size, N, N]
                'golden_edge': torch.tensor(golden_edge, dtype=torch.long),  # [E]
                'in_piece_edge_idx': torch.tensor(in_piece_edge_idx, dtype=torch.long),  # [E']
                'props': torch.tensor(props),  # [batch_size, num_props]
                'neg_edge_index': neg_edge_index,
                #'group_x':group_datas.x,
                #'group_batch':group_datas.batch,
                #'group_edge_index': group_datas.edge_index,
                #'group_edge_attr': group_datas.edge_attr,
                #'group_inter_tensor': group_datas.inter_tensor

                }


    def process(self):
          # load smiles
        file_path = os.path.join(self.root_path, self.file_path)
        with open(file_path, 'r') as fin:
            lines = fin.readlines()
        smiles = [s.strip('\n') for s in lines]
          # turn smiles into data type of PyG
        data_list = []
        for s in tqdm(smiles):
            mol = chem_utils.smiles2molecule(s, kekulize=True)
            if mol != None:
                  data_list.append(self.process_step1(s, self.tokenizer))
            else:
                print(s,'santize error')
        torch.save(data_list, self.save_path)
        return data_list

    def __getitem__(self, idx):
        data = self.data[idx]
        return self.process_step2(data, self.tokenizer)

    def __len__(self):
       return len(self.data)

    def collate_fn(self, data_list):
       return self.process_step3(data_list, self.tokenizer)


def get_graph(tokenizer,s):
    gg = tokenizer(s)
    gg_data, gg = gg_to_data(gg, tokenizer)
    # print(pieces, groups)

    label = gg.graph['label']
    #print(label)
    #label = None

    mol = gg.graph['mol']
    #print('gg',label)
    inter_pairs = gg.graph['inter_pair']


    atom_cls = gg.graph['atom_cls']
    if len(list(gg)) == 1:
        atom_cls = [0] * len(atom_cls)


    Chem.Kekulize(mol, clearAromaticFlags=True)  #原子图是从原子表示中预测到原子之间的连接，芳香标签可能在连接过程产生错误，所以需要去掉
    # x = [tokenizer.chem_vocab.atom_to_idx(mol.GetAtomWithIdx(i).GetSymbol())
    # for i in range(mol.GetNumAtoms())]  # [num_nodes]
    # AllChem.ComputeGasteigerCharges(mol)

    x = []
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        atom_f = atom_features(atom)
        # atom_f = atom_f + [atom.GetDoubleProp('_GasteigerCharge')]
        x.append(atom_f)

    #x = [tokenizer.chem_vocab.atom_to_idx(mol.GetAtomWithIdx(i).GetSymbol()) for i in range(mol.GetNumAtoms())]


    x_pieces, x_pos, x_type = [0 for _ in range(len(x))], [0 for _ in range(len(x))], [0 for _ in range(len(x))]

    inter_atoms = set()
    inter_atoms_list = []
    for (a,b) in inter_pairs:
        inter_atoms_list.append(a)
        inter_atoms_list.append(b)
        inter_atoms.add(a)
        inter_atoms.add(b)

    inter_c = Counter(inter_atoms_list)
    sort_id = gg.graph['sort_id']
    for n, attr in gg.nodes(data=True):
        group = attr['frag_idx']
        sub_smiles = attr['sub_smiles']
        frag_id = tokenizer.piece2idx[sub_smiles]
        group_len = len(group)
        for j in range(group_len):
            aid = group[j]
            x_pieces[aid] = sort_id.index(n)
            x_type[aid] = tokenizer.id2type[frag_id]
            if aid in inter_c:
                x_pos[aid] = inter_c[aid]
            else:
                x_pos[aid] = 0


    edge_index, edge_attr, mask_edge = [], [], []

    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()
        attr = bond_features(bond)
        edge_attr.append(attr)
        edge_index.append(sorted([begin, end]))  # do not repeat for storage
        if tuple(sorted([begin, end])) in inter_pairs:
            mask_edge.append(0) # 要mask的边
        else:
            mask_edge.append(1)  # 非piece-link

    if len(edge_index) == 0:
        edge_index = [[0,0]]
        edge_attr = [[0,0]]
        mask_edge.append(1)

    neg_edge_index = []


    inter_atoms = list(inter_atoms)



    #print(x)
    neg_edge_index_1 = []
    neg_edge_index_2 = []
    for a in range(len(x)):
        for b in range(len(x)-1, a, -1):
            cls_i, cls_j = atom_cls[a], atom_cls[b]
            type_a, type_b = x_type[a], x_type[b]
            #print(cls_i, list(gg.neighbors(cls_i)))
            if [a, b] not in edge_index and not (type_a == type_b == 0):
                if  (a in inter_atoms) and (b in inter_atoms):
                    neg_edge_index_1.append([a, b])
                else:
                    neg_edge_index_2.append([a, b])


      # qed, sa, logD

    pieces_pad = [tokenizer.pad_idx()] * 100

    pieces_select = [0] * 100

    pieces_pad[:len(pieces_id) + 2] = [tokenizer.piece2idx['<s>']] + pieces_id + [tokenizer.piece2idx['<s>']]
    pieces_select[:len(pieces_id) + 2] = [0] + [1] * len(pieces_id) + [0]
    gg_data.pieces = torch.tensor(pieces_pad, dtype=torch.long)
    gg_data.pieces_select = torch.tensor(pieces_select, dtype=torch.long)


    x = torch.tensor(x, dtype=torch.long)
    x_pieces = torch.tensor(x_pieces, dtype=torch.long)
    x_type = torch.tensor(x_type, dtype=torch.long)
    x_pos = torch.tensor(x_pos, dtype=torch.long)
    edge_index = torch.tensor(edge_index, dtype=torch.long).T


    neg_edge_index_1 = torch.tensor(neg_edge_index_1, dtype=torch.long).T
    neg_edge_index_2 = torch.tensor(neg_edge_index_2, dtype=torch.long).T

    mask_edge = torch.tensor(mask_edge, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.long)

    #print(label)
    if type(label) == list and len(label) != 0:
        properties = torch.tensor(label, dtype=torch.float)
    elif type(label) == list and len(label) == 0:
        properties = evaluation.utils.get_normalized_property_scores(mol)
        properties = torch.tensor(properties, dtype=torch.float)
    else:
        properties = evaluation.utils.get_normalized_property_scores(mol)
        properties = torch.tensor(properties, dtype=torch.float)

    #print(properties)
    #print(gg.nodes(data=True))
    #print(x_pieces)
    #print(gg_data.x)
    atom_data = Data(x=x, x_type=x_type,x_piece=x_pieces, x_pos=x_pos, edge_index=edge_index, neg_edge_index_1=neg_edge_index_1,
                     neg_edge_index_2=neg_edge_index_2,
                     mask_edge=mask_edge, edge_attr=edge_attr,
                     prop=properties)
                     #prop=torch.tensor(properties, dtype=torch.float))

    return atom_data, gg_data




def gg_to_data(g, tokenizer):
    res = []
    for i, n in g.nodes(data=True):
        sub_smiles = n['sub_smiles']
        frag_id = tokenizer.piece2idx[sub_smiles]
        frag_type = tokenizer.id2type[frag_id]
        inter_num = tokenizer.id2inter[frag_id]
        res.append([frag_id, frag_type, inter_num, i])

    res = sorted(res, key=lambda i: i[0], reverse=True)
    frag_id_list, frag_type_list, inter_num_list, sorted_id = zip(*res)  # 对frag_id 按照vocab_id 的大小排序
    g.graph['sort_id'] = sorted_id
    edge_attr = []
    edge_index = []
    edge_index_set = set()
    inter_tensor = []

    if len(g.edges(data=True)) == 0:
        edge_index.append([0, 0])
        edge_index_set.add((0, 0))
        edge_attr.append([0, 0, 0, 0])
        inter_tensor.append([[0] * 133, [0] * 133])

    else:
        for u, v, attr in g.edges(data=True):
            index_u, index_v = sorted_id.index(u), sorted_id.index(v)
            if index_u < index_v:
                edge_index.append([index_u, index_v])
                edge_index_set.add((index_u, index_v))
            else:
                edge_index.append([index_v, index_u])
                edge_index_set.add((index_v, index_u))
            edge_attr.append(attr['edge_attr'])
            inter_tensor.append(attr['inter_tensor'])


    neg_edge_index = []
    for i in range(len(frag_id_list)):
        for j in range(i, len(frag_id_list)):
            type_i,  type_j = frag_type_list[i], frag_type_list[j]
            if [i, j] not in edge_index and i != j and not (type_i == type_j == 0):
                neg_edge_index.append([i, j])


    #frags_id = np.array(frag_id_list).astype(int)
    frags_id = torch.tensor(frag_id_list).to(torch.long)
    frags_type = torch.tensor(frag_type_list).to(torch.long)
    inters_num = torch.tensor(inter_num_list).to(torch.long)

    edge_index_set = torch.tensor(list(edge_index_set), dtype=torch.long).T

    edge_index = torch.tensor(edge_index, dtype=torch.long).T
    edge_attr = torch.tensor(edge_attr, dtype=torch.float64)
    neg_edge_index = torch.tensor(neg_edge_index, dtype=torch.long).T

    inter_tensor = torch.tensor(inter_tensor, dtype=torch.float64)


    edge_attr_0 = torch.cat([inter_tensor[:, 0, :], edge_attr[:, 0:2]], dim=1).to(torch.float)
    edge_attr_1 = torch.cat([inter_tensor[:, 1, :], edge_attr[:, 2:4]], dim=1).to(torch.float)
    edge_attr = torch.cat([edge_attr_0, edge_attr_1], dim=1)

    group_data = Data(x=frags_id, frags_type=frags_type, frags_inter=inters_num, edge_index=edge_index, neg_edge_index=neg_edge_index,
                      edge_attr=edge_attr)

    return group_data, g







def acylic_gg_to_data(g, tokenizer):

    frag_id_list = [tokenizer.piece2idx[n['sub_smiles']] for i, n in g.nodes(data=True)]
    atom_num = [len(n['frag_idx']) for i, n in g.nodes(data=True)]
    edge_attr = []
    edge_index = []
    inter_tensor = []

    if len(g.edges()) == 0:
        edge_index.append([0, 0])
        edge_attr.append([0, 0, 0, 0])
        inter_tensor.append([[0] * 133, [0] * 133])

    else:
        for u, v,attr in g.edges(data=True):
            edge_index.append([u, v])
            edge_attr.append(attr['edge_attr'])
            inter_tensor.append(attr['inter_tensor'])

    edge_index = np.array(edge_index).T
    edge_index = edge_index.astype(int)

    edge_attr = np.array(edge_attr).astype(int)

    inter_tensor = np.array(inter_tensor).astype(int)

    frags_id = np.array(frag_id_list).astype(int)
    frags_id = torch.from_numpy(frags_id).to(torch.int64)
    atom_num = torch.tensor(atom_num).to(torch.int64)
    edge_index = torch.from_numpy(edge_index).to(torch.int64)
    edge_attr = torch.from_numpy(edge_attr).to(torch.int64)
    inter_tensor = torch.from_numpy(inter_tensor).to(torch.int64)
    smiles_data = Data(x=frags_id,  edge_index=edge_index, edge_attr=edge_attr, inter_tensor=inter_tensor)

    return smiles_data


def get_sort_g(g,seq):
    tree = TreeCodec()
    decode_nodes, decode_edges = tree.decode_from_sequence(seq)

    new_g = nx.Graph()

    for a, b in decode_nodes.items():
        new_g.add_node(a, **g.nodes[b])

    for x, y in decode_edges:
        attr = g[decode_nodes[x]][decode_nodes[y]]
        new_g.add_edge(x, y, **attr)

    new_g.graph['decode_edge'] = decode_edges
    new_g.graph['seq'] = [g.nodes[n]['sub_smiles'] if isinstance(n, int) else n for n in seq]
    # print(label)
    # label = None

    new_g.graph['mol'] = g.graph['mol']
    # print('gg',label)
    new_g.graph['inter_pair'] = g.graph['inter_pair']
    new_g.graph['vocab_dict'] = g.graph['vocab_dict']
    new_g.graph['smiles'] = g.graph['smiles']
    return new_g


def find_max_list(lists):
    n = len(lists)
    candidates = list(range(n))  # 初始候选列表索引

    # 遍历每个位置进行比较（假设所有列表长度相同）
    for i in range(len(lists[0])):
        # 提取当前候选列表在位置 i 的值
        current_values = [lists[idx][i] for idx in candidates]
        # 找到当前最大值
        max_val = max(current_values)
        # 筛选出值等于最大值的候选列表索引
        new_candidates = [candidates[j] for j, val in enumerate(current_values) if val == max_val]

        # 唯一最大值，直接返回
        if len(new_candidates) == 1:
            return new_candidates[0]
        # 否则更新候选列表，继续下一轮比较
        else:
            candidates = new_candidates

    # 根据题目条件，所有列表元素各不相同，此处不会被执行
    return candidates[0]


def get_acylic_graph(tokenizer,s):
    g = tokenizer(s)
    seq_set = g.graph['seq']

    atom_data_list = []
    group_data_list = []

    element_index_list = []
    inter_list = []
    piece_list = []
    seq_list = list(seq_set)
    for seq in seq_list:
        #print(sort_g.graph['seq'])
        seq = [g.nodes[node]['sub_smiles'] if isinstance(node, int) else node for node in seq]
        inter_l = []
        piece_l = []
        element_index = []
        for token in seq:
            inter_num, piece_id = tokenizer.get_smiles_location(token)
            # print(s,inter_num, piece_id)
            if None not in [inter_num, piece_id]:
                inter_l.append(inter_num)
                piece_l.append(piece_id)
                element_index.append(piece_id * len(tokenizer.sort_df.columns) + inter_num)
            else:
                print(token, 'smiles location error')

        inter_list.append(inter_l)
        piece_list.append(piece_l)
        element_index_list.append(element_index)

    max_id = find_max_list(element_index_list)
    for id in [max_id]:
        inter_max_list = inter_list[id]
        piece_max_list = piece_list[id]
        seq = seq_list[id]

        #需要按照seq的顺序遍历生成边
        sort_g = get_sort_g(g, seq)


        gg_data = acylic_gg_to_data(sort_g, tokenizer)

        inters_pad = [tokenizer.inter_pad_idx()] * 100
        inter_start_token = tokenizer.get_smiles_location('<inter_start>')
        inters_pad[:len(inter_max_list) + 2] = [inter_start_token] + inter_max_list + [inter_start_token]

        pieces_pad = [tokenizer.piece_pad_idx()] * 100
        piece_start_token = tokenizer.get_smiles_location('<piece_start>')
        pieces_pad[:len(piece_max_list) + 2] = [piece_start_token] + piece_max_list + [piece_start_token]


        gg_data.pieces = torch.tensor(pieces_pad, dtype=torch.long)
        gg_data.inters = torch.tensor(inters_pad, dtype=torch.long)


        try:
            label = sort_g.graph['label']
        except:

            label = None

        mol = sort_g.graph['mol']
        #print('gg',label)
        inter_pairs = sort_g.graph['inter_pair']

        '''
        atom_cls = gg.graph['atom_cls']
        if len(list(gg)) == 1:
            atom_cls = [0] * len(atom_cls)
        '''

        Chem.Kekulize(mol, clearAromaticFlags=True)  #原子图是从原子表示中预测到原子之间的连接，芳香标签可能在连接过程产生错误，所以需要去掉
        # x = [tokenizer.chem_vocab.atom_to_idx(mol.GetAtomWithIdx(i).GetSymbol())
        # for i in range(mol.GetNumAtoms())]  # [num_nodes]
        # AllChem.ComputeGasteigerCharges(mol)

        x = []
        for i in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(i)
            atom_f = atom_features(atom)
            # atom_f = atom_f + [atom.GetDoubleProp('_GasteigerCharge')]
            x.append(atom_f)

        x_pieces = [0 for _ in range(len(x))]

        for n, attr in sort_g.nodes(data=True):
            group = attr['frag_idx']
            group_len = len(group)
            for j in range(group_len):
                aid = group[j]
                x_pieces[aid] = n

        edge_index, edge_attr, mask_edge = [], [], []
        for bond in mol.GetBonds():
            begin, end = bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()
            attr = bond_features(bond)
            edge_attr.append(attr)
            edge_index.append(sorted([begin, end]))  # do not repeat for storage
            if tuple(sorted([begin, end])) in inter_pairs:
                mask_edge.append(0) # 要mask的边
            else:
                mask_edge.append(1)  # 非piece-link


        x = torch.tensor(x, dtype=torch.long)
        x_pieces = torch.tensor(x_pieces, dtype=torch.long)

        edge_index = torch.tensor(edge_index, dtype=torch.long).T
        mask_edge = torch.tensor(mask_edge, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.long)

        if len(edge_index) == 0:
            edge_index = [[0,0]]
            edge_attr = [[0,0]]
            mask_edge.append(1)

        inter_atoms = set()
        inter_atoms_list = []
        for (a,b) in inter_pairs:
            inter_atoms_list.append(a)
            inter_atoms_list.append(b)
            inter_atoms.add(a)
            inter_atoms.add(b)

        inter_atoms_c = Counter(inter_atoms_list)

        atom_index = []
        true_atom_index = []

        # 边的产生顺序
        piece_e_list = sort_g.graph['decode_edge']

        piece_edge_id = []
        #cur_frag_idx = sort_g.nodes[piece_e_list[0][0]]['frag_idx']
        #cur_fp_list = []
        cur_mol = Chem.RWMol(mol)
        for a,b in inter_pairs:
            cur_mol.RemoveBond(a, b)

        cur_mol.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(cur_mol)
        fp_list = []
        #print(piece_e_list, gg_data.edge_index)
        inters = inter_atoms.copy()
        for id, (cls_i, cls_j) in enumerate(piece_e_list):
            #print(Chem.MolToSmiles(cur_mol.GetMol()))
            cur_fp = list(MACCSkeys.GenMACCSKeys(cur_mol.GetMol()))
            #print(id, cur_fp)
            atoms_i, atoms_j = sort_g.nodes[cls_i]['frag_idx'], sort_g.nodes[cls_j]['frag_idx']
            inter_i = inters.intersection(set(atoms_i))
            inter_j = inters.intersection(set(atoms_j))
            inter_pair = ()
            # 两个group的连接原子任意连接的连接概率
            for a_i, a in enumerate(inter_i):
                for a_j, b in enumerate(inter_j):
                    e = sorted([a, b])
                    atom_index.append(e)
                    fp_list.append(cur_fp)
                    piece_edge_id.append(id)
                    # a原子已经的连接位置都被连接， b原子还有连接位置
                    if inter_atoms_c[a] == 0 and inter_atoms_c[b] > 0:
                        true_atom_index.append(0)
                        '''
                        if a in inter_atoms:
                            inter_atoms.remove(a)
                        '''
                    # a原子和 b原子还有连接位置
                    elif inter_atoms_c[b] and inter_atoms_c[a]:
                        if tuple(e) in inter_pairs:
                            inter_pair = tuple(e)
                            true_atom_index.append(1)
                            inter_atoms_c[e[0]] -= 1
                            inter_atoms_c[e[1]] -= 1
                        else:
                            true_atom_index.append(2)

                    elif inter_atoms_c[b] == 0 and inter_atoms_c[b] >0:
                        true_atom_index.append(0)
                        '''
                        if b in inter_atoms:
                            inter_atoms.remove(b)
                        '''
                    # a原子和 b原子都没有连接位置
                    else:
                        true_atom_index.append(3)


            #cur_frag_idx = list(cur_frag_idx) + list(atoms_i) + list(atoms_j)
            #cur_frag_idx = set(cur_frag_idx)
            cur_mol.AddBond(inter_pair[0], inter_pair[1], Chem.BondType.SINGLE)
            cur_mol.UpdatePropertyCache(strict=False)
            Chem.GetSymmSSSR(cur_mol)


        # poss_atom_index len(inter_pairs) * 500 * 2
        atom_index = torch.tensor(atom_index, dtype=torch.long).T


        # atom_index  len(inter_pairs) * 500
        #print(true_atom_index)
        true_atom_id = torch.tensor(true_atom_index, dtype=torch.long).reshape(-1)

        # atom_pad len(inter_pairs) * 500
        piece_edge_id = torch.tensor(piece_edge_id, dtype=torch.long)
        piece_fp = torch.tensor(fp_list, dtype=torch.float)
        piece_fp = torch.nan_to_num(piece_fp, nan=0.0)

        # qed, sa, logD
        #print(label)
        if type(label) == list and len(label) != 0:
            properties = torch.tensor(label, dtype=torch.float)
        elif type(label) == list and len(label) == 0:
            properties = evaluation.utils.get_normalized_property_scores(mol)
            properties = torch.tensor(properties, dtype=torch.float)
        else:
            properties = evaluation.utils.get_normalized_property_scores(mol)
            properties = torch.tensor(properties, dtype=torch.float)

        #print(poss_atom_index.size(), edge_index.size(),atom_index.size(), atom_pad.size())
        #print(gg.nodes(data=True))
        #print(x_pieces)
        #print(gg_data.x)


        atom_data = Data(x=x,x_piece=x_pieces, edge_index=edge_index, mask_edge=mask_edge, edge_attr=edge_attr,
                         atom_index=atom_index, link_fp=piece_fp,
                         true_atom_id=true_atom_id, piece_edge_id=piece_edge_id,
                         prop=properties)
                         #prop=torch.tensor(properties, dtype=torch.float))
        atom_data_list.append(atom_data)
        group_data_list.append(gg_data)

    return atom_data_list, group_data_list





class MolDataset1(Dataset):
    def __init__(self, fname, tokenizer):
        self.max_piece = 100
        super(MolDataset1, self).__init__()
        #self.root_path = root_path
        self.root_path, self.file_path = os.path.split(fname)
        self.save_path = os.path.join(self.root_path, 'processed_group_data.pkl')
        self.tokenizer = tokenizer
        try:
            self.data = torch.load(self.save_path)
            print('load group data')
        except FileNotFoundError:
            print('process group graph')
            self.data = self.process()


    def process(self):
        # load smiles
        file_path = os.path.join(self.root_path, self.file_path)
        df = pd.read_csv(file_path,header=0)
        atom_data_list = []
        group_data_list = []
        #label_list = []
        for i in tqdm(range(len(df))):
            s = df.loc[i,'smiles']

            if s != None:
                atom_datas, group_datas = get_acylic_graph(self.tokenizer,s)
                atom_data_list = atom_data_list + atom_datas

                group_data_list = group_data_list + group_datas

                #label_list.append(l)
            else:
                print(s,'santize error')
        #label_list = torch.tensor(label_list, dtype=torch.float)
        torch.save((atom_data_list, group_data_list), self.save_path)
        return atom_data_list, group_data_list

    def __getitem__(self, idx):
        atom_data_list, group_data_list = self.data[0], self.data[1]
        atom_data, group_data = atom_data_list[idx],group_data_list[idx]
        #data = self.data[idx]
        return atom_data, group_data

    def __len__(self):
        return len(self.data[0])




def get_dataloader(fname, tokenizer, batch_size, shuffle=False, num_workers=0):
    dataset = MolDataset1(fname, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def piece_combine(tokenizer,piece_ids):
    x, edge_index, edge_attr = [], [], []
    groups = [[]] * len(piece_ids)
    aid2gid = {}  # map atom idx to group idx
    aid2bid = {}  # map atom idx to connected block (bid)
    block_atom_cnt = []
    # gen_mol = Chem.RWMol()  # generated mol
    edge_sets = []  # record each atom is connected to which kinds of bonds
    x_pieces, x_pos = [], []
    inters = []
    atoms = []
    merged_mol = Chem.RWMol()
    for pos, pid in enumerate(piece_ids):
        mol_i = tokenizer.idx_to_mol(pid)
        Chem.Kekulize(mol_i, clearAromaticFlags=True)
        for atom in mol_i.GetAtoms():
            atom.SetUnsignedProp('piece_id', pos)
            atom.SetUnsignedProp('atom_id', atom.GetIdx())
        merged_mol = Chem.CombineMols(merged_mol, mol_i)

    for i in range(merged_mol.GetNumAtoms()):
        atom = merged_mol.GetAtomWithIdx(i)
        atoms.append(tokenizer.chem_vocab.atom_to_idx(atom.GetSymbol()))
        x.append(atom_features(atom))
        edge_sets.append([])
        piece_id = int(atom.GetProp('piece_id'))
        groups[piece_id] = groups[piece_id] + [i]
        x_pieces.append(piece_id)
        x_pos.append(int(atom.GetProp('atom_id')))

    for bond in merged_mol.GetBonds():
        begin, end = bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()
        edge_attr.append(bond_features(bond))
        edge_index.append(sorted([begin, end]))  # do not repeat for storage
        edge_sets[begin].append(bond.GetBondType())
        edge_sets[end].append(bond.GetBondType())

    print(x_pieces)
    print(groups)




if __name__ == '__main__':
    #import sys

      # dataset = BPEMolDataset(sys.argv[1], sys.argv[2])
      # data = dataset[0]
      # for key in data.keys:
      #     print(key, data[key])
    #mol = chem_utils.smiles2molecule('N#CC1=C[C@H]2CO[C@@H]1O2CCOC([NH-])C#CCO', kekulize=True)

    path = '/home/pycao/PS-VAE-main-1/data/zinc250k'
    f_train = '/home/pycao/PS-VAE-main-1/data/zinc250k/train.txt'
    '''
    path = '/home/pycao/PS-VAE-main-1/data/finetune/class_task/bbbp'
    f_train = '/home/pycao/PS-VAE-main-1/data/finetune/class_task/bbbp/bbbp.csv'
    
    path = '/work/home/zhy_hzau/cpy/PS-VAE-main-1/data/zinc250k/'
    f_train = '/work/home/zhy_hzau/cpy/PS-VAE-main-1/data/zinc250k/train.txt'
    '''

    print(path)
    df = pd.read_csv(f_train, header=0)
    label = list(df.columns).index('smiles') + 1
    label_list = list(range(label, len(df.columns)))

    tokenizer = Tokenizer(f_train, label=None)
    #piece_ids = [11, 6, 6, 6, 21, 3, 7]
    #piece_combine(tokenizer, piece_ids)
    piece_start_token = tokenizer.get_smiles_location('<piece_start>')
    print('piece_start_token',piece_start_token)  # 7382
    inter_start_token = tokenizer.get_smiles_location('<inter_start>')
    print('inter_start_token',inter_start_token)  # 11
    print('piece_pad_token', tokenizer.piece_pad_idx(), 'inter_pad_token', tokenizer.inter_pad_idx())
    s = 'Cc1ccc([C@@H](C)[NH2+][C@H](C)C(=O)Nc2ccccc2F)cc1'

    g = get_acylic_graph(tokenizer,s)
    print(g[0][0],g[1][0])



    dataset = MolDataset1(f_train, tokenizer)
    atom_data,gg_data = dataset[0]

    
    batch_size = 128
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    #print(len(train_loader))
    vocab_tensors = tokenizer.get_vocab_data()
    print(vocab_tensors.size())
    atom_index = []
    all_poss_piece_index = []
    atom_pad = []
    from torch import nn
    edge_loss = nn.CrossEntropyLoss(ignore_index=tokenizer.piece_pad_idx())

    index_max = []
    mask_tensor = tokenizer.get_mask_tensor().T  # 为mask的部分 inter_num * piece_num

    for b in tqdm(train_loader):


        atom_batch, group_batch = b

        batch_size = atom_batch.num_graphs
        x, x_pieces = atom_batch.x, atom_batch.x_piece
        x_input_1 = vocab_tensors.index_select(index=group_batch.x.squeeze(), dim=0)
        inters = group_batch.inters
        print(inters)
        inters = inters.view(batch_size, -1)

        pieces = group_batch.pieces
        pieces = pieces.view(batch_size, -1)
        gold_piece = pieces[:, 1:].flatten()

        #print(mask_tensor.size())
        #print(inters[:, :-1].reshape(-1))
        #print(pieces[:, :-1].reshape(-1))
        mask_select = mask_tensor.index_select(index=inters[:, 1:].reshape(-1), dim=0)
        #print(mask_select)
        pieces_output = torch.rand(pieces[:, :-1].reshape(-1).size(0), mask_tensor.shape[1])
        #pieces_output = nn.Softmax(pieces_output)

        pieces_output[mask_select==1] = -torch.inf
        #select = (gold_piece != 7384).nonzero(as_tuple=True)[0]
        #print(select)
        #pieces_output = pieces_output[select,:]
        #print(pieces_output)
        #gold_piece = gold_piece[select]
        #print(gold_piece)

        piece_loss = edge_loss(pieces_output, gold_piece)
        print(piece_loss)


        '''
        piece_edge_id = atom_batch.piece_edge_id
        true_atom_index = atom_batch.true_atom_id
        link_logits = torch.rand(piece_edge_id.size())
        piece_link_logits = []
        piece_link_truth = []

        # Temporary groups for current sequence
        current_a_group = [link_logits[0]]
        current_b_group = [true_atom_index[0]]

        # Iterate over the list to group adjacent elements
        for i in range(1, len(piece_edge_id)):
            if piece_edge_id[i] == piece_edge_id[i - 1]:  # If the current element matches the previous
                current_a_group.append(link_logits[i])
                current_b_group.append(true_atom_index[i])
            else:  # Otherwise, finalize the current group and start a new one
                piece_link_logits.append(current_a_group)
                piece_link_truth.append(current_b_group)
                current_a_group = [link_logits[i]]
                current_b_group = [true_atom_index[i]]

        # Add the last group
        piece_link_logits.append(current_a_group)
        piece_link_truth.append(current_b_group)

        # Output the results
        #print(piece_link_logits)
        max_index = [len(sequence) for sequence in piece_link_logits]
        padded_pred = [F.pad(torch.tensor(sequence), (0, 800- len(sequence)),value=-torch.inf) for sequence in piece_link_logits]
        padded_pred = torch.stack(padded_pred, dim=0)


        piece_link_truth = [p.index(torch.tensor(1)) for p in piece_link_truth]
        index_max.append(max(max_index))
        #print(max(piece_link_truth))
        piece_link_truth = torch.tensor(piece_link_truth)
        #print(edge_loss(padded_pred, piece_link_truth))
        '''
    print(max(index_max))

