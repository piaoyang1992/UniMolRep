#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/10/14 10:53
# @Author : piaoyang cao
# @File : prepare_mol_data.py


import os
import pandas as pd
import numpy
import numpy as np
import torch
import random

from src.data.molecule_frags_no_dot import *
# 处理每一个mol 形成大图
import networkx as nx


import scipy.sparse as sp


from tqdm import tqdm
from multiprocessing import Pool

import time
import warnings
warnings.filterwarnings('ignore')



def trans_adj_coo(adj):
    adj = sp.coo_matrix(adj)
    indices = np.vstack((adj.row, adj.col))  # 我们真正需要的coo形式  # PyG框架需要的coo形式
    return indices

MAX_ATOMIC_NUM = 100
ATOM_FEATURES = {
    'atomic_num': list(range(MAX_ATOMIC_NUM)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

# Distance feature sizes
PATH_DISTANCE_BINS = list(range(10))
THREE_D_DISTANCE_MAX = 20
THREE_D_DISTANCE_STEP = 1
THREE_D_DISTANCE_BINS = list(range(0, THREE_D_DISTANCE_MAX + 1, THREE_D_DISTANCE_STEP))

# len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
BOND_FDIM = 14


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




class MolSplit(object):
    # 元素及其最大成键电子数
    def __init__(self, smiles):

        self.mol = get_mol(smiles)
        self.smiles = get_smiles(self.mol)
        #self.smiles = smiles
        #print(self.mol)
        # print(self.smiles)
        # self.mol_graph = self.build_mol_graph()
        self.cluster, self.atom_cls = find_clusters(self.mol)
        # print(self.cluster, self.atom_cls)
        self.inter_pair = get_inter(self.mol)
        self.max_inter_num = 4

        #self.vocab_mol_list = Vocab.get_vocab_mol()
    def get_mol_seq(self):
        bond_indices = []  # bond indices of these bonds
        bond_labels = []  # labels are defined so that attachment points belonging to same bond are marked with same label
        for i, b in enumerate(self.inter_pair):
            bond_indices.append(self.mol.GetBondBetweenAtoms(b[0], b[1]).GetIdx())
            bond_labels.append((i+1, i+1))

        if len(bond_indices) > 0:
            Chem.Kekulize(self.mol, clearAromaticFlags=True)
            inter_cut = Chem.FragmentOnBonds(self.mol, bond_indices, dummyLabels=bond_labels)
            #smi_list = Chem.MolToSmiles(inter_cut).split(".")

            inter_idx = Chem.GetMolFrags(inter_cut, asMols=False)
            inter_s = Chem.GetMolFrags(inter_cut, asMols=True)
            inter_s = [Chem.MolToSmiles(i) for i in inter_s]
        else:
            inter_s = [self.smiles]
            inter_idx = [[i for i in range(len(self.atom_cls))]]
        return inter_s, inter_idx



    def get_mol_graph(self):
        g1 = nx.MultiGraph()
        g1.graph['mol'] = self.mol
        g1.graph['smiles'] = self.smiles
        mol = self.mol

        cluster, atom_cls = self.cluster, self.atom_cls
        g1.graph['atom_cls'] = atom_cls
        g1.graph['inter_pair'] = self.inter_pair
        cls = list(cluster.keys())

        inter_pair = self.inter_pair
        Chem.Kekulize(self.mol)
        vocab_dict = {}
        if len(inter_pair) == 0 or len(cls)==1:
            #print(self.smiles,'no inter',cls, get_smiles(self.mol))

            # print( [ i for i in range(len(self.atom_cls))])

            g1.add_node(0, frag_idx=[i for i in range(len(self.atom_cls))], sub_smiles=self.smiles,
                        sub_mol=self.mol)
            if self.smiles not in vocab_dict:
                vocab_dict[self.smiles] = 1
            else:
                vocab_dict[self.smiles] = vocab_dict[self.smiles] + 1
        else:
            for i, c in enumerate(cls):
                #new_c = [idx + 1 for idx in c]
                sub_mol = get_sub_mol(mol,c)
                sub_smiles = get_smiles(sub_mol,isomeric=False)

                if get_mol(sub_smiles) != None:
                    # print(self.smiles, get_smiles(sub_mol))
                    # sub_smiles = get_smiles(get_sub_mol(mol,c))
                    g1.add_node(i, frag_idx=c, sub_smiles=sub_smiles, sub_mol=sub_mol)
                    if sub_smiles not in vocab_dict:
                        vocab_dict[sub_smiles] = 1
                    else:
                        vocab_dict[sub_smiles] = vocab_dict[sub_smiles] + 1
                else:
                    print(self.smiles, sub_smiles, c)
                    print(cls)
                # frags_label[i] = {'sub_smiles': sub_smiles, 'frag_idx': tuple(c), 'sub_mol': sub_mol}


            for x, y in inter_pair:
                cut_bond = mol.GetBondBetweenAtoms(x, y)
                cls_x, cls_y = atom_cls[x], atom_cls[y]
                sub_mol_x, sub_mol_y = g1.nodes[cls_x]['sub_mol'], g1.nodes[cls_y]['sub_mol']
                # sub_smiles_x,sub_smiles_y = frags_label[atom_cls[x]]['sub_smiles'], frags_label[atom_cls[y]]['sub_smiles']
                inter_x = []
                inter_tensor_x = []
                for a in sub_mol_x.GetAtoms():
                    if int(a.GetProp('mol_idx')) == x:
                        inter_tensor_x.append(atom_features(a))
                        inter_x.append(a.GetIdx())
                if len(inter_x) == 0:
                    print(g1.nodes[cls_x]['sub_smiles'], 'do not match vocab_mol_list')
                    inter_x[0] = 0

                inter_y = []
                inter_tensor_y = []
                for a in sub_mol_y.GetAtoms():
                    if int(a.GetProp('mol_idx')) == y:
                        inter_tensor_y.append(atom_features(a))
                        inter_y.append(a.GetIdx())
                if len(inter_y) == 0:
                    print(g1.nodes[cls_x]['sub_smiles'], 'do not match vocab_mol_list')
                    inter_y[0] = 0

                inter_cut = Chem.FragmentOnBonds(mol, [cut_bond.GetIdx()])
                inter_mol = Chem.GetMolFrags(inter_cut, asMols=True)
                if len(inter_mol) < 2:
                    Chem.Kekulize(inter_mol[0], clearAromaticFlags=True)
                    # print(get_smiles(attr['inter_mol'][0]), g.graph['smiles'])
                    inter_smiles = get_smiles(inter_mol[0])
                    if get_mol(inter_smiles) != None:
                        inter_s = inter_smiles
                    else:
                        print(self.smiles, inter_smiles, inter_pair)
                else:
                    Chem.Kekulize(inter_mol[0])
                    Chem.Kekulize(inter_mol[1])
                    # print(get_smiles(attr['inter_mol'][0]), g.graph['smiles'])
                    inter_smiles_0 = get_smiles(inter_mol[0])
                    inter_smiles_1 = get_smiles(inter_mol[1])
                    if get_mol(inter_smiles_0) and get_mol(inter_smiles_1) != None:
                        inter_s = [inter_smiles_0, inter_smiles_1]
                    else:
                        print(self.smiles, inter_smiles_0, inter_smiles_1, inter_pair)


                bond_feat =bond_features(cut_bond)
                if cls_x <= cls_y:
                    edge_attr = [sub_mol_x.GetNumAtoms(), inter_x[0],
                                      sub_mol_y.GetNumAtoms(), inter_y[0]]

                    inter_tensor = [inter_tensor_x[0], inter_tensor_y[0]]
                    g1.add_edge(cls_x, cls_y, edge_attr=edge_attr, inter_tensor=inter_tensor, inter_mol=inter_mol, inter_s=inter_s, bond_feat=bond_feat)

                else:
                    edge_attr = [sub_mol_y.GetNumAtoms(), inter_y[0],
                                      sub_mol_x.GetNumAtoms(), inter_x[0]]
                    inter_tensor = [inter_tensor_y[0], inter_tensor_x[0]]

                    g1.add_edge(cls_y, cls_x, edge_attr=edge_attr, inter_tensor=inter_tensor, inter_mol=inter_mol, inter_s=inter_s, bond_feat=bond_feat)

        g1.graph['vocab_dict'] = vocab_dict
        return g1

'''
s = 'c1cnc2c[nH]nc2c1'
print(s)
ms =MolSplit(s)
g = ms.get_mol_graph()
print(g.graph['vocab_dict'])
print(g.nodes(data=True))
'''

def update_vocab(vocab_dict, df):
    vocab = vocab_dict
    smiles_col = list(df.columns.values).index('smiles')
    # value_list = np.array(df.iloc[:,smiles_col+1:]).tolist()
    g_dict = []
    for i in tqdm(range(len(df))):
        #s_list = df.iloc[i, smiles_col].split('*')
        s_list = [df.iloc[i, smiles_col]]
        for s in s_list:
            if get_mol(s) != None:
                mol_split = MolSplit(s)
                smiles_g = mol_split.get_mol_graph()
                smiles_g.graph['smiles'] = s
                smiles_g.graph['label'] = df.iloc[i, smiles_col + 1:].values
                #smiles_g.graph['label'] = df.loc[i, 'output']
                smiles_g.graph['smiles_id'] = i
                # print(i,smiles_g.graph['smiles'], smiles_g.graph['label'])
                # smiles_g.graph['smiles_id'] = df.loc[i,'smiles_id']
                # print(smiles_g.nodes(data=True))
                g_dict.append(smiles_g)
                vocab_s = smiles_g.graph['vocab_dict']
                for k in vocab_s.keys():
                    if k not in vocab:
                        vocab[k] = 1
                    else:
                        vocab[k] = vocab[k] + 1

            else:
                print(s, 'error')
                g_dict.append(None)
    return g_dict, vocab


def get_vocab(root, task, ncpu=8):
    data_path = os.path.join(root,task)
    #df = pd.read_csv(os.path.join(root,'raw.csv'))
    df = pd.read_csv(os.path.join(data_path, 'raw.csv'))

    # df['smiles_id'] = df.index.tolist()
    df = df.fillna(0)
    smiles_list = df['smiles'].tolist()
    print(len(smiles_list))
    vocab_dict = {}
    vocab_df = pd.DataFrame()
    g_list, vocab_dict = update_vocab(vocab_dict, df)
    for k, v in vocab_dict.items():
        vocab_df = vocab_df.append({'smiles': k, 'frequency': v}, ignore_index=True)

    '''
    batch_size = len(smiles_list) // ncpu + 1
    batches = [smiles_list[i: i + batch_size] for i in range(0, len(smiles_list), batch_size)]
    pool = Pool(ncpu)
    func = partial(update_vocab, vocab_dict)
    res = pool.map(func, batches)
    g_list, vocab_dict_list = zip(*res)
    g_list = get_list(g_list)
    pool.close()
    pool.join()

    print(smiles_list[40], g_list[40].graph['vocab_dict'])

    # vocab_df = pd.concat(vocab_df,axis=0)

    # print(np_dict_list)
    for d in vocab_dict_list:
        for k, v in d.items():
            vocab_df = vocab_df.append({'smiles': k, 'frequency': v}, ignore_index=True)
            # print(vocab_df)
    '''
    #print(g_list[40].graph['vocab_dict'], g_list[40].graph['label'])
    vocab_df = vocab_df.groupby('smiles', as_index=False).sum()
    # print(vocab_df[vocab_df['smiles'] == 'C'])
    vocab_df = vocab_df.sort_values(by='frequency', ascending=False)
    vocab_df = vocab_df.reset_index(drop=True)
    vocab_df['num_atoms'] = vocab_df['smiles'].apply(lambda x: get_mol(x).GetNumAtoms())
    vocab_df['frequency'] = vocab_df['frequency'].astype(int)
    torch.save(g_list, os.path.join(path,  'group_graph.pt'))
    vocab_df.to_csv(os.path.join(path, 'vocab.csv'))
    print(len(g_list))
    return g_list, vocab_df


def has_edge(g,nodes_a,nodes_b):
    inter_pairs = []
    for a in nodes_a:
        for b in nodes_b:
            if g.has_edge(a, b):
                for e_id, attr in g[a][b].items():
                    inter_pairs.append(attr['inter_pair'])
    return inter_pairs






def get_mol_tree(g_list,depth = 3):
    active_frag = set()
    mol_tree_list = []
    for i,smiles_g in enumerate(tqdm(g_list)):
        mol = smiles_g.graph['mol']
        mol_cls = tuple(i for i in range(mol.GetNumAtoms()))
        #print(s)
        path_list = set()
        for u in  list(smiles_g):
            path =[tuple(set(sorted(v))) for v in nx.single_source_shortest_path(smiles_g, u,cutoff=depth).values()]
            if mol_cls in path_list:
                path.remove(mol_cls)
            path_list.update(path)

        for nodes in path_list:
            nodes_n = set(tuple(smiles_g)) - nodes
            inter_pairs = set()
            nodes_idx = set()
            nodes_n_idx = set()
            for a in nodes:
                nodes_idx.update(smiles_g.nodes[a]['frag_idx'])
                for b in nodes_n:
                    nodes_n_idx.update(smiles_g.nodes[b]['frag_idx'])
                    if smiles_g.has_edge(a, b):
                        for e_id, attr in g[a][b].items():
                            inter_pairs.add(attr['inter_pair'])

            bond_indices = []  # bond indices of these bonds
            bond_labels = []  # labels are defined so that attachment points belonging to same bond are marked with same label
            for i, b in enumerate(inter_pairs):
                bond_indices.append(mol.GetBondBetweenAtoms(b[0], b[1]).GetIdx())
                bond_labels.append((i + 1, i + 1))

            frag_mol = Chem.FragmentOnBonds(mol, bond_indices, dummyLabels=bond_labels)
            frags_smiles = Chem.MolToSmiles(frag_mol)
            mol_tree_list.append(frags_smiles)
            print(frags_smiles)

    return mol_tree_list






def update_active_frag(smiles_list,depth = 3):
    active_frag = set()
    for i,s in enumerate(tqdm(smiles_list)):
        try:
            #print(s)
            mol_split = MolSplit(s)
            smiles_g = mol_split.get_mol_graph()
            #print(smiles_g.edges(data=True))

            mol = smiles_g.graph['mol']
            mol_cls = tuple(i for i in range(mol.GetNumAtoms()))
            #print(s)
            path_list = set()
            for u in  list(smiles_g):
                #print(nx.single_source_shortest_path(smiles_g, u,cutoff=depth))
                path =[tuple(set(sorted(v))) for v in nx.single_source_shortest_path(smiles_g, u,cutoff=depth).values()]
                #print(path)
                path_list.update(path)
            if mol_cls in path_list:
                path_list.remove(mol_cls)
            #print(path_list)
            for nodes in path_list:
                path_idx = set()
                sub_g = smiles_g.subgraph(nodes)
                path_mw = 0
                for n, attr in sub_g.nodes(data=True):
                    path_mw = path_mw + Descriptors.MolWt(attr['sub_mol'])
                    path_idx.update(attr['frag_idx'])
                for u,v, attr in sub_g.edges(data=True):
                    path_idx.update(attr['inter_pair'])
                if 150 <= path_mw <= 300:
                    #print(path_idx)
                    active_smiles = get_smiles(get_sub_mol(mol,path_idx))
                    if '.' not in  active_smiles:
                        active_frag.add(active_smiles)
        except:
            print(i,s)
            continue

    return active_frag


def get_active_frag(smiles_list):
    ncpu = 8
    batch_size = len(smiles_list) // ncpu + 1
    batch = [smiles_list[i: i + batch_size] for i in range(0, len(smiles_list), batch_size)]
    pool = Pool(ncpu)
    #func = partial(update_active_frag,depth)
    actice_frag_list = pool.map(update_active_frag, batch)
    pool.close()
    pool.join()
    active_frags = set()
    for x in actice_frag_list:
        active_frags.update(x)
    #print(active_frags)

    df = pd.DataFrame(columns=['smiles', 'mw', 'hbd', 'hba', 'logp', 'tpsa', 'nrot'])
    df['smiles'] = list(active_frags)
    df['mw'], df['logp'], df['hbd'], df['hba'], df['tpsa'], df['nrot'] = zip(*df['smiles'].apply(compute_ro3))
    return df

def compute_ro3(s):
    m = get_mol(s)

    declist = Descriptors.descList
    calc = {}
    for (i, j) in declist:
        calc[i] = j
    mw = np.round(Descriptors.MolWt(m), 1)
    logp = np.round(Descriptors.MolLogP(m), 2)
    hbd = rdMolDescriptors.CalcNumLipinskiHBD(m)
    hba = rdMolDescriptors.CalcNumLipinskiHBA(m)
    tpsa = np.round(Descriptors.TPSA(m), 1)
    nrot = rdMolDescriptors.CalcNumRotatableBonds(m)
    # qed = np.round(QED.qed(m), 2)
    # chiral_center = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    #print(mw, logp,hbd,hba,tpsa,nrot)
    return mw, logp,hbd,hba,tpsa,nrot



if __name__ == "__main__":
    root =  '/home/pycao/hgraph2graph-master/hgraph2graph-master/dataset/dataset/class_task'
    #root = 'J:/hgraph2graph-master/hgraph2graph-master/dataset/chembl'

    task_name_list = ['bbbp']

    #root = '/home/pycao/GraphRXN-master/GraphRXN-master/data_scaler/Buchwald-Hartwig/random_split'
    for task in task_name_list:
        path = os.path.join(root, task)
        #path = root
        time_start = time.time()
        data_path = os.path.join(path, task + '.csv')
        data_df = pd.read_csv(data_path, header=0)
        '''
        data_path_test = os.path.join(path, 'FullCV_01_test_temp_scaler' + '.csv')
        data_df_test = pd.read_csv(data_path_test, header=0)
        data_df = pd.concat([data_df_train,data_df_test])
        '''

        l = len(data_df)
        #data_df['reaction'] = data_df['reaction'].apply(lambda x: x.replace('*','.'))
        #data_df['smiles'] = data_df['smiles'].apply(lambda x: remove_chirality(x))
        #data_df['smiles'] = data_df['reaction']
        data_df = data_df.replace(to_replace='None', value=np.nan).dropna(subset=['smiles'])
        data_df.fillna(0.)

        data_df.to_csv(path + '/raw.csv',index=False)
        #print(data_df['smiles'].tolist())
        #smiles_list = data_df['smiles'].tolist()
        print(task, l - len(data_df), ' smiles sanitize fail')

        g_list, vocab_df = get_vocab(root, task, ncpu=8)
        #vocab_df.to_csv(path + '/vocab.csv')
        #pd.set_option('display.max_columns', None)
        print(vocab_df)


        '''
     
        data_path = os.path.join(path, task +'_mw500.smi')
        smiles_list = csv_2_list(data_path)
        data_df = pd.DataFrame()
        data_df['smiles'] = smiles_list

        l = len(data_df)
        data_df['smiles'] = data_df['smiles'].apply(
            lambda x: standardize_smi(x, basicClean=True, clearCharge=False, clearFrag=False))
        print(len(data_df))
        # data_df['smiles'] = data_df['smiles'].apply(lambda x: sanitize(x, kekulize=True, isomeric=False))
        data_df = data_df.replace(to_replace='None', value=np.nan).dropna(subset=['smiles'])
        # print(data_df)
        data_df.fillna(0.)
        print(task, l - len(data_df), ' smiles sanitize fail')
        
        data_df.to_csv(path + '/data_remove_chirality.csv')
        

        smiles_list = data_df['smiles'].tolist()
        #smiles_list = [' O=c1[nH]cnc2oc3ccccc3c(=O)c12','O=c1ssc2ncccc12','[CaH2]','O=c1ccc2ccccc2[se]1']
        #update_active_frag(smiles_list, depth=3)
        df = get_active_frag(smiles_list)
        df.to_csv(path + '/active_frags.csv', index=False)
        print(len(df))
        data = df.query("(-3<=logp<=3) and (hbd<=3) and (hba <=3) and (tpsa<=60) and (nrot<=3)")
        print(len(data))

        time_end = time.time()
        print(task, time_end - time_start)
       
        #vocab_list = pd.read_csv(vocab_path,header=0)['smiles'].tolist()
        #if not os.path.exists(path):
            #os.mkdir(path)
            #shutil.copyfile('../dataset/ADMETlab_data/' + task+ '_canonical.csv', data_path)
        #smiles_list = csv_2_list(data_path)[:10]
        #smiles_list = data_df['smiles'].tolist()[:10]


        '''


