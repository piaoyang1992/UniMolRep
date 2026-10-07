#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 2024/7/3 19:36 
# @Author : piaoyang cao 
# @File : mol_group_graph.py

import os
import argparse
import random

import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem

from src.data.chem_utils import molecule2smiles, smiles2molecule
from src.data.chem_utils import GeneralVocab
from src.data.logger import print_log
from src.data.prepare_acylic_group_graph import MolSplit
#from src.data.prepare_mol_data_2 import MolSplit

PIECE_CONNECT_NUM = 9  # max 5 connects, add 1 padding and 3 reserved
import torch

from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
#from sklearn import preprocessing

# 记录搜索路径
import re
from rdkit.Chem.MolStandardize import rdMolStandardize
from torch_geometric.data import Data
import networkx as nx

def standardize_s(s):
    try:
        uncharger = rdMolStandardize.Uncharger()
        mol = Chem.MolFromSmiles(s)
        mol = uncharger.uncharge(mol)
        s = Chem.MolToSmiles(mol)
    except:
        s= None
        print(s, 'standardize error')
    return s





def get_vocab(fname,label=None):
    print_log(f'Loading mols from {fname} ...')
    root, task  = os.path.split(fname)
    vocab_path = os.path.join(root,'vocab.csv')
    graph_path = os.path.join(root,'group_graph.pt')
    df = pd.read_csv(fname, header=0)
    #df['smiles'] = df['smiles'].apply(lambda x: standardize_s(x))

    vocab = {}
    g_dict = {}
    for i in tqdm(range(len(df))):
        s = df.loc[i, 'smiles']
        if smiles2molecule(s) != None:
            mol_split = MolSplit(s)
            g = mol_split.get_mol_graph()

            nodes = {n: n for n in g.nodes()}  # 节点id:遍历顺序
            #nodes_label = {n: attr['sub_smiles'] for n, attr in g.nodes(data=True)}
            edges = [tuple(sorted(e)) for e in g.edges()]

            seq_set = set()
            #root_list = list(nodes.keys())
            i=0

            while i<10:
                #root = random.choice(root_list)
                tree = TreeCodec()
                seq = tree.encode_to_sequence(nodes, edges)
                decode_nodes, decode_edges = tree.decode_from_sequence(seq)
                check_edge = [tuple(sorted([decode_nodes[x], decode_nodes[y]])) for x, y in decode_edges]
                #print(sorted(edges), sorted(check_edge))
                #print(sorted(list(decode_nodes.keys())), sorted(nodes))
                if sorted(list(decode_nodes.keys())) == sorted(nodes) and sorted(check_edge)==sorted(edges):


                    seq_set.add(tuple(seq))
                    #print(seq_set)
                else:
                    print(s,'group tree error')
                i+=1
            vocab_s = list(seq_set)[0]
            vocab_s = [g.nodes[node]['sub_smiles'] if isinstance(node, int) else node for node in vocab_s]
            for k in vocab_s:
                if k not in vocab:
                    vocab[k] = 1
                else:
                    vocab[k] = vocab[k] + 1
            g.graph['seq'] = seq_set
            if label:
                g.graph['label'] = list(df.iloc[i,label])
            else:
                g.graph['label'] = None
            g_dict[s] = g

        else:
            print(s, 'smiles error')
            g_dict[s] = None

        #print(g_dict[s].graph['seq'])
    torch.save(g_dict,graph_path)
    #v = list(vocab.items())
    #v = sorted(v, key=lambda i: i[1], reverse=True)
    vocab_df = pd.DataFrame.from_dict(vocab, orient='index', columns=['frequency'])
    vocab_df = vocab_df.reset_index().rename(columns={'index': 'smiles'})
    vocab_df = vocab_df.groupby('smiles', as_index=False).sum()
    # print(vocab_df[vocab_df['smiles'] == 'C'])
    vocab_df = vocab_df.sort_values(by='frequency', ascending=False)
    vocab_df = vocab_df.reset_index(drop=True)

    vocab_df['frequency'] = vocab_df['frequency'].astype(int)
    vocab_df['atom_num'] = vocab_df['smiles'].apply(lambda x: smiles2molecule(x).GetNumAtoms() if smiles2molecule(x) else 0)
    vocab_df['inter_num'] = vocab_df['smiles'].apply(lambda x: x.count('[*]'))
    vocab_df.to_csv(vocab_path)
    return g_dict, vocab_df


def get_inter_pieces_idx(s):
    mol_split = MolSplit(s)
    mol = mol_split.mol
    pieces_list, group_idx = mol_split.get_mol_seq()
    group_idx = list(group_idx)
    n = []
    for p in pieces_list:
        rep = re.sub('\[\d+\*]', r'[*]', p)
        n.append(rep)
    for i, g in enumerate(group_idx):
        group_idx[i] = [idx for idx in g if idx < mol.GetNumAtoms()]
    return n, group_idx



def get_vocab_descriptors(vocab_list,n_components=10):
    vocab_tensor = []
    for v in vocab_list:
        mol = Chem.MolFromSmiles(v)
        #fp = Torsions.GetTopologicalTorsionFingerprintAsIntVect(mol)

        des_list = [x[0] for x in Descriptors._descList]
        # print(des_list)
        calculator = MoleculeDescriptors.MolecularDescriptorCalculator(des_list)
        fp = list(calculator.CalcDescriptors(mol))
        #print(len(descriptors))

        vocab_tensor.append(fp)


    vocab_tensor = np.matrix(vocab_tensor)
    vocab_tensor = np.nan_to_num(vocab_tensor)

    '''
    minmax_scaler = preprocessing.MinMaxScaler()
    vocab_tensor = minmax_scaler.fit_transform(vocab_tensor)

    
    pca = PCA(n_components=n_components)
    vocab_pca = pca.fit_transform(vocab_tensor)
    '''
    #fa = FactorAnalysis(n_components=10)
    #X_transformed = fa.fit_transform(vocab_tensor)
    #print(X_transformed.size())

    vocab_tensor = torch.from_numpy(vocab_tensor).to(torch.float32)

    vocab_tensor = torch.where(torch.isinf(vocab_tensor), torch.full_like(vocab_tensor, 0), vocab_tensor)
    vocab_tensor = torch.where(torch.isnan(vocab_tensor), torch.full_like(vocab_tensor, 0), vocab_tensor)
    return vocab_tensor


#函数一，获取marker邻居原子的index, 注意marker只能是一个单键连接核上的原子，否则邻居会多于一个
def get_nei_bysymbol(mol,marker):
    atom_nbs = []
    try:
        for atom in mol.GetAtoms():
            if atom.GetSymbol()==marker:
                neighbors=atom.GetNeighbors()
                if len(neighbors)>1:
                    print ('Cannot process more than one neighbor, will only return one of them')
                n = neighbors[0]
                atom_nbs.append(n)
        return atom_nbs
    except Exception as e:
        print (e)
        return None
#函数二，获取marker原子的index
def remove_marker(mol,marker):
    while marker in molecule2smiles(mol):
        for atom in mol.GetAtoms():
            if atom.GetSymbol()==marker:
                ed_mol = Chem.EditableMol(mol)
                #print(molecule2smiles(mol),mol.GetNumAtoms())
                ed_mol.RemoveAtom(atom.GetIdx())
                mol = ed_mol.GetMol()
                break

        #print(molecule2smiles(mol))
    #final_mol = ed_merged_mol.GetMol()
    return mol

split_smi = ['C+','C-','O','S','N','P+','As', 'As+','Fe', 'Al','Sn', '#0', 'Mg', 'Si', 'Se', 'F', 'Cl', 'Br', 'I', '#5',
               'Ge', 'Ni', 'Se','Ca', 'Cu','Li', 'Ru', 'Co', 'Pt','Ir', 'Pd', 'W','N','O','C=C','C#C','C=N', 'C=O','C=S' ,
               'C#N','C=P','[N+][O-]','N=N','N#N','N=O','S=O','S=S','S=N','P=O','As=O','O=N=O','[N+]=[N-](=O)',
               '[N+]=[N-]=N','O=S=O','O=P=O']

def Calc_AROM(mol):
    m = Chem.RemoveHs(mol)
    ring_info = m.GetRingInfo()
    atoms_in_rings = ring_info.AtomRings()
    num_aromatic_ring = 0
    for ring in atoms_in_rings:
        aromatic_atom_in_ring = 0
        for atom_id in ring:
            atom = m.GetAtomWithIdx(atom_id)
            if atom.GetIsAromatic():
                aromatic_atom_in_ring += 1
        if aromatic_atom_in_ring == len(ring):
            num_aromatic_ring += 1
    return num_aromatic_ring

def frag_type(smi):
    mol = smiles2molecule(smi,kekulize=False)
    if Calc_AROM(mol) > 0:
        return 2
    for s in split_smi:
        if s in smi:
            return 1
        else:
            continue
    return 0



import random



class TreeCodec:
    def __init__(self, root=None):
        """
        Initialize the TreeCodec class.

        Args:
            random (bool): If True, enables random traversal for encoding;
                           if False, uses deterministic traversal.
        """
        self.root = root

    def encode_to_sequence(self, nodes, edges):
        """
        Encodes a tree into a sequence.
        """

        # Step 2: Find the longest path (main branch)
        root = self.root
        # Step 1: Build adjacency list
        graph = {node_id: [] for node_id, label in nodes.items()}
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)

        self.graph = graph
        main_branch = list(self.find_longest_path(graph, root))
        # print(main_branch,self.root)
        root_list = range(len(main_branch))

        c = random.choice(root_list)
        main_branch = main_branch[c]

        # Step 3: Generate the sequence
        visited = set()
        result = []
        max_branch_num = 0  # Track the maximum branch number used

        def get_branch_number(branch_name):
            """Helper function to extract number from branch name"""
            if not branch_name or not branch_name.startswith('branch'):
                return 0
            return int(branch_name[6:])  # Extract number after 'branch'

        def tree_to_sequence_with_branch(node, graph, visited, main_branch, result, parent_branch=None):
            """
            Helper function that includes branch markers in the sequence
            """
            nonlocal max_branch_num

            if node in visited:
                return
            visited.add(node)
            result.append(node)

            # Get all unvisited neighbors
            neighbors = [n for n in graph[node] if n not in visited]
            if not neighbors:
                return

            # Separate main branch and side branch neighbors
            main_neighbors = [n for n in neighbors if n in main_branch]
            side_neighbors = [n for n in neighbors if n not in main_branch]

            # Process side branches first
            for neighbor in side_neighbors:
                # Find current maximum branch number in the sequence
                current_max = max(
                    (get_branch_number(x) for x in result if isinstance(x, str) and x.startswith('branch')),
                    default=0
                )

                # Use next number after current maximum
                next_branch_num = current_max + 1
                max_branch_num = max(max_branch_num, next_branch_num)
                branch_name = f'branch{next_branch_num}'

                # Start branch
                result.append(branch_name)

                # Process branch
                tree_to_sequence_with_branch(neighbor, graph, visited, main_branch, result, branch_name)

                # Close branch
                result.append(branch_name)

            # Process main branch neighbors
            for neighbor in main_neighbors:
                tree_to_sequence_with_branch(neighbor, graph, visited, main_branch, result, parent_branch)

        # Use the new helper function starting from the first node in main_branch
        tree_to_sequence_with_branch(main_branch[0], graph, visited, main_branch, result)

        # Convert node IDs to labels
        sequence_with_labels = [nodes[node] if isinstance(node, int) else node for node in result]

        return sequence_with_labels

    def decode_from_sequence(self, sequence):
        """
        Decodes a sequence back into a tree structure.
        Args:
            sequence: List of tokens (strings or integers) including branch markers and SMILES fragments
        Returns:
            nodes: Dictionary mapping node IDs to their labels
            edges: List of tuples representing edges (parent, child)
        """
        nodes = {}
        edges = []
        node_idx = 0
        stack = []
        branch_stacks = {}  # {branchX: parent_node_id}

        def is_branch_token(token):
            return isinstance(token, str) and token.startswith('branch')

        for i, token in enumerate(sequence):
            if is_branch_token(token):
                if token in branch_stacks:
                    # 关闭分支，返回到父节点
                    if stack:
                        stack.pop()
                else:
                    # 开启新分支，记录父节点
                    branch_stacks[token] = stack[-1] if stack else None
            else:
                # 按顺序添加节点
                nodes[node_idx] = token

                # 如果栈不为空，创建边
                if stack:
                    edges.append((stack[-1], node_idx))

                # 更新栈
                stack.append(node_idx)
                node_idx += 1

        return nodes, edges

    def find_longest_path(self, graph, root):
        """
        Finds the longest path in the tree using two DFS traversals (deterministic).
        """
        # First DFS: Find the farthest node from an arbitrary starting node (e.g., node 0)
        if self.root:
            farthest_node = self.dfs_find_farthest(root, graph)
            longest_path = self.dfs_find_path(farthest_node, graph)
            longest_path = {tuple(longest_path)}
        else:
            longest_path = set()
            max_l = 0
            for n in range(len(graph)):
                farthest_node = self.dfs_find_farthest(n, graph)
                p = self.dfs_find_path(farthest_node, graph)
                if len(p) > max_l:
                    longest_path = {tuple(p)}
                    max_l = len(p)
                elif len(p) == max_l:
                    longest_path.add(tuple(p))
        return longest_path

    def tree_to_sequence(self, node, graph, visited, main_branch, result):
        """
        Recursively traverse the tree and generate the encoded sequence.
        """
        visited.add(node)
        result.append(node)

        # Separate children into main branch and side branches
        children = [neighbor for neighbor in graph[node] if neighbor not in visited]
        # if self.standard:
        random.shuffle(children)  # Randomize the order of children
        main_children = [child for child in children if child in main_branch]
        side_children = [child for child in children if child not in main_branch]

        # Process side branches
        for child in side_children:
            result.append("(")
            self.tree_to_sequence(child, graph, visited, main_branch, result)
            result.append(")")

        # Process main branch children
        for child in main_children:
            self.tree_to_sequence(child, graph, visited, main_branch, result)

    def dfs_find_farthest(self, start, graph):
        """
        Finds the farthest node from a starting node using DFS.
        """
        visited = set()
        stack = [(start, 0)]
        farthest_node = start
        max_depth = 0

        while stack:
            node, depth = stack.pop()
            if depth > max_depth:
                max_depth = depth
                farthest_node = node

            visited.add(node)
            neighbors = [neighbor for neighbor in graph[node] if neighbor not in visited]
            # if self.standard:
            random.shuffle(neighbors)
            for neighbor in neighbors:
                stack.append((neighbor, depth + 1))

        return farthest_node

    def dfs_find_path(self, start, graph):
        """
        Finds the longest path from a starting node using DFS.
        """
        visited = set()
        stack = [(start, [start])]
        max_path = []

        while stack:
            node, path = stack.pop()
            visited.add(node)

            if len(path) > len(max_path):
                max_path = path

            neighbors = [neighbor for neighbor in graph[node] if neighbor not in visited]
            # if self.standard:
            random.shuffle(neighbors)
            for neighbor in neighbors:
                stack.append((neighbor, path + [neighbor]))

        return max_path


def organize_fragments(df):
    # 创建一个字典来存储结果
    fragments_dict = {}

    # 先处理普通情况
    df_normal = df[~((df['inter_num'] == 0) & (df['atom_num'] == 0))]
    for inter_num in df_normal['inter_num'].unique():
        mask = df_normal['inter_num'] == inter_num
        smiles_list = df_normal[mask].sort_values('frequency', ascending=False)['smiles'].tolist()
        fragments_dict[inter_num] = smiles_list

    # 处理特殊情况：inter_num = 0 且 atom_num = 0
    special_key = len(fragments_dict)
    mask_special = (df['inter_num'] == 0) & (df['atom_num'] == 0)  # branch
    special_smiles = df[mask_special].sort_values('frequency', ascending=False)['smiles'].tolist()
    fragments_dict[special_key] = special_smiles

    # 添加特殊标记
    # fragments_dict[len(fragments_dict)] = ['<pad>']
    #fragments_dict[len(fragments_dict)] = ['<inter_s>']  # 倒数第二行
    max_length = max(len(value) for value in fragments_dict.values())

    # 创建一个空的DataFrame
    new_df = pd.DataFrame(index=range(max_length))

    # 填充DataFrame
    for key in sorted(fragments_dict.keys()):
        # 获取当前key的smiles列表
        smiles_list = fragments_dict[key]
        # 将smiles列表扩展到最大长度，用None填充
        padded_list = smiles_list + [None] * (max_length - len(smiles_list))
        # 添加到DataFrame中
        new_df[key] = padded_list

    #print(new_df)
    new_df.loc[len(new_df)] = ['<piece_start>'] * new_df.shape[1]
    #new_df.iloc[len(new_df)-1,0] = '<piece_start>'
    #print(new_df)
    new_df[new_df.shape[1]] = ['<inter_start>'] * len(new_df)
    #new_df.iloc[0,new_df.shape[1]-1] = '<inter_start>'
    #print(new_df)

    new_df[new_df.shape[1]] = ['<inter_pad>'] * len(new_df)   # piece_num pad  倒数第一列
    #new_df.iloc[0, new_df.shape[1] - 1] = '<inter_pad>'

    new_df.loc[len(new_df)] = ['<piece_pad>'] * new_df.shape[1]  # inter_num pad  倒数第一行
    #new_df.iloc[len(new_df) - 1, 0] = '<piece_pad>'
    return new_df




class Tokenizer:
    def __init__(self, task_file,label=None):
        self.path  = task_file
        self.root, self.task = os.path.split(task_file)
        vocab_path = os.path.join(self.root, 'vocab.csv')
        data_path = os.path.join(self.root, 'group_graph.pt')
        if not os.path.exists(data_path):
            print('get group graph')
            g_dict, vocab_df = get_vocab(self.path, label=label)
            #vocab_df['inter_num'] = vocab_df['smiles'].apply(lambda x: x.count('[*]'))
            self.group_graphs = g_dict
        else:
            self.group_graphs = torch.load(data_path)
            #print(len(self.group_graphs))

            vocab_df = pd.read_csv(vocab_path, header=0)

            #vocab_df['inter_num'] = vocab_df['smiles'].apply(lambda x: x.count('[*]'))
            #vocab_df.to_csv(vocab_path)
        self.vocab_df = vocab_df
        self.sort_df = organize_fragments(vocab_df)
        #print(self.sort_df)
        #self.vocab_dict = {}
        self.idx2piece, self.piece2idx = [], {}
        self.id2type = []
        self.id2inter = []



        for i in range(len(vocab_df)):
            smi, atom_num, freq = vocab_df.loc[i,'smiles'], vocab_df.loc[i,'atom_num'], vocab_df.loc[i,'frequency']
            #self.vocab_dict[smi] = (int(atom_num), int(freq))
            self.piece2idx[smi] = len(self.idx2piece)
            self.idx2piece.append(smi)
            #self.id2type.append(frag_type(smi))
            self.id2inter.append(smi.count('[*]'))

        self.pad, self.end = '<pad>', '<s>'
        #self.branch_start, self.branch_end = '(', ')'
        for smi in [self.pad, self.end]:
            self.piece2idx[smi] = len(self.idx2piece)
            self.idx2piece.append(smi)
            self.id2inter.append(smi.count('[*]'))



        # for fine-grained level (atom level)
        #print(self.piece2idx['('])
        self.atom_pad = '<pad>'
        self.chem_vocab = GeneralVocab(atom_special=[self.atom_pad])


        self.label = label

    def tokenize(self, s):
        '''
        group_idxs = []
        pieces = []
        g = self.group_graphs[s]
        inter_pair = g.graph['inter_pair']
        mol = g.graph['mol']
        #trees = list(nx.bfs_tree(group_graph, source=0))

        pieces_list = list(g)

        for p in pieces_list:
            group_idxs.append(g.nodes[p]['frag_idx'])
            pieces.append(self.piece2idx[g.nodes[p]['sub_smiles']])

        '''

        if s not in self.group_graphs:
            print(s,'not in group_graph')
            mol_split = MolSplit(s)
            g = mol_split.get_mol_graph()
            nodes = {n: n for n in g.nodes()}  # 节点id:遍历顺序
            edges = [tuple(sorted(e)) for e in g.edges()]

            #root_list = list(nodes.keys())
            seq_set = set()
            i = 0
            while i < 10:
                #root = random.choice(root_list)
                tree = TreeCodec()
                seq = tree.encode_to_sequence(nodes, edges)

                decode_nodes, decode_edges = tree.decode_from_sequence(seq)
                check_edge = [tuple(sorted([decode_nodes[x], decode_nodes[y]])) for x, y in decode_edges]
                if sorted(list(decode_nodes.keys())) == sorted(nodes) and sorted(check_edge) == sorted(edges):
                    seq_set.add(tuple(seq))
                i += 1

            g.graph['seq'] = seq_set

        else:
            g = self.group_graphs[s]

        '''
        pieces_list = list(g)
        group_idxs = []
        pieces = []

        for p in pieces_list:
            group_idxs.append(g.nodes[p]['frag_idx'])
            pieces.append(self.piece2idx[g.nodes[p]['sub_smiles']])

        res =list(zip(pieces, group_idxs))
        res = sorted(res, key=lambda i: i[0], reverse=True)
        #np.random.shuffle(res)
        pieces = [self.piece2idx['<s>']] + [x[0] for x in res] + [self.piece2idx['<s>']] # 基团的序号
        group_idxs = [[]] +[x[1] for x in res]  + [[]] # 基团的原子编号
        '''
        return g

    def get_vocab_data(self):
        self.group_tensor_path = os.path.join(self.root,'vocab_tensors.pt')
        if os.path.exists(self.group_tensor_path):
            vocab_tensor = torch.load(self.group_tensor_path)
        else:
            vocab_tensor = get_vocab_descriptors(self.idx2piece)
            torch.save(vocab_tensor, self.group_tensor_path)

        return vocab_tensor

    def idx_to_piece(self, idx):
        return self.idx2piece[idx]

    def piece_to_mol(self, p, marker=None):
        # maker *
        #p = self.idx2piece[idx]
        try:
            mol = Chem.MolFromSmiles(p)
            Chem.Kekulize(mol,clearAromaticFlags=True) # 不能写成 mol = Chem.Kekulize(mol,clearAromaticFlags=True)
            for atom in mol.GetAtoms():
                atom.SetUnsignedProp('inter', 0)
            if marker:
                bind_atoms = get_nei_bysymbol(mol, marker)
                for atom in bind_atoms:
                    i = int(atom.GetProp('inter')) + 1
                    atom.SetUnsignedProp('inter', i)
                final_mol = remove_marker(mol, marker)
            else:
                final_mol = mol
            #print(final_mol)
            #final_mol = Chem.Kekulize(final_mol,clearAromaticFlags=True)
        except Exception:
            print(p, 'Kekulize error')
            final_mol = None

        return final_mol

    def piece_to_idx(self, piece):
        return self.piece2idx[piece]

    def piece_to_type(self, idx):
        return self.id2type[idx]

    def id_to_inter(self, idx):
        return self.id2inter[idx]

    def pad_idx(self):
        return self.piece2idx[self.pad]

    def inter_pad_idx(self):
        return self.sort_df.shape[1] - 1
    def piece_pad_idx(self):
        return self.sort_df.shape[0] - 1

    def end_idx(self):
        return self.piece2idx[self.end]

    def atom_pad_idx(self):
        return self.chem_vocab.atom_to_idx(self.atom_pad)

    def num_piece(self):
        return len(self.idx2piece)


    def get_smiles_location(self, piece):

        if piece == '<piece_start>':
            return self.sort_df.shape[0] - 2
        if piece == '<inter_start>':
            return self.sort_df.shape[1] - 2
        if 'branch' in piece:
            inter_num = self.sort_df.shape[1] - 3
        else:
            inter_num = piece.count('[*]')
        mask = self.sort_df[inter_num] == piece
        if mask.any():
            # 获取行索引
            row_idx = mask[mask].index[0]
            # 计算元素索引
            # element_index = row_idx * len(self.sort_df.columns) + inter_num
            return (inter_num, row_idx)

        return (None, None)
    def num_inter(self):
        return self.sort_df.shape[1]

    def num_frag_type(self):
        return self.sort_df.shape[0]

    def get_mask_tensor(self):
        result = np.where(self.sort_df.isna(), 1, 0)
        mask_tensor = torch.tensor(result, dtype=torch.long)
        return mask_tensor

    def num_atom_type(self):
        return self.chem_vocab.num_atom_type()

    def num_bond_type(self):
        return self.chem_vocab.num_bond_type()

    def __call__(self, s):
        return self.tokenize(s)

    def __len__(self):
        return len(self.idx2piece)


def parse():
    parser = argparse.ArgumentParser(description='Graph bpe')
    parser.add_argument('--smiles', type=str, default='CC[NH+]1CCC[C@@H]1C[NH+](C)CCC(=O)[O-]',
                        help='The molecule to tokenize (example)')
    parser.add_argument('--data', type=str, default='/home/pycao/PS-VAE-main-1/data/np/np_sample.csv',
                        help='Path to molecule corpus')
    parser.add_argument('--vocab_size', type=int, default=300, help='Length of vocab')
    parser.add_argument('--output', type=str, default='/home/pycao/PS-VAE-main-1/data/np',
                        help='Path to save vocab')
    parser.add_argument('--workers', type=int, default=16, help='Number of cpus to use')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse()
    # args.data = '/../data/zinc250k/train.txt'
    # args.output =  '/../data/zinc_bpe_300.txt'
    #g_dict, vocab = get_vocab(args.data)
    '''
    s = 'CC[NH+]1CCC[C@@H]1C[NH+](C)CCC(=O)[O-]'
    mol_split = MolSplit(s)
    smi_list, inter_mol = mol_split.get_mol_seq()
    print(smi_list, inter_mol)
    pieces_list = get_clean_pieces_idx(s)
    print(pieces_list)
    smiles_g = mol_split.get_mol_graph()
    smiles_g = [n for n in smiles_g.nodes(data=True)]
    print(smiles_g)

    '''
    train_set = '/home/pycao/PS-VAE-main-1/data/zinc250k/train.txt'
    s = 'CCC1C(C(C(N(CC(CC(C(C(C(C(C(=O)O1)C)OC2CC(C(C(O2)C)O)(C)OC)C)OC3C(C(CC(O3)C)N(C)C)O)(C)O)C)C)C)O)(C)O'
    tokenizer = Tokenizer(train_set)

    #pieces, group_idxs, data, inter_pair = tokenizer('N=1C=NC2=C(N=CN2C3OC(COCCC(=C)C)C(O[Si](C)(C)C)C3O[Si](C)(C)C)C1N[Si](C)(C)C', return_idx=True)

    m = tokenizer.idx_to_mol(16, marker='*')
    s = molecule2smiles(m)
    print(s, tokenizer.idx_to_piece(16))

    m = Chem.MolFromSmiles('c1ccccc1')
    Chem.Kekulize(m)
    print(Chem.MolToSmiles(m))






