#from rdkit.Chem.Draw import SimilarityMaps
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import BRICS
from rdkit.Chem import Recap
from collections import Counter
import numpy as np
import numpy.random as npr
#import pubchempy as pcp
import re
from random import sample
import pandas as pd
import os
#from mordred import Calculator, descriptors
from rdkit.Chem import PandasTools, QED, rdMolDescriptors,Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
#from rdkit.Chem.MolStandardize import rdMolStandardize

from rdkit.Chem import MACCSkeys
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
#from ogb.utils.mol import smiles2graph
#import torch



split_smart = ['[#6;X3&+]', '[#6;X3&-]','[#8;X2]', '[#8;-]', '[#8;+]', '[#16;X2]', '[#16;X3&+]', '[#16;X1&-]', '[#7;X3&+0]',
                '[#7;X4&+]', '[#7;X5&2+]',  '[#7;-]', '[#15;X4&+]',
                '[As;X5]', '[As;X3]','[As;X4&+]','[Fe]', '[Al]',
               '[Sn]', '[#0]', '[Mg]', '[Si]', '[Se]', '[F]', '[Cl]', '[Br]', '[I]', '[#5]', '[Ge]', '[Ni]', '[Se]',
               '[Ca]', '[Cu]','[Li]', '[Ru]', '[Co]', '[Pt]','[Ir]', '[Pd]', '[W]',
               '[#6]=[#6]', '[#6]#[#6]', '[#6]=[#7]', '[#6]=[#16]','[#6]#[#7]', '[#6]=[#15]','[#7;X3&+][#8;X1&+0]',
               '[#7]=[#7]', '[#7]#[#7]', '[#7]=[#8]','[#6]=[#8]', '[#16]=[#8]','[#16]=[#16]','[#16]=[#7]', '[#15]=[#8]',
               '[As]=[#16]','[As]=[#8]',
               '[#7](=[#8])(=[#8])', '[#6]=[#7;+]=[#7;-]', '[#7]=[#7;+]=[#7;-]',
               '[#16](=[#8])(=[#8])',  '[#15](=[#8])(=[#8])']

'''
split_smart = ['[#8;X2]', '[#8;-]', '[#16;X2]', '[#16;X3&+]', '[#16;X1&-]', '[#7;X3&+0]', '[#7;X4&+]', '[#6]=[#6]', '[#6]#[#6]', '[#6]=[#7]', '[#6]#[#7]',
               '[#6]=[#15]', '[#7]=[#7]', '[#7]#[#7]', '[#7]=[#8]', '[#7](=[#8])(=[#8])', '[#6](=[#8])', '[#15;X4&+]',
               '[#16](=[#8])(=[#8])', '[#16](=[#8])', '[#16](=[#7])', '[#15](=[#8])', '[#15](=[#8])(=[#8])', '[#6](=[#16])', '[Fe]', '[Al]',
               '[Sn]', '[#0]', '[Mg]', '[Si]', '[Se]', '[#9]', '[#17]', '[#35]', '[#53]', '[#5]']
'''

split_smi_1 = ['[C+]','[C-]','[O]','[S]','[N]','[P+]',
            '[As]', '[As+]','[Fe]', '[Al]',
            '[Sn]', '[#0]', '[Mg]', '[Si]', '[Se]', '[F]', '[Cl]', '[Br]', '[I]', '[#5]', '[Ge]', '[Ni]', '[Se]',
            '[Ca]', '[Cu]','[Li]', '[Ru]', '[Co]', '[Pt]','[Ir]', '[Pd]', '[W]']
split_smi_2 = ['N','O','C=C','C#C','C=N', 'C=O','C=S' ,'C#N','C=P','[N+][O-]','N=N','N#N','N=O','S=O','S=S','S=N','P=O',
            'As=O','O=N=O','[N+]=[N-](=O)','[N+]=[N-]=N','O=S=O','O=P=O']





get_list = lambda a: [j for i in a for j in i]


#mol_list = [Chem.MolFromSmarts(s) for s in split_smi_2]



def list_2_csv(l, file):  # 将列表保存为csv
    with open(file, 'w') as f:
        for i in l:
            f.write(str(i) + '\n')


def csv_2_list(file):
    l = []
    with open(file, 'r') as f:
        for line in f:
            if len(line) != 0:
                l.append(line.strip('\n'))
    return l

def get_mol(smiles,kekulize=True):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if kekulize:
            Chem.Kekulize(mol)
    except:
        mol = None
    return mol


def get_smiles(mol, kekulize=True, isomeric=False):
    return Chem.MolToSmiles(mol, kekuleSmiles=kekulize,isomericSmiles=isomeric)


def remove_chirality(smi):
    smi = smi.replace('/', '')
    smi = smi.replace('\\', '')
    #patt = r'\[\d*.@+.{0,1}\]'
    patt = r'\[[C,c]H{0,2}\]'
    res = re.finditer(patt, smi)
    for r in res:
        #print(r)
        if '+' in r.group() or '-' in r.group():
            rep = re.sub(r'[0-9]+', '', r.group())
            rep = rep.replace('H', '')
            rep = rep.replace('@', '')
            # rep = ''.join(rep).replace('H','')
            smi = smi.replace(r.group(), rep)
        else:
            rep = re.sub(r'[0-9]+', '', r.group())
            rep = rep.replace('H', '')
            rep = rep.replace('@', '')
            rep = rep.replace('[', '')
            rep = rep.replace(']', '')
            smi = smi.replace(r.group(), rep)
    if sanitize(smi) == 'None':
        print(smi, 'error')

    return smi



def standardize_smi(smiles,basicClean=True,clearCharge=False, clearFrag=False):
    try:
        clean_mol = Chem.MolFromSmiles(smiles)
        # 除去氢、金属原子、标准化分子
        if basicClean:
            clean_mol = rdMolStandardize.Cleanup(clean_mol)

        if clearFrag:
             clean_mol = rdMolStandardize.FragmentParent(clean_mol)

        if clearCharge:
            uncharger = rdMolStandardize.Uncharger()
            clean_mol = uncharger.uncharge(clean_mol)
        # 处理互变异构情形，这一步在某些情况下可能不够完美
        stan_smiles = Chem.MolToSmiles(clean_mol)# idem
    except:
        print('molecule standard error')

    return stan_smiles


'''
def sanitize(smiles, kekulize=True, isomeric=False):  # kekulize不为True，拆出的芳香环sanitize错误
    try:
        mol = get_mol(smiles)
        new_smiles = get_smiles(mol,kekulize,isomeric)
        mol = get_mol(new_smiles)
        if mol == None:
            new_smiles = None
    except:
        try:
            new_smiles = pcp.get_properties('canonical_smiles', smiles, 'smiles')[0]['CanonicalSMILES']
            mol = get_mol(new_smiles)
            new_smiles = get_smiles(mol,kekulize,isomeric)
            mol = get_mol(new_smiles)
            if mol == None:
                new_smiles = None
            #new_smiles = remove_chirality(new_smiles)
            print(smiles, 'pubchem', new_smiles)
        except:
            print(smiles, 'sanitize error')
            new_smiles = None
    return new_smiles
'''

def sanitize(smiles, kekulize=True, isomeric=False):  # kekulize不为True，拆出的芳香环sanitize错误
    try:
        mol = get_mol(smiles)
        new_smiles = get_smiles(mol,kekulize,isomeric)
    except:
        new_smiles = None
    return new_smiles



def compute_similarity(m0,m1):
    f0 = MACCSkeys.GenMACCSKeys(m0)
    f1 = MACCSkeys.GenMACCSKeys(m1)
    s = DataStructs.FingerprintSimilarity(f0, f1)
    return s


def isRingAromatic(mol, ring):
    '''给定分子（MOL对象）和环中化学键的索引，若所有键都是芳香的，则这个环为芳香环。
    '''
    # print([mol.GetBondWithIdx(index).GetIsAromatic() for index in ring])
    if False in [mol.GetBondWithIdx(index).GetIsAromatic() for index in ring]:
        return False
    else:
        return True

def union_rings(rings):
    c = Counter(get_list(rings))
    a = [i for i, x in c.items() if x > 1]
    if len(a) > 0:
        new_ar_rings = rings
        for i in a:  # 共同元素是a的合并
            dup = set()
            n_dup = set()
            # print(i,new_ar_rings)
            for j in new_ar_rings:  # 更新ar_rings
                if i in j:  # 有相同元素合并
                    dup.update(j)
                else:  # 无则不变
                    n_dup.add(tuple(j))
            n_dup.add(tuple(dup))
            new_ar_rings = n_dup
            # print(new_ar_rings)
    else:
        new_ar_rings = rings
    return new_ar_rings



def get_ar_rings(mol):
    '''给定分子(MOL对象）， 返回分子中所有芳香环的原子索引。
       返回一个元素为元组的列表，一个元组是分子中一个芳香环的原子索引。
    '''

    ar_rings = []
    rings = mol.GetRingInfo()

    if rings:
        bond_rings = rings.BondRings()  # 返回每个环键序号
        for i, x in enumerate(bond_rings):
            if isRingAromatic(mol, x):
                ar_rings.append(list(rings.AtomRings()[i]))
                #print(ar_rings)

    # 将共轭的芳香环合并

    c = Counter(get_list(ar_rings))
    a = [i for i, x in c.items() if x > 1]
    if len(a) > 0:
        new_ar_rings = ar_rings
        for i in a:  # 共同元素是a的合并
            dup = set()
            n_dup = set()
            # print(i,new_ar_rings)
            for j in new_ar_rings:  # 更新ar_rings
                if i in j:  # 有相同元素合并
                    dup.update(j)
                else:  # 无则不变
                    n_dup.add(tuple(j))
            n_dup.add(tuple(dup))
            new_ar_rings = n_dup
            # print(new_ar_rings)
    else:
        new_ar_rings = ar_rings

    new_ar_rings = list(new_ar_rings)

    for i, x in enumerate(new_ar_rings): new_ar_rings[i] = tuple(x)  # tuple能作为字典的键，而list不能
    # print(new_ar_rings)

    return new_ar_rings




def get_all_rings(mol):
    '''给定分子(MOL对象）， 返回分子中所有芳香环的原子索引。
       返回一个元素为元组的列表，一个元组是分子中一个芳香环的原子索引。
    '''


    rings = Chem.GetSymmSSSR(mol)

    c = Counter(get_list(rings))
    a = [i for i, x in c.items() if x > 1]
    if len(a) > 0:
        new_ar_rings = rings
        for i in a:  # 共同元素是a的合并
            dup = set()
            n_dup = set()
            # print(i,new_ar_rings)
            for j in new_ar_rings:  # 更新ar_rings
                if i in j:  # 有相同元素合并
                    dup.update(j)
                else:  # 无则不变
                    n_dup.add(tuple(j))
            n_dup.add(tuple(dup))
            new_ar_rings = n_dup
            # print(new_ar_rings)
    else:
        new_ar_rings = rings


    new_ar_rings = list(new_ar_rings)
    for i, x in enumerate(new_ar_rings): new_ar_rings[i] = tuple(x)  # tuple能作为字典的键，而list不能
    # print(new_ar_rings)

    return new_ar_rings


def get_diff(a, b):
    new_b = []
    for i in a:
        for j in b:
            if not set(j).intersection(set(i)):
                new_b.append(j)
    return tuple(new_b)


def union_group(groups):
    c = Counter(get_list(groups))
    a = [i for i, x in c.items() if x > 1]
    if len(a) > 0:
        new_groups = groups
        for i in a:  # 共同元素是a的合并
            dup = set()
            n_dup = set()
            # print(i,new_ar_rings)
            for j in new_groups:  # 更新ar_rings
                if i in j:  # 有相同元素合并
                    dup.update(j)
                else:  # 无则不变
                    n_dup.add(tuple(j))
            n_dup.add(tuple(dup))
            new_groups = n_dup
            # print(new_ar_rings)
    else:
        new_groups = groups
    return new_groups


def get_split_dict(mol, ar_rings=True):
    split_dict = {}
    # 芳香环不作为双键识别,将芳香环作为一个断点
    if ar_rings:
        ar_rings = get_ar_rings(mol)
    else:
        ar_rings = get_all_rings(mol)


    ar_list = get_list(ar_rings)

    match_ar = set()
    for s in split_smart:
        # print(s)
        match_i = mol.GetSubstructMatches(Chem.MolFromSmarts(s))
        match_s = set()
        for i in match_i:
            # 芳香环可能有重合原子或键
            if not set(i).intersection(set(ar_list)):
                match_s.add(i)
            else:
                match_ar.add(i)
        if match_s:
            split_dict[s] = list(match_s)

    # 将重合的split合并。s=o,o=s=o; p=o,o=p=o;
    if '[#16](=[#8])(=[#8])' in split_dict.keys():
        split_dict['[#16]=[#8]'] = get_diff(split_dict['[#16](=[#8])(=[#8])'], split_dict['[#16]=[#8]'])
    if '[#15](=[#8])(=[#8])' in split_dict.keys():
        split_dict['[#15]=[#8]'] = get_diff(split_dict['[#15](=[#8])(=[#8])'], split_dict['[#15]=[#8]'])
    if '[#7](=[#8])(=[#8])' in split_dict.keys():
        split_dict['[#7]=[#8]'] = get_diff(split_dict['[#7](=[#8])(=[#8])'], split_dict['[#7]=[#8]'])
        split_dict['[#7;X3&+0]'] = get_diff(split_dict['[#7](=[#8])(=[#8])'], split_dict['[#7;X3&+0]'])
    if '[#6]=[#7;+]=[#7;-]' in split_dict.keys():
        split_dict['[#6]=[#7]'] = get_diff(split_dict['[#6]=[#7;+]=[#7;-]'], split_dict['[#6]=[#7]'])
        split_dict['[#7]=[#7]'] = get_diff(split_dict['[#6]=[#7;+]=[#7;-]'], split_dict['[#7]=[#7]'])
        #split_dict['[#7;X4+]'] = get_diff(split_dict['[#6]=[#7;+]=[#7;-]'], split_dict['[#7;X4+]'])
        split_dict['[#7;-]'] = get_diff(split_dict['[#6]=[#7;+]=[#7;-]'], split_dict['[#7;-]'])
    if '[#7]=[#7;+]=[#7;-]' in split_dict.keys():
        split_dict['[#7]=[#7]'] = get_diff(split_dict['[#7]=[#7;+]=[#7;-]'], split_dict['[#7]=[#7]'])
        #split_dict['#7;X4+'] = get_diff(split_dict['[#7]=[#7;+]=[#7;-]'], split_dict['#7;X4+'])
        split_dict['[#7;-]'] = get_diff(split_dict['[#7]=[#7;+]=[#7;-]'], split_dict['[#7;-]'])

    #print(ar_rings)
    #print(match_ar)
    # 将与芳香环共轭的原子合并
    for m in match_ar:
        for i, ar in enumerate(ar_rings):
            if set(m).intersection(set(ar)):
                ar_rings[i] = tuple(set(m).union(set(ar)))

    #将具有重合原子的环合并
    # print(ar_rings)
    new_groups = union_group(ar_rings)
    split_dict['AromaticRing'] = new_groups
    return split_dict

'''
s = 'CCOCc1ccc2c(c1)C(O)=C([O-])C(=C1C=CC(=[N+]3CC[NH+](C)CC3)C=C1)O2'
mol = Chem.MolFromSmiles(s)
print(get_split_dict(mol,False))
'''

# mol must be RWMol object # 将提取出来的frag_mol的原子序号设为原来的原子序号

# 如果只有一个原子
def get_sub_mol(mol, sub_atoms):
    Chem.Kekulize(mol, clearAromaticFlags=True)
    new_mol = Chem.RWMol()
    atom_map = {}
    for idx in sub_atoms:
        atom = mol.GetAtomWithIdx(idx)
        atom_map[idx] = new_mol.AddAtom(atom)  # 返回新加原子的序号
        new_atom = new_mol.GetAtomWithIdx(atom_map[idx])
        new_atom.SetUnsignedProp('mol_idx', atom.GetIdx())

    sub_atoms = set(sub_atoms)
    bond_i = -1
    for idx in sub_atoms:
        a = mol.GetAtomWithIdx(idx)
        for b in a.GetNeighbors():
            if b.GetIdx() not in sub_atoms: continue
            bond = mol.GetBondBetweenAtoms(a.GetIdx(), b.GetIdx())
            bt = bond.GetBondType()
            # print(bt)
            add_bond = []
            if a.GetIdx() < b.GetIdx():  # each bond is enumerated twice
                new_mol.AddBond(atom_map[a.GetIdx()], atom_map[b.GetIdx()], bt)
                bond_i = bond_i + 1
                add_bond = new_mol.GetBondWithIdx(bond_i)

                # begin_atom,end_atom = add_bond.GetBeginAtom(),add_bond.GetEndAtom()
                # begin_atom.SetUnsignedProp('mol_idx', a.GetIdx())
                # end_atom.SetUnsignedProp('mol_idx', b.GetIdx())

    #print(Chem.MolToSmiles(new_mol))

    #print(Chem.MolToSmiles(new_mol))
    sub_mol = new_mol.GetMol()
    Chem.Kekulize(sub_mol,True)


    #Chem.SanitizeMol(sub_mol)

    return sub_mol




get_nei_list = lambda idx, mol: [nei.GetIdx() for nei in mol.GetAtomWithIdx(idx).GetNeighbors()]


def get_next_idxs(before_idxs, cur_idxs, mol):
    inter_dict = {}
    for id in cur_idxs:
        next_idx = get_nei_list(id, mol)
        inter = set(next_idx).difference(set(cur_idxs))
        inter = inter.difference(before_idxs)
        if len(inter) > 0:
            inter_dict[id] = inter
    return inter_dict


# 'C=[N+]=[N-]', 'N=[N+]=[N-]'
def get_frags(mol, ar_rings=True):
    split = get_split_dict(mol,ar_rings=ar_rings)
    # print(split)
    split_set = set(get_list(get_list(split.values())))
    idx_set = set([n for n in range(mol.GetNumAtoms())])
    frag_set = idx_set.difference(set(split_set))

    frags = set()
    before_idxs = {-1}
    split_frags = get_list(split.values())
    # print(split_frags)
    # 所有frag都连在split_frag两边
    for cur_idxs in split_frags:
        next_dict = get_next_idxs(before_idxs, cur_idxs, mol)  # 找到split_frags的下一个节点
        inter = set(get_list(next_dict.values())).difference(split_set)  # 删除split_frag部分,作为连接点

        for next_inter in inter:  # 连接点可能有好几个，分别遍历
            n_inter = {next_inter}
            while True:  # 找到连接点后，循环直到两个split之间的frag全被找出来为止
                next_dict = get_next_idxs(cur_idxs, n_inter, mol)  # 找到连接点的下一个节点
                next_idxs = set(get_list(next_dict.values())).difference(split_set)  # 删除split_frag部分
                # print(next_inter)
                if len(next_idxs) != 0:  # 如果下一个节点有不在split的部分，说明该节点应该归属于frag部分
                    n_inter = n_inter.union(next_idxs)
                else:  # 如果下一个节点全是split，说明所有哦属于该frag的部分都被找到
                    n_inter = tuple(sorted(list(n_inter)))
                    frags.add(n_inter)
                    frag_set = frag_set.difference(set(n_inter))
                    break  # 遍历split_frag下一个连接点

    if len(split_frags) == 0:
        frags = [tuple(idx_set)]
        frag_set = []
    # assert len(frag_set) == 0, print(get_smiles(mol), frag_set, 'undistributed')

    if len(frag_set) > 0:
        frags.add(tuple(frag_set))

    x = get_list(split_frags + list(frags))
    groups = split_frags + list(frags)
    groups = union_group(groups)

    if set(x) != idx_set:
        print(get_smiles(mol), frag_set, 'mol_split error')
    # print(split_frags, list(frags))


    return groups

'''
s = 'O=C2\C(=C1\OC=NN1)C=CC=C2'
s = standardize_smi(s,basicClean=False,clearCharge=False, clearFrag=False)
print(s)
s = sanitize(s)
print(s)
mol = get_mol(s)
print(get_ar_rings(mol))
'''


def get_index(a, l):
    index = []
    x, y = a
    b = y, x
    for i, x in enumerate(l):
        if x == a or x == b:
            index.append(i)
    return index


def find_clusters(mol, ar_rings=True):
    cluster = get_frags(mol,ar_rings=ar_rings)  # mol_graph 保存的是键，因此原子会被重复，每个原子只属于一个cluster
    '''
    cluster = {}
    for i in node[0]: cluster[i] = 'split_frag'
    for j in node[1]: cluster[j] = 'frag'
    '''
    atom_cls = [i for i in range(mol.GetNumAtoms())]
    for i in atom_cls:
        for j, x in enumerate(cluster):
            if i in x:
                atom_cls[i] = j
    return cluster, atom_cls


# 找到所有连接对
def get_inter(mol, ar_rings=True):
    inter_pair = set()  # inter_pair去重
    cluster, atom_cls = find_clusters(mol, ar_rings)
    for i in cluster:
        inter = get_next_idxs({-1}, i, mol)
        for k in inter.items():
            for j in k[1]:
                c = tuple(sorted([k[0], j]))
                inter_pair.add(c)
    inter_pair = list(inter_pair)  # 将集合变为列表

    '''
    for i, j in inter_pair:
        cls_i, cls_j = atom_cls[i], atom_cls[j]
        inter_cls.append(tuple([cls_i, cls_j]))
    '''
    return inter_pair






#f = 'J:/hgraph2graph-master/hgraph2graph-master/dataset/dataset/chembl/chembl.txt'
#l = csv_2_list(f)[1:26]
#mol_list =  [get_mol(s) for s in l]
#draw_mol(mol_list)

#m = Chem.MolFromSmiles('CCOc1ccc2ccccc2c1C(=O)NC1C(=O)N2C1SC(C)(C)C2C(=O)O')
#draw_frag_mol(m,'a.png')
#draw_rdmol(m,name='a.png')
'''
split_frags, frags = get_frags(m)
sub_mol_list = [get_sub_mol(m,s) for i, s in enumerate(frags)]

for i, s in enumerate(sub_mol_list):
    print(get_smiles(s))
    print(s)
'''

def atom_equal(a1, a2):
    return a1.GetSymbol() == a2.GetSymbol() and a1.GetFormalCharge() == a2.GetFormalCharge()



'''
#root =  'F:/bioinfor/hgraph2graph-master-0/hgraph2graph-master/data'
#root = 'J:/fragment-based-dgm-master/utils/data'
root = 'J:/hgraph2graph-master/hgraph2graph-master/dataset'
task_list = ['np']
#task_list = ['np','chembl','zinc']
for task in task_list:
    vocab_df = pd.read_csv(os.path.join(root,task, 'vocab.csv'),header=0)
    print(len(vocab_df))
    atoms_df = count_vocab(vocab_df)
    print(len(atoms_df))
    atoms_df.to_csv(os.path.join(root,task +'_vocab_compute.csv'),index=False)
'''


def unimolrep_nxgraph_2_mol(unimolrep: nx.Graph):
    """
    UniMolRep(nx.Graph) 重构原始RDKit分子
    unimolrep: nx.Graph
        node attr: sub_mol -> Chem.Mol，基团子分子
        edge attr: atom_pair -> tuple(local_atom_u, local_atom_v)
                    local_atom_u：u节点sub_mol内部原子idx
                    local_atom_v：v节点sub_mol内部原子idx
    return: Chem.Mol
    """
    merged_mol = Chem.RWMol()
    # 取出所有基团节点列表
    node_list = list(unimolrep.nodes)

    # Step1：依次合并所有基团sub_mol，给原子打 node_index / atom_id 标记
    for node_id in node_list:
        sub_mol = unimolrep.nodes[node_id]["sub_mol"]
        # 给当前基团所有原子设置属性
        for atom in sub_mol.GetAtoms():
            atom.SetUnsignedProp("node_idx", node_id)
            atom.SetUnsignedProp("atom_id", int(atom.GetProp("mol_idx")))  #该原子在原分子中的编号
        merged_mol = Chem.CombineMols(merged_mol, sub_mol)

    ed_merged_mol = Chem.EditableMol(merged_mol)
    atom_num = merged_mol.GetNumAtoms()

    # Step2：遍历nx.Graph所有边，添加基团之间的连接键
    for u, v, edge_attr in unimolrep.edges(data=True):
        inter_u, inter_v = edge_attr["inter_pair"]
        # 找到atom_id分别是inter_u, inter_v；node_idx分别是u,v 的两个原子
        for n in range(atom_num):
            atom_n = merged_mol.GetAtomWithIdx(n)
            if int(atom_n.GetProp("node_idx")) == u and int(atom_n.GetProp("atom_id")) == inter_u:
                # 向后搜索v基团的连接原子j
                for j in range(atom_num - 1, n, -1):
                    atom_j = merged_mol.GetAtomWithIdx(j)
                    if int(atom_j.GetProp("node_index")) == v and int(atom_j.GetProp("atom_id")) == inter_v:
                        ed_merged_mol.AddBond(n, j, order=Chem.rdchem.BondType.SINGLE)

    final_mol = ed_merged_mol.GetMol()
    return final_mol








def group_graph_2_mol(frag_list,edges_list,inter_list,Vocab):
    # frag_list data.x.tolist(), edges_list= data.edge_index.tolist()
    merged_mol = Chem.RWMol()
    for i, n in enumerate(frag_list):
        mol_i =  Vocab.get_mol(n)
        for atom in mol_i.GetAtoms():
            atom.SetUnsignedProp('frag_id', n)
            atom.SetUnsignedProp('node_index', i)
            atom.SetUnsignedProp('atom_id', atom.GetIdx())
        merged_mol = Chem.CombineMols(merged_mol,mol_i)

    ed_merged_mol = Chem.EditableMol(merged_mol)
    #draw_mol(ed_merged_mol, 'ed_merged_mol.png')
    for i,e in enumerate(edges_list):
        u,v = e[0],e[1]
        #print(u, v)
        inter_u,inter_v = inter_list[i][0],inter_list[i][1]
        atom_num = merged_mol.GetNumAtoms()

        for n in range(atom_num):
            #print(i, u,v)
            atom_n = merged_mol.GetAtomWithIdx(n)
            #print(type(atom_n.GetProp('node_index')),atom_n.GetProp('atom_id') == inter_u, u, inter_u)
            if int(atom_n.GetProp('node_index')) == u and int(atom_n.GetProp('atom_id')) == inter_u:
                #print(n)
                for j in range(atom_num-1,n,-1):
                    #print(j)
                    atom_j = merged_mol.GetAtomWithIdx(j)
                    if int(atom_j.GetProp('node_index')) == v and int(atom_j.GetProp('atom_id')) == inter_v:
                        #print(n,j)
                        ed_merged_mol.AddBond(n, j, order=Chem.rdchem.BondType.SINGLE)

    final_mol = ed_merged_mol.GetMol()
    #final_mol = Chem.SanitizeMol(final_mol)
    #Chem.Kekulize(final_mol, False)
    #print(final_mol)
    #smiles = get_smiles(final_mol)

    #smiles = sanitize(smiles)
    return final_mol

# Alias-Method采样, 给定每个位置出现的概率，随机其中一个位置，以不等概率生成随机数
def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


# http://shomy.top/2017/05/09/alias-method-sampling/
def alias_draw(J, q):
    '''
    输入: Prob数组和Alias数组
    输出: 一次采样结果
    '''
    K = len(J)
    # Draw from the overall uniform mixture.
    kk = int(np.floor(npr.rand() * K))  # 随机取一列

    # Draw from the binary mixture, either keeping the
    # small one, or choosing the associated larger one.
    if npr.rand() < q[kk]:  # 比较
        return kk
    else:
        return J[kk]







'''
if __name__ == "__main__":
    #smi_list = csv_2_list('F:/bioinfor/hgraph2graph-master/hgraph2graph-master/data/np/all_remove_chirality.txt')[100:200]
    smi_list = csv_2_list('./data/np/np_remove_chirality.txt')
    #all_frags = pd.DataFrame(columns=['smiles', 'split_frags', 'split_smi', 'split_atom', 'frags', 'frags_smi', 'frags_atom'])
    ncpu = 8
    batch_size = len(smi_list) // ncpu + 1
    batches = [smi_list[i: i + batch_size] for i in range(0, len(smi_list), batch_size)]
    pool = Pool(ncpu)
    vocab_list = pool.map(process, batches)
    frags_vocab = []
    split_vocab = []
    for vocab in vocab_list:
        frags_vocab = frags_vocab + vocab[1]
        split_vocab = split_vocab + vocab[0]

    frags_vocab = Counter(frags_vocab)
    frags_vocab = sorted(frags_vocab.items(), key=lambda d: d[1], reverse=True)
    #print(frags_vocab)
    split_vocab = Counter(split_vocab)
    split_vocab = sorted(split_vocab.items(), key=lambda d: d[1], reverse=True)
    print(len(frags_vocab), len(split_vocab))
    #frags_vocab = get_list(all_frags['frags_smi'])
    #atom_num = get_list(all_frags['frags_atom'])
    #frags_vocab = {x: atom_num[i] for i, x in enumerate(frags_vocab)}
    #frags_vocab = sorted(frags_vocab.items(), key=lambda d: d[1], reverse=False)
    list_2_csv(frags_vocab, './data/np/frags_vocab.txt')
    #list_2_csv(frags_vocab, 'F:/bioinfor/hgraph2graph-master/hgraph2graph-master/data/np/frags_vocab.txt')

    #split_vocab = get_list(all_frags['split_smi'])
    #atom_num = get_list(all_frags['split_atom'])
    #split_vocab = {x: atom_num[i] for i, x in enumerate(split_vocab)}
    #split_vocab = sorted(split_vocab.items(), key=lambda d: d[1], reverse=False)
    list_2_csv(split_vocab, './data/np/split_vocab.txt')
    #list_2_csv(split_vocab, 'F:/bioinfor/hgraph2graph-master/hgraph2graph-master/data/np/split_vocab.txt')
    #all_frags.to_csv('./data/frags_df.csv')
'''