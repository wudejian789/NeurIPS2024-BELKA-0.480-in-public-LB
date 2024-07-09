import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import Counter
import torch,random,os,jieba,re
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from itertools import permutations
from rdkit.Chem import AllChem
from rdkit import Chem
import deepchem
from deepchem.models.graph_models import GraphConvModel
from deepchem.feat import graph_features
from rdkit import DataStructs
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSeq2SeqLM
from rdkit.Chem import Lipinski,Descriptors
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import AllChem
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem.AtomPairs import Torsions
from rdkit.Chem.AtomPairs import Pairs
from tools import *

import logging,pickle,gc
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from typing import Union

# --- UTILITY FUNCTIONS ---
def string_to_sequence(s: str, dtype=np.int32) -> np.ndarray:
    return np.array([ord(c) for c in s], dtype=dtype)

def sequence_to_string(seq: np.ndarray) -> str:
    return ''.join([chr(c) for c in seq])

def pack_sequences(seqs: Union[np.ndarray, list]) -> (np.ndarray, np.ndarray):
    values = np.concatenate(seqs, axis=0)
    offsets = np.cumsum([len(s) for s in tqdm(seqs)])
    return values, offsets

def unpack_sequence(values: np.ndarray, offsets: np.ndarray, index: int) -> np.ndarray:
    off1 = offsets[index]
    if index > 0:
        off0 = offsets[index - 1]
    elif index == 0:
        off0 = 0
    else:
        raise ValueError(index)
    return values[off0:off1]

class PubChemDB(Dataset):
    def __init__(self, path, samples=None):
        self.path = path
        print('Loading SMILES from PubChemDB...')
        with open(path, 'r') as f:
            if isinstance(samples, int):
                lines = [i.strip().split() for i in tqdm(f.readlines()[:samples])]
            elif isinstance(samples, list):
                tmp = f.readlines()
                lines = [tmp[i].strip().split() for i in tqdm(samples)]
            else:
                lines = [i.strip().split() for i in tqdm(f.readlines())]
        print('string to sequencing...')
        self.cids,smiles = np.array([i[0] for i in lines], dtype=np.int32),[string_to_sequence(i[1]) for i in tqdm(lines)]
        print('packing...')
        self.smiles_v,self.smiles_o = pack_sequences(smiles)
    def __len__(self):
        return len(self.cids)
    def __getitem__(self, index):
        smiles = sequence_to_string(unpack_sequence(self.smiles_v, self.smiles_o, index))
        
        mol = Chem.MolFromSmiles(smiles) if len(smiles)>0 else None
        try:
            smi = Chem.MolToSmiles(mol, doRandom=True, canonical=False, isomericSmiles=True) if mol is not None else ""
            mol = Chem.MolFromSmiles(smi) if len(smi)>0 else None
        except:
            smi = smiles
            gc.collect()
        return {'cid':self.cids[index],      \
                'smiles':smi, \
                'mol':mol}

class BELKA(Dataset):
    def __init__(self, filePath, protein_name, predict=False):
        self.filePath = filePath
        if filePath.endswith('.csv'):
            df_ = pd.read_csv(filePath)
        elif filePath.endswith('.parquet'):
            import duckdb
            con = duckdb.connect()
            df_ = df = con.query(f"""(SELECT *
                                        FROM parquet_scan('{filePath}')
                                      )""").df()
            con.close()
            
        if protein_name is None:
            if predict:
                df_['binds'] = -1
            df1 = df_[['id','buildingblock1_smiles','buildingblock2_smiles','buildingblock3_smiles','molecule_smiles','binds']].loc[df_['protein_name']=='BRD4'].reset_index(drop=True)
            df2 = df_[['id','buildingblock1_smiles','buildingblock2_smiles','buildingblock3_smiles','molecule_smiles','binds']].loc[df_['protein_name']=='HSA'].reset_index(drop=True)
            df3 = df_[['id','buildingblock1_smiles','buildingblock2_smiles','buildingblock3_smiles','molecule_smiles','binds']].loc[df_['protein_name']=='sEH'].reset_index(drop=True)
            df1 = df1.rename(columns={'binds':'bind1','id':'id1'})
            df2 = df2.rename(columns={'binds':'bind2','id':'id2'})
            df3 = df3.rename(columns={'binds':'bind3','id':'id3'})
            
            df = pd.merge(df1,df2, on=['buildingblock1_smiles','buildingblock2_smiles','buildingblock3_smiles','molecule_smiles'], how='outer')
            df = pd.merge(df, df3, on=['buildingblock1_smiles','buildingblock2_smiles','buildingblock3_smiles','molecule_smiles'], how='outer')

            df['id1'],df['id2'],df['id3'] = df['id1'].fillna(-1),df['id2'].fillna(-1),df['id3'].fillna(-1)
            df['bind1'],df['bind2'],df['bind3'] = df['bind1'].fillna(-1),df['bind2'].fillna(-1),df['bind3'].fillna(-1)
            self.ids = np.array([list(i) for i in zip(df['id1'].tolist(),df['id2'].tolist(),df['id3'].tolist())], dtype=np.int32)
            self.Y = np.array([list(i) for i in zip(df['bind1'].tolist(),df['bind2'].tolist(),df['bind3'].tolist())], dtype=np.int32)
        else:
            df = df_.loc[df_['protein_name']==protein_name].reset_index(drop=True)
            df['id'] = df['id'].fillna(-1)
            self.ids = np.array([[i] for i in df['id']], dtype=np.int32)
            self.Y = np.array([[i] for i in df['binds']], dtype=np.int32)

        synsmiles = [string_to_sequence(smi) for smi in df['molecule_smiles'].tolist()]
        self.synsmiles_v,self.synsmiles_o = pack_sequences(synsmiles)
        
        self.id2smiles = sorted(list(set(df['buildingblock1_smiles'].tolist()+df['buildingblock2_smiles'].tolist()+df['buildingblock3_smiles'].tolist())))
        self.smiles2id = {v:k for k,v in enumerate(self.id2smiles)}
        self.id2mol = np.array([Chem.MolFromSmiles(i) for i in tqdm(self.id2smiles)])
        
        self.id2cansmiles = np.array([Chem.MolToSmiles(i, doRandom=False, canonical=True, isomericSmiles=True) for i in tqdm(self.id2mol)])
        self.id2cansmi2molOrder = [[int(j) for j in i.GetProp('_smilesAtomOutputOrder')[1:-2].split(',')] for i in tqdm(self.id2mol)]

        self.id2dFeaData = [[graph_features.atom_features(j) for j in i.GetAtoms()] for i in tqdm(self.id2mol)]
        self.id2dFinData = []
        for i in tqdm(self.id2mol):
            tmp = np.ones((1,))
            DataStructs.ConvertToNumpyArray(AllChem.GetHashedMorganFingerprint(i,2,nBits=2048), tmp)
            self.id2dFinData.append(tmp)
        self.id2dFinData = np.array(self.id2dFinData, dtype=np.float32)

        self.sidList1 = np.array(df['buildingblock1_smiles'].map(self.smiles2id).values, dtype=np.int32)
        self.sidList2 = np.array(df['buildingblock2_smiles'].map(self.smiles2id).values, dtype=np.int32)
        self.sidList3 = np.array(df['buildingblock3_smiles'].map(self.smiles2id).values, dtype=np.int32)

    def __len__(self):
        return len(self.Y)
    def __getitem__(self, index):
        synsmiles = sequence_to_string(unpack_sequence(self.synsmiles_v, self.synsmiles_o, index))
        
        return {'index':index,
                'synsmiles':synsmiles,
                
                'mol1':self.id2mol[self.sidList1[index]],
                'mol2':self.id2mol[self.sidList2[index]],
                'mol3':self.id2mol[self.sidList3[index]],

                'smiles1':self.id2smiles[self.sidList1[index]],
                'smiles2':self.id2smiles[self.sidList2[index]],
                'smiles3':self.id2smiles[self.sidList3[index]],

                'cansmiles1':self.id2cansmiles[self.sidList1[index]],
                'cansmiles2':self.id2cansmiles[self.sidList2[index]],
                'cansmiles3':self.id2cansmiles[self.sidList3[index]],
                
                'cansmi2molOrder1':self.id2cansmi2molOrder[self.sidList1[index]],
                'cansmi2molOrder2':self.id2cansmi2molOrder[self.sidList2[index]],
                'cansmi2molOrder3':self.id2cansmi2molOrder[self.sidList3[index]],

                'dFeaData1':self.id2dFeaData[self.sidList1[index]],
                'dFeaData2':self.id2dFeaData[self.sidList2[index]],
                'dFeaData3':self.id2dFeaData[self.sidList3[index]],

                'dFinData1':self.id2dFinData[self.sidList1[index]],
                'dFinData2':self.id2dFinData[self.sidList2[index]],
                'dFinData3':self.id2dFinData[self.sidList3[index]],
                
                'y':self.Y[index]}

class HuggingfaceNoisingCollateFunc_final_forBELKA: # for GPT
    def __init__(self, bertDir, seqMaxLen, randomSMILES=True):
        self.tokenizer = AutoTokenizer.from_pretrained(bertDir, trust_remote_code=True, do_lower_case=False)
        self.tknList = list(self.tokenizer.get_vocab().keys())
        self.seqMaxLen = seqMaxLen
        self.train = False
        self.randomSMILES = randomSMILES

        self.restorePass = False

    def __call__(self, data):
        if self.restorePass:
            return None

        if self.train and self.randomSMILES:
            synsmilesList = []
            smilesList1,smilesList2,smilesList3 = [],[],[]
            for i in data:
                mol = Chem.MolFromSmiles(i['synsmiles'])
                smi = Chem.MolToSmiles(mol, doRandom=True, canonical=False, isomericSmiles=True)
                synsmilesList.append(smi)
                
                try:
                    smi = Chem.MolToSmiles(i['mol1'], doRandom=True, canonical=False, isomericSmiles=True)
                except:
                    smi = i['smiles1']
                smilesList1.append(smi)
                
                try:
                    smi = Chem.MolToSmiles(i['mol2'], doRandom=True, canonical=False, isomericSmiles=True)
                except:
                    smi = i['smiles2']
                smilesList2.append(smi)
                
                try:
                    smi = Chem.MolToSmiles(i['mol3'], doRandom=True, canonical=False, isomericSmiles=True)
                except:
                    smi = i['smiles3']
                smilesList3.append(smi)
        else:
            synsmilesList = [i['synsmiles'] for i in data]
            smilesList1,smilesList2,smilesList3 = [i['cansmiles1'] for i in data],[i['cansmiles2'] for i in data],[i['cansmiles3'] for i in data]

        sequence,label = self.prepare_nosed_seq2seq_batch(synsmilesList, smilesList1,smilesList2,smilesList3)

        # if self.train and self.randomSMILES:
        #     padding = 'longest'
        # else:
        #     padding = 'max_length'
        padding = 'longest'

        batch = self.tokenizer(sequence, return_tensors='pt', max_length=self.seqMaxLen, padding=padding, truncation=True)
        label = self.tokenizer(label, return_tensors='pt', max_length=self.seqMaxLen, padding=padding, truncation=True)
        batch['labels'] = label['input_ids']

        if 'token_type_ids' in batch:
            batch.pop('token_type_ids')

        return {'batch':batch, 'sequence':sequence}

    def prepare_nosed_seq2seq_batch(self, synsmilesList, smilesList1,smilesList2,smilesList3):
        sourceList,labelList = [],[]

        for idx in range(len(synsmilesList)):
            t = random.random()
            sequence = (" ".join(list(synsmilesList[idx])) + " [SEP] " + \
                        " ".join(list(smilesList1[idx])) + " [SEP] " + \
                        " ".join(list(smilesList2[idx])) + " [SEP] " + \
                        " ".join(list(smilesList3[idx]))).split(' ')

            if t<1/3: # mlm
                mask = [False if random.random()>0.15 else True for i in sequence]
                source = ['[MASK]' if m else i for m,i in zip(mask,sequence)]
                target = [i for m,i in zip(mask,sequence) if m]
            elif t<2/3: # glm
                if random.random()<0.3: # random [SPAN] mask
                    s,e = random.sample(range(len(sequence)+1), 2)
                    if s>e: s,e = e,s
                    source = sequence[:s] + ['[SPAN]'] + sequence[e:]
                    target = sequence[s:e]
                else: # whole entity [SPAN] mask
                    t = random.choice([0,1,2,3])
                    sequence = (" ".join(sequence)).split(' [SEP] ')
                    mask = [False if random.random()>0.15 else True for i in sequence]
                    if sum(mask)==0 or (sum(mask)==1 and mask[t]==True):
                        mask[random.choice([i for i in [0,1,2,3] if i!=t])] = True
                    source = ['[SPAN]' if (m or loc==t) else i for loc,(m,i) in enumerate(zip(mask,sequence))]
                    target = [i for loc,(m,i) in enumerate(zip(mask,sequence)) if m and loc!=t]
            else: # plm
                if random.random()<0.3: # random shuffle
                    s,e = random.sample(range(len(sequence)+1), 2)
                    if s>e: s,e = e,s
                    target = sequence
                    
                    tmp = sequence[s:e]
                    random.shuffle(tmp)
                    source = sequence[:s] + tmp + sequence[e:]
                else: # whole entity shuffle
                    target = sequence

                    t = random.choice([0,1,2,3])
                    sequence = [i.split(' ') for i in (" ".join(sequence)).split(' [SEP] ')]
                    for i in range(4):
                        if i==t or random.random()<0.15:
                            random.shuffle(sequence[i])
                    source = (" [SEP] ".join([" ".join(i) for i in sequence])).split(' ')

            source = " ".join(source+['[SOS]']+target)
            s,t = source.split(' [SOS] ')
            label = " ".join(['[PAD]' for i in s.split(' ')]+['[PAD]']) + " ".join(target)
            sourceList.append(source)
            labelList.append(label)

        return sourceList,labelList

class FinetuneCollateFunc_final_forBELKA:
    def __init__(self, bertDir, seqMaxLen, prompt, randomSMILES=True):
        self.tokenizer = AutoTokenizer.from_pretrained(bertDir, trust_remote_code=True, do_lower_case=False)
        self.tokenizer.padding_side = 'left'
        self.seqMaxLen = seqMaxLen
        self.prompt = prompt
        self.randomSMILES = randomSMILES
        self.train = False
    def __call__(self, data):
        if self.train and self.randomSMILES:
            synsmilesList = []
            smilesList1,smilesList2,smilesList3 = [],[],[]
            for i in data:
                mol = Chem.MolFromSmiles(i['synsmiles'])
                smi = Chem.MolToSmiles(mol, doRandom=True, canonical=False, isomericSmiles=True)
                synsmilesList.append(smi)
                
                try:
                    smi = Chem.MolToSmiles(i['mol1'], doRandom=True, canonical=False, isomericSmiles=True)
                except:
                    smi = i['smiles1']
                smilesList1.append(smi)
                
                try:
                    smi = Chem.MolToSmiles(i['mol2'], doRandom=True, canonical=False, isomericSmiles=True)
                except:
                    smi = i['smiles2']
                smilesList2.append(smi)
                
                try:
                    smi = Chem.MolToSmiles(i['mol3'], doRandom=True, canonical=False, isomericSmiles=True)
                except:
                    smi = i['smiles3']
                smilesList3.append(smi)
        else:
            synsmilesList = [i['synsmiles'] for i in data]
            smilesList1,smilesList2,smilesList3 = [i['cansmiles1'] for i in data],[i['cansmiles2'] for i in data],[i['cansmiles3'] for i in data]

        source = [self.prompt.replace('[SMILES]', " ".join(list(synsmi)+['[SEP]']+list(smi1)+['[SEP]']+list(smi2)+['[SEP]']+list(smi3))) for synsmi,smi1,smi2,smi3 in zip(synsmilesList,smilesList1,smilesList2,smilesList3)]
        
        # if self.train:
        #     padding = 'longest'
        # else:
        #     padding = 'max_length'
        padding = 'longest'

        sequence = source
        batch = self.tokenizer(sequence, return_tensors='pt', max_length=self.seqMaxLen, padding=padding, truncation=True)
        if 'token_type_ids' in batch:
            batch.pop('token_type_ids')

        y = np.array([i['y'] for i in data], dtype=np.float32)
        
        res = {'batch':batch, 'y':torch.tensor(y, dtype=torch.float32)}
            
        return res

class DTICollateFunc_forBELKA:
    def __init__(self, bertDir, seqMaxLen, randomSMILES=False, removeDy=False,
                 useFP=False, padding_longest=False):
        self.tokenizer = AutoTokenizer.from_pretrained(bertDir, trust_remote_code=True, do_lower_case=False)
        
        self.seqMaxLen = seqMaxLen
        # self.prompt = f"[sPPM] [SMILES] [SEP] [CUSPRO] [VALUE] ; [SOS]"
        self.prompt = "[SMILES]"
        self.randomSMILES = randomSMILES
        self.train = False
        self.removeDy = removeDy
        self.useFP = useFP

        self.padding_longest = padding_longest

    def __call__(self, data):
        if self.train and self.randomSMILES:
            synsmilesList,synsmi2molOrder = [],[]
            molList = []
            smilesList1,smilesList2,smilesList3 = [],[],[]
            smi2molOrder1,smi2molOrder2,smi2molOrder3 = [],[],[]
            for i in data:
                if self.removeDy:
                    mol = Chem.MolFromSmiles(i['synsmiles'].replace('[Dy]',''))
                else:
                    mol = Chem.MolFromSmiles(i['synsmiles'])
                smi = Chem.MolToSmiles(mol, doRandom=True, canonical=False, isomericSmiles=True)
                s2mo = [int(j) for j in mol.GetProp('_smilesAtomOutputOrder')[1:-2].split(',')]
                molList.append(mol)
                synsmilesList.append(smi)
                synsmi2molOrder.append(s2mo)
                
                try:
                    smi = Chem.MolToSmiles(i['mol1'], doRandom=True, canonical=False, isomericSmiles=True)
                    s2mo = [int(j) for j in i['mol1'].GetProp('_smilesAtomOutputOrder')[1:-2].split(',')]
                except:
                    smi = i['smiles1']
                    s2mo = i['smi2molOrder1']
                smilesList1.append(smi)
                smi2molOrder1.append(s2mo)
                
                try:
                    smi = Chem.MolToSmiles(i['mol2'], doRandom=True, canonical=False, isomericSmiles=True)
                    s2mo = [int(j) for j in i['mol2'].GetProp('_smilesAtomOutputOrder')[1:-2].split(',')]
                except:
                    smi = i['smiles2']
                    s2mo = i['smi2molOrder2']
                smilesList2.append(smi)
                smi2molOrder2.append(s2mo)
                
                try:
                    smi = Chem.MolToSmiles(i['mol3'], doRandom=True, canonical=False, isomericSmiles=True)
                    s2mo = [int(j) for j in i['mol3'].GetProp('_smilesAtomOutputOrder')[1:-2].split(',')]
                except:
                    smi = i['smiles3']
                    s2mo = i['smi2molOrder3']
                smilesList3.append(smi)
                smi2molOrder3.append(s2mo)
        else:
            molList = []
            synsmilesList,synsmi2molOrder = [],[]
            for i in data:
                if self.removeDy:
                    mol = Chem.MolFromSmiles(i['synsmiles'].replace('[Dy]',''))
                else:
                    mol = Chem.MolFromSmiles(i['synsmiles'])
                smi = Chem.MolToSmiles(mol, doRandom=False, canonical=True, isomericSmiles=True)
                s2mo = [int(j) for j in mol.GetProp('_smilesAtomOutputOrder')[1:-2].split(',')]
                molList.append(mol)
                synsmilesList.append(smi)
                synsmi2molOrder.append(s2mo)
                
            smilesList1,smilesList2,smilesList3 = [i['cansmiles1'] for i in data],[i['cansmiles2'] for i in data],[i['cansmiles3'] for i in data]
            smi2molOrder1,smi2molOrder2,smi2molOrder3 = [i['cansmi2molOrder1'] for i in data],[i['cansmi2molOrder2'] for i in data],[i['cansmi2molOrder3'] for i in data]
        
        # if self.train or self.padding_longest:
        #     padding = 'longest'
        # else:
        #     padding = 'max_length'
        padding = 'longest'
        
        source = [self.prompt.replace('[SMILES]', " ".join(list(smi))) for smi in synsmilesList]
        # if isinstance(data[0]['y'], list):
        #     target = ["[SEP] " + " ".join(sum([list(f"{v:.3f}")+[';'] for v in i['y']],[])) + " [EOS]" for i in data]
        # else:
        #     target = [" ".join(['[SEP]']+list(f"{i['y']:.3f}")+[';','[EOS]']) for i in data]
        sequence = source
        batch = self.tokenizer(sequence, return_tensors='pt', max_length=self.seqMaxLen*3, padding=padding, truncation=True)
        if 'token_type_ids' in batch:
            batch.pop('token_type_ids')

        source = [self.prompt.replace('[SMILES]', " ".join(list(smi))) for smi in smilesList1]
        sequence = source
        batch1 = self.tokenizer(sequence, return_tensors='pt', max_length=self.seqMaxLen, padding=padding, truncation=True)
        if 'token_type_ids' in batch1:
            batch1.pop('token_type_ids')

        source = [self.prompt.replace('[SMILES]', " ".join(list(smi))) for smi in smilesList2]
        sequence = source
        batch2 = self.tokenizer(sequence, return_tensors='pt', max_length=self.seqMaxLen, padding=padding, truncation=True)
        if 'token_type_ids' in batch2:
            batch2.pop('token_type_ids')

        source = [self.prompt.replace('[SMILES]', " ".join(list(smi))) for smi in smilesList3]
        sequence = source
        batch3 = self.tokenizer(sequence, return_tensors='pt', max_length=self.seqMaxLen, padding=padding, truncation=True)
        if 'token_type_ids' in batch3:
            batch3.pop('token_type_ids')

        y = np.array([i['y'] for i in data], dtype=np.float32)
        
        res =  {'batch':batch, 'batch1':batch1, 'batch2':batch2, 'batch3':batch3, 
                'source':source, # 'target':target,
                'y':torch.tensor(y, dtype=torch.float32)}

        if self.useFP:
            maxAtomNum1 = min(max([len(i['mol1'].GetAtoms()) for i in data]), batch1['input_ids'].shape[1])
            maxAtomNum2 = min(max([len(i['mol2'].GetAtoms()) for i in data]), batch2['input_ids'].shape[1])
            maxAtomNum3 = min(max([len(i['mol3'].GetAtoms()) for i in data]), batch3['input_ids'].shape[1])
            maxAtomNum = min(max([len(i.GetAtoms()) for i in molList]), batch['input_ids'].shape[1])
            
            atom_mask1 = torch.ones((len(data),maxAtomNum1), dtype=bool)
            atom_mask2 = torch.ones((len(data),maxAtomNum2), dtype=bool)
            atom_mask3 = torch.ones((len(data),maxAtomNum3), dtype=bool)
            atom_mask = torch.ones((len(data),maxAtomNum), dtype=bool)

            dFeaData1 = np.zeros((len(data), maxAtomNum1, 75), dtype=np.float32)
            dFinData1 = np.zeros((len(data), 2048), dtype=np.float32)
            dFeaData2 = np.zeros((len(data), maxAtomNum2, 75), dtype=np.float32)
            dFinData2 = np.zeros((len(data), 2048), dtype=np.float32)
            dFeaData3 = np.zeros((len(data), maxAtomNum3, 75), dtype=np.float32)
            dFinData3 = np.zeros((len(data), 2048), dtype=np.float32)
            
            dFeaData = np.zeros((len(data), maxAtomNum, 75), dtype=np.float32)
            dFinData = np.zeros((len(data), 2048), dtype=np.float32)
            
            for idx,i in enumerate(data):

                tmp1 = np.array(i['dFeaData1'])[smi2molOrder1[idx]].tolist()
                dFeaData1[idx,:len(tmp1)] = tmp1
                atom_mask1[idx,:len(tmp1)] = False
                dFinData1[idx] = i['dFinData1']

                tmp2 = np.array(i['dFeaData2'])[smi2molOrder2[idx]].tolist()
                dFeaData2[idx,:len(tmp2)] = tmp2
                atom_mask2[idx,:len(tmp2)] = False
                dFinData2[idx] = i['dFinData2']

                tmp3 = np.array(i['dFeaData3'])[smi2molOrder3[idx]].tolist()
                dFeaData3[idx,:len(tmp3)] = tmp3
                atom_mask3[idx,:len(tmp3)] = False
                dFinData3[idx] = i['dFinData3']

                tmp = [graph_features.atom_features(a) for a in np.array(molList[idx].GetAtoms())[synsmi2molOrder[idx]]][:dFeaData.shape[1]]
                dFeaData[idx,:len(tmp)] = tmp
                atom_mask[idx,:len(tmp)] = False 

                tmp = np.ones((1,))
                DataStructs.ConvertToNumpyArray(AllChem.GetHashedMorganFingerprint(molList[idx],2,nBits=2048), tmp)
                dFinData[idx] = tmp


            res['atomFea1'] = torch.tensor(dFeaData1, dtype=torch.float32)
            res['atomFin1'] = torch.tensor(dFinData1, dtype=torch.float32)
            res['atom_mask1'] = ~atom_mask1

            res['atomFea2'] = torch.tensor(dFeaData2, dtype=torch.float32)
            res['atomFin2'] = torch.tensor(dFinData2, dtype=torch.float32)
            res['atom_mask2'] = ~atom_mask2

            res['atomFea3'] = torch.tensor(dFeaData3, dtype=torch.float32)
            res['atomFin3'] = torch.tensor(dFinData3, dtype=torch.float32)
            res['atom_mask3'] = ~atom_mask3

            res['atomFea'] = torch.tensor(dFeaData, dtype=torch.float32)
            res['atomFin'] = torch.tensor(dFinData, dtype=torch.float32)
            res['atom_mask'] = ~atom_mask

        return res

fpr2func = {'rdk': Chem.RDKFingerprint, # RDK指纹
            'Toplogical': Chem.rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect, # Toplogical指纹
            'MACCS': Chem.rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect, # MACCS指纹
            'AtomPair': Chem.rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect, # AtomPair指纹
            'ECFP4': lambda x:AllChem.GetMorganFingerprintAsBitVect(x, 2), # ECFP4指纹
            'FCFP4': lambda x:AllChem.GetMorganFingerprintAsBitVect(x, 2, useFeatures=True), # FCFP6指纹
            'FCFP6': lambda x:AllChem.GetMorganFingerprintAsBitVect(x, 3, useFeatures=True), # FCFP6指纹
            'Avalon': pyAvalonTools.GetAvalonFP, # Avalon指纹
            'Layered': Chem.rdmolops.LayeredFingerprint, # Layered指纹
            'Pattern': Chem.rdmolops.PatternFingerprint} # Pattern指纹
def get_molecule_vector(smiList, func, convert2Mol=False, post_fn=None, usebar=True, parallel=False, njobs=2, dtype=np.float32):
    if usebar:
        smiList = tqdm(smiList)

    if parallel:
        from joblib import Parallel, delayed
        if convert2Mol:
            molList = Parallel(n_jobs=njobs)(delayed(wrapped_mol_from_smiles)(smi) for smi in smiList)
            if usebar: molList = tqdm(molList, position=0,leave=True)
            tmp = Parallel(n_jobs=njobs)(delayed(func)(mol) for mol in molList)
        else:
            tmp = Parallel(n_jobs=njobs)(delayed(func)(smi) for smi in smiList)
        if post_fn is not None:
            if usebar:
                tmp = tqdm(tmp, position=0,leave=True)
            tmp = Parallel(n_jobs=njobs)(delayed(getattr(i, post_fn))() for i in tmp)
    else:
        if convert2Mol:
            tmp = [func(Chem.MolFromSmiles(smi)) for smi in smiList]
        else:
            tmp = [func(smi) for smi in smiList]
        if post_fn is not None:
            tmp = [getattr(i, post_fn)() for i in tmp]
    return np.array(tmp, dtype=dtype)

import lmdb
def bitstring2uint8(bits):
    n = len(bits)//4
    return np.array([int(bits[i*4:(i+1)*4],2) for i in range(n)], np.uint8)
def uint82bitstring(uint8):
    return "".join([f"{bin(i)[2:]:0>4}" for i in uint8])
class FPRCollateFunc_forBELKA:
    def __init__(self, fpr_list=['ECFP4', 'Toplogical', 'AtomPair', 'FCFP6'], smi2fpr=None, use_lmdb=False, memmap=None, **kwargs):
        self.fpr_list = fpr_list
        self.smi2fpr = smi2fpr
        self.use_lmdb = use_lmdb
        self.memmap = memmap
        if use_lmdb:
            self.memmap_fp = {k:lmdb.open(f'./fpr_cache/{k}_fpr_lmdb', lock=False, readonly=True, **kwargs).begin() for k in fpr_list}
        if memmap is not None:
            self.memmap_fp = {k:np.memmap(f'{memmap[0]}_{k}_fpr.mem', dtype='uint8', mode='r',
                                          shape=(memmap[1], 512 if k!='Avalon' else 128)) for k in fpr_list}

    def __call__(self, data):
        if self.use_lmdb:
            fprArr = []
            for k in self.fpr_list:
                fprArr.append( np.array([list(uint82bitstring(pickle.loads(self.memmap_fp[k].get(i['synsmiles'].encode())))) for i in data], dtype=np.float32) )
            fprArr = np.hstack(fprArr)
        elif self.memmap is not None:
            idxList = [i['index'] for i in data]
            fprArr = np.hstack([np.array([list(uint82bitstring(i)) for i in self.memmap_fp[k][idxList]],dtype=np.float32)\
                                for k in self.fpr_list])
        else:
            molList = [Chem.MolFromSmiles(i['synsmiles'].replace('[Dy]','')) for i in data]
            fprArr = np.hstack([get_molecule_vector(molList, fpr2func[key], post_fn='ToList', usebar=False) for key in self.fpr_list])
        
        if self.smi2fpr is None:
            molList1,molList2,molList3 = [],[],[]
            molList1 = [i['mol1'] for i in data]
            molList2 = [i['mol2'] for i in data]
            molList3 = [i['mol3'] for i in data]
            fprArr1 = np.hstack([get_molecule_vector(molList1, fpr2func[key], post_fn='ToList', usebar=False) for key in self.fpr_list])
            fprArr2 = np.hstack([get_molecule_vector(molList2, fpr2func[key], post_fn='ToList', usebar=False) for key in self.fpr_list])
            fprArr3 = np.hstack([get_molecule_vector(molList3, fpr2func[key], post_fn='ToList', usebar=False) for key in self.fpr_list])
        else:
            fprArr1 = np.array([self.smi2fpr[i['cansmiles1']] for i in data], dtype=np.float32)
            fprArr2 = np.array([self.smi2fpr[i['cansmiles2']] for i in data], dtype=np.float32)
            fprArr3 = np.array([self.smi2fpr[i['cansmiles3']] for i in data], dtype=np.float32)

        y = np.array([i['y'] for i in data], dtype=np.float32)

        return {'fprArr':torch.tensor(fprArr, dtype=torch.float32),
                'fprArr1':torch.tensor(fprArr1, dtype=torch.float32),
                'fprArr2':torch.tensor(fprArr2, dtype=torch.float32),
                'fprArr3':torch.tensor(fprArr3, dtype=torch.float32),
                'y':torch.tensor(y, dtype=torch.float32)}

class FPRCollateFunc2_forBELKA:
    def __init__(self, bertDir, smi2fpr, seqMaxLen=128):
        self.tokenizer = AutoTokenizer.from_pretrained(bertDir, trust_remote_code=True, do_lower_case=False)
        self.smi2fpr = smi2fpr
        self.seqMaxLen = seqMaxLen

    def __call__(self, data):
        source = [" ".join(list(i['synsmiles'])) for i in data]
        batch = self.tokenizer(source, return_tensors='pt', max_length=self.seqMaxLen, padding='longest', truncation=True)
        if 'token_type_ids' in batch:
            batch.pop('token_type_ids')

        fprArr1 = np.array([self.smi2fpr[i['smiles1']] for i in data], dtype=np.float32)
        fprArr2 = np.array([self.smi2fpr[i['smiles2']] for i in data], dtype=np.float32)
        fprArr3 = np.array([self.smi2fpr[i['smiles3']] for i in data], dtype=np.float32)

        y = np.array([i['y'] for i in data], dtype=np.float32)

        return {'batch':batch,
                'fprArr1':torch.tensor(fprArr1, dtype=torch.float32),
                'fprArr2':torch.tensor(fprArr2, dtype=torch.float32),
                'fprArr3':torch.tensor(fprArr3, dtype=torch.float32),
                'y':torch.tensor(y, dtype=torch.float32)}

class GraphCollateFunc_forBELKA:
    def __init__(self, bertDir, seqMaxLen=128, useFP=False):
        self.tokenizer = AutoTokenizer.from_pretrained(bertDir, trust_remote_code=True, do_lower_case=False)
        self.seqMaxLen = seqMaxLen
        self.useFP = useFP

    def __call__(self, data):
        molList = [Chem.MolFromSmiles(i['synsmiles'].replace('[Dy]','')) for i in data]
        molList1 = [i['mol1'] for i in data]
        molList2 = [i['mol2'] for i in data]
        molList3 = [i['mol3'] for i in data]

        # get atom identity matrix
        atomArr = [" ".join([f"[(XYZ)ATOM:{a.GetSymbol()}]" for a in mol.GetAtoms()]) for mol in molList]
        batch = self.tokenizer(atomArr, return_tensors='pt', max_length=self.seqMaxLen*2, padding='longest', truncation=True)
        maxAtomNum = batch['input_ids'].shape[1]

        atomArr1 = [" ".join([f"[(XYZ)ATOM:{a.GetSymbol()}]" for a in mol.GetAtoms()]) for mol in molList1]
        batch1 = self.tokenizer(atomArr1, return_tensors='pt', max_length=self.seqMaxLen, padding='longest', truncation=True)
        maxAtomNum1 = batch1['input_ids'].shape[1]

        atomArr2 = [" ".join([f"[(XYZ)ATOM:{a.GetSymbol()}]" for a in mol.GetAtoms()]) for mol in molList2]
        batch2 = self.tokenizer(atomArr2, return_tensors='pt', max_length=self.seqMaxLen, padding='longest', truncation=True)
        maxAtomNum2 = batch2['input_ids'].shape[1]

        atomArr3 = [" ".join([f"[(XYZ)ATOM:{a.GetSymbol()}]" for a in mol.GetAtoms()]) for mol in molList3]
        batch3 = self.tokenizer(atomArr3, return_tensors='pt', max_length=self.seqMaxLen, padding='longest', truncation=True)
        maxAtomNum3 = batch3['input_ids'].shape[1]

        # get adjacency matrix
        feaArr = np.zeros((len(batch['input_ids']),maxAtomNum,75), dtype=np.float32)
        adjArr = np.zeros((len(batch['input_ids']),maxAtomNum,maxAtomNum), dtype=np.float32)
        for idx,mol in enumerate(molList):
            tmp = [graph_features.atom_features(a) for a in list(mol.GetAtoms())[:adjArr.shape[1]]]
            feaArr[idx,:len(tmp)] = tmp

            adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
            adj = adj + np.eye(len(adj))
            ma = min(adj.shape[0],maxAtomNum)
            adjArr[idx,:ma,:ma] = adj[:ma,:ma]
        D = adjArr.sum(axis=-1,keepdims=True)**(-1/2) # B,L,1
        D[np.isnan(D)|np.isinf(D)] = 0
        D = D*np.eye(maxAtomNum, dtype=np.float32)[None]
        adjArr = D@adjArr@D

        feaArr1 = np.zeros((len(batch1['input_ids']),maxAtomNum1,75), dtype=np.float32)
        adjArr1 = np.zeros((len(batch1['input_ids']),maxAtomNum1,maxAtomNum1), dtype=np.float32)
        for idx,mol in enumerate(molList1):
            tmp = data[idx]['dFeaData1'][:adjArr1.shape[1]]
            feaArr1[idx,:len(tmp)] = tmp

            adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
            adj = adj + np.eye(len(adj))
            ma = min(adj.shape[0],maxAtomNum1)
            adjArr1[idx,:ma,:ma] = adj[:ma,:ma]
        D = adjArr1.sum(axis=-1,keepdims=True)**(-1/2) # B,L,1
        D[np.isnan(D)|np.isinf(D)] = 0
        D = D*np.eye(maxAtomNum1, dtype=np.float32)[None]
        adjArr1 = D@adjArr1@D

        feaArr2 = np.zeros((len(batch2['input_ids']),maxAtomNum2,75), dtype=np.float32)
        adjArr2 = np.zeros((len(batch2['input_ids']),maxAtomNum2,maxAtomNum2), dtype=np.float32)
        for idx,mol in enumerate(molList2):
            tmp = data[idx]['dFeaData2'][:adjArr2.shape[1]]
            feaArr2[idx,:len(tmp)] = tmp

            adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
            adj = adj + np.eye(len(adj))
            ma = min(adj.shape[0],maxAtomNum2)
            adjArr2[idx,:ma,:ma] = adj[:ma,:ma]
        D = adjArr2.sum(axis=-1,keepdims=True)**(-1/2) # B,L,1
        D[np.isnan(D)|np.isinf(D)] = 0
        D = D*np.eye(maxAtomNum2, dtype=np.float32)[None]
        adjArr2 = D@adjArr2@D

        feaArr3 = np.zeros((len(batch3['input_ids']),maxAtomNum3,75), dtype=np.float32)
        adjArr3 = np.zeros((len(batch3['input_ids']),maxAtomNum3,maxAtomNum3), dtype=np.float32)
        for idx,mol in enumerate(molList3):
            tmp = data[idx]['dFeaData3'][:adjArr3.shape[1]]
            feaArr3[idx,:len(tmp)] = tmp

            adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
            adj = adj + np.eye(len(adj))
            ma = min(adj.shape[0],maxAtomNum3)
            adjArr3[idx,:ma,:ma] = adj[:ma,:ma]
        D = adjArr3.sum(axis=-1,keepdims=True)**(-1/2) # B,L,1
        D[np.isnan(D)|np.isinf(D)] = 0
        D = D*np.eye(maxAtomNum3, dtype=np.float32)[None]
        adjArr3 = D@adjArr3@D

        y = np.array([i['y'] for i in data], dtype=np.float32)

        res = {'batch':batch, 'batch1':batch1, 'batch2':batch2, 'batch3':batch3,

               'atomFea': torch.tensor(feaArr, dtype=torch.float32),
               'atomFea1': torch.tensor(feaArr1, dtype=torch.float32),
               'atomFea2': torch.tensor(feaArr2, dtype=torch.float32),
               'atomFea3': torch.tensor(feaArr3, dtype=torch.float32),
                
               'atomAdj': torch.tensor(adjArr, dtype=torch.float32),
               'atomAdj1': torch.tensor(adjArr1, dtype=torch.float32),
               'atomAdj2': torch.tensor(adjArr2, dtype=torch.float32),
               'atomAdj3': torch.tensor(adjArr3, dtype=torch.float32),

               'y':torch.tensor(y, dtype=torch.float32)}

        if self.useFP:
            dFinData1 = np.zeros((len(data), 2048), dtype=np.float32)
            dFinData2 = np.zeros((len(data), 2048), dtype=np.float32)
            dFinData3 = np.zeros((len(data), 2048), dtype=np.float32)
            dFinData = np.zeros((len(data), 2048), dtype=np.float32)
            
            for idx,i in enumerate(data):
                dFinData1[idx] = i['dFinData1']
                dFinData2[idx] = i['dFinData2']
                dFinData3[idx] = i['dFinData3']

                tmp = np.ones((1,))
                DataStructs.ConvertToNumpyArray(AllChem.GetHashedMorganFingerprint(molList[idx],2,nBits=2048, useFeatures=True), tmp)
                dFinData[idx] = tmp

            res['atomFin1'] = torch.tensor(dFinData1, dtype=torch.float32)
            res['atomFin2'] = torch.tensor(dFinData2, dtype=torch.float32)
            res['atomFin3'] = torch.tensor(dFinData3, dtype=torch.float32)
            res['atomFin'] = torch.tensor(dFinData, dtype=torch.float32)

        return res