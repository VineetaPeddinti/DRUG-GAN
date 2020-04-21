#!/usr/bin/env python
import argparse
import os
from tqdm import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem import MolStandardize
from Tokenizer import *
from Encoder import *
from GANModel import *
# from lstm_chem.utils.smiles_tokenizer2 import SmilesTokenizer

class Preprocessor():
    def __init__(self):
        self.normarizer = MolStandardize.normalize.Normalizer()
        self.lfc = MolStandardize.fragment.LargestFragmentChooser()
        self.uc = MolStandardize.charge.Uncharger()
        self.max_len = 0
        self.one_hot_dict = {}

    def process(self, smi):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            mol = self.normarizer.normalize(mol)
            mol = self.lfc.choose(mol)
            mol = self.uc.uncharge(mol)
            smi = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
            return smi
        else:
            return None
    def tokenize(self, smiles):
        t = Tokenizer()
        smile_tokens= []
        for smile in smiles:
            smile_tokens.append(t.tokenize(smile))
        self.max_len = t.max_len
        self.one_hot_dict = t.one_hot_dict 
        self.table_len = t.table_len
        return smile_tokens
    def main(self):
        with open("dataset.smi", 'r') as f:
            smiles = [l.rstrip() for l in f] #Remove \n
        print(f'input SMILES num: {len(smiles)}')
        print('start to clean up')
        pp = Preprocessor()
        # pp_smiles = [pp.process(smi) for smi in tqdm(smiles)]
        pp_smiles = [pp.process(smi) for smi in smiles]
        cl_smiles = list(set([s for s in pp_smiles if s]))
        out_smiles = []
        for cl_smi in cl_smiles:
            out_smiles.append(cl_smi)
        print('done.')
        print(f'output SMILES num: {len(out_smiles)}')
        final_tokens = pp.tokenize(out_smiles)
        one_hot_tokens = []
        for smile_token in final_tokens:
            se = Encoder(pp.max_len, pp.one_hot_dict)
            one_hot_tokens.append(se.one_hot_encode(smile_token))
        print("*******************Data Exploration****************************")
        print(f'input SMILES num: {len(smiles)}')
        print(f'output SMILES num: {len(out_smiles)}')
        print(f'final_tokens: {len(final_tokens)}')
        print(f'one_hot_tokens: {len(one_hot_tokens)}')
        print(f'Each smile string one_hot_token length: {len(one_hot_tokens[0][0])},{one_hot_tokens[0][0].shape}')
        print(f'Each smile string each token one_hot_token length: {len(one_hot_tokens[0][0][0])}',{one_hot_tokens[0][0][0].shape})
        print("********************Data Exploration***************************")
        with open("out_smiles.smi", 'w') as f:
            for smi in out_smiles:
                f.write(smi + '\n')
        train_data = np.asarray(one_hot_tokens, dtype=np.float32)
        print(train_data.shape)
        # print(pp.max_len)
        # print(pp.table_len)
        # print(train_data[0])
        g =  GANModel(pp.max_len,pp.table_len)
        g.train(100,64,50,train_data)
        imagesss = g.sample_images()
        print(imagesss[0].shape)
        print(imagesss[0].reshape(1, 134,50)[0][0:20])
if __name__ == "__main__":
    Preprocessor().main()