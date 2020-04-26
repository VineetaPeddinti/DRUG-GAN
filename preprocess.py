#!/usr/bin/env python
import argparse
import os
from tqdm import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem import MolStandardize
from Tokenizer import *
from Encoder import *
from GANModel import *
from sample_preprocess import SamplePreprocess
# from lstm_chem.utils.smiles_tokenizer2 import SmilesTokenizer

class Preprocessor():
    def __init__(self):
        self.normarizer = MolStandardize.normalize.Normalizer()
        self.lfc = MolStandardize.fragment.LargestFragmentChooser()
        self.uc = MolStandardize.charge.Uncharger()
        self.max_len = 0
        self.one_hot_dict = {}
        self.data_length = 10000

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
        self.table = t.table
        return smile_tokens
    def main(self):
        with open("original_dataset.smi", 'r') as f:
            smiles = [l.rstrip() for l in f] #Remove \n
        print(f'input SMILES num: {len(smiles)}')
        print('start to clean up')
        smiles = smiles[:self.data_length]
        pp = Preprocessor()
        pp_smiles = [pp.process(smi) for smi in tqdm(smiles)]
        cl_smiles = list(set([s for s in pp_smiles if s]))
        out_smiles = []
        for cl_smi in cl_smiles:
            out_smiles.append(cl_smi)
        print('done.')
        print(f'output SMILES num: {len(out_smiles)}')
        with open("out_smiles.smi", 'w') as f:
            for smi in out_smiles:
                f.write(smi + '\n')
        final_tokens = pp.tokenize(out_smiles)
        smi_to_int = dict((smi, number) for number, smi in enumerate(pp.table))
        int_to_smi = dict((number, smi) for number, smi in enumerate(pp.table))
        print(smi_to_int)
        print("*******************Data Exploration****************************")
        print(f'input SMILES num: {len(smiles)}')
        print(f'output SMILES num: {len(out_smiles)}')
        print(f'final_tokens: {len(final_tokens)}')
        sp = SamplePreprocess(final_tokens,smi_to_int,pp.max_len,pp.table_len)
        sp.main()
        print("********************Data Exploration***************************")
        # one_hot_tokens = []
        # for smile_token in final_tokens:
        #     se = Encoder(pp.max_len, pp.one_hot_dict)
        #     one_hot_tokens.append(se.one_hot_encode(smile_token))
        # train_data = np.asarray(one_hot_tokens, dtype=np.float32)
        # print(train_data.shape)
        # print(pp.max_len)
        # print(pp.table_len)
        # print(train_data[0])
        g =  GANModel(pp.max_len,pp.table_len,int_to_smi)
        g.train(100,64,50,sp.network_input)
        g.sample_images()
        # print(imagesss[0].reshape(1, pp.table_len,pp.max_len)[0][0:50][562])
if __name__ == "__main__":
    Preprocessor().main()