import argparse
import os
from tqdm import tqdm
# from rdkit import Chem, RDLogger
# from rdkit.Chem import MolStandardize
from Tokenizer import *
from Encoder import *
from GANModel import *
# from lstm_chem.utils.smiles_tokenizer2 import SmilesTokenizer

class SamplePreprocess():
    def __init__(self,final_tokens,smi_to_int,max_len,table_len):
        self.smi_to_int = smi_to_int
        self.final_tokens = final_tokens
        self.max_len = max_len
        self.table_len = table_len
        self.network_input = []
    def main(self):
        # network_output = []

        # create input sequences and the corresponding outputs
        for smile in self.final_tokens:
            if(len(smile) < self.max_len):
                smile += ["A"]*(self.max_len-len(smile))
            sequence_in = smile
            # sequence_out = notes[i + sequence_length]
            self.network_input.append([self.smi_to_int[smile] for smile in sequence_in])
            # network_output.append(note_to_int[sequence_out])

        n_patterns = len(self.network_input)

        # Reshape the input into a format compatible with LSTM layers
        self.network_input = np.reshape(self.network_input, (n_patterns, self.max_len, 1))
        
        # Normalize input between -1 and 1
        self.network_input = (self.network_input - float(self.table_len)/2) / (float(self.table_len)/2)
        print(f'Input shape: {self.network_input.shape}')
        # network_output = np_utils.to_categorical(network_output)

        # return (network_input, network_output)