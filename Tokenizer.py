import numpy as np
# from tqdm import tqdm
class Tokenizer():
    def __init__(self):
        self.max_len=0
        atoms = [
            'Li',
            'Na',
            'Al',
            'Si',
            'Cl',
            'Sc',
            'Zn',
            'As',
            'Se',
            'Br',
            'Sn',
            'Te',
            'Cn',
            'H',
            'B',
            'C',
            'N',
            'O',
            'F',
            'P',
            'S',
            'K',
            'V',
            'I'
        ]
        special = [
            '(', ')', '[', ']', '=', '#', '%', '0', '1', '2', '3', '4', '5',
            '6', '7', '8', '9', '+', '-', 'se', 'te', 'c', 'n', 'o', 's'
        ]
        padding = ['A'] #Total 50 tokens
        self.table = sorted(atoms, key=len, reverse=True) + special + padding
        self.table_len = len(self.table)
        print(self.table_len)
        self.one_hot_dict = {}
        for i, symbol in enumerate(self.table):
            vec = np.zeros(self.table_len, dtype=np.float32)
            vec[i] = 1
            self.one_hot_dict[symbol] = vec
    
    def tokenize(self, smiles):
        N = len(smiles)
        i = 0
        tokens = []
        while (i < N):
            for j in range(self.table_len):
                symbol = self.table[j]
                if symbol == smiles[i:i + len(symbol)]:
                    tokens.append(symbol)
                    i += len(symbol)
                    break
        if(self.max_len < len(tokens)):
            self.max_len= len(tokens)
        return  tokens
    