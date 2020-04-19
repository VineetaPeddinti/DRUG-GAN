import numpy as np
from tqdm import tqdm
class Encoder(object):
    def __init__(self,max_len,one_hot_dict):
        self.one_hot_dict = one_hot_dict
        self.max_len = max_len
    def one_hot_encode(self,smile):
        if(len(smile) < self.max_len):
            smile += ["A"]*(self.max_len-len(smile))
        result = np.array(
            # [self.one_hot_dict[symbol] for symbol in tqdm(smile)],
            [self.one_hot_dict[symbol] for symbol in smile],
            dtype=np.float32)
        result = result.reshape(1, result.shape[0], result.shape[1])
        return result
