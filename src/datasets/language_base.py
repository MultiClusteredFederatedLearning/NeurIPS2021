import torch
from torch.utils.data import Dataset

class LanguageBaseDataset(Dataset):
    all_letters = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
    num_letters = len(all_letters)

    def _one_hot(self, index, size):
        vec = torch.zeros(size, dtype=torch.int64)
        vec[index] = 1
        return vec
    
    def _letter2vec(self, letter):
        index = self.all_letters.find(letter)
        return self._one_hot(index, self.num_letters)

    def _word2ind(self, word):
        indices = torch.tensor([self.all_letters.find(c) for c in word])
        return indices