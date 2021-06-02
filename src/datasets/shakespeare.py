import json
import torch
from datasets.language_base import LanguageBaseDataset

class ShakespeareDataset(LanguageBaseDataset):
    def __init__(self, corpus, client_index):
        self.user = corpus['users'][client_index]
        self.num_sample = corpus['num_samples'][client_index]
        self.data = corpus['user_data'][self.user] # dict('x', 'y')
        self.hierarchy = corpus['hierarchies'][client_index]
    
    def __getitem__(self, index):
        inp = self.data['x'][index]
        target = self.data['y'][index]
        inp = self._word2ind(inp)
        # target = self._letter2vec(target)
        target = self.all_letters.find(target)
        return inp, target

    def __len__(self):
        return self.num_sample

class ShakespeareAllDataset(LanguageBaseDataset):
    def __init__(self, corpus, max_client=None):
        self.users = corpus['users'] if max_client is None else corpus['users'][:max_client]
        self.data = dict(x=[], y=[])
        self.num_sample = 0
        for i, user in enumerate(self.users):
            self.num_sample += corpus['num_samples'][i]
            data = corpus['user_data'][user]
            self.data['x'].extend(data['x'])
            self.data['y'].extend(data['y'])

    def __getitem__(self, index):
        inp = self.data['x'][index]
        target = self.data['y'][index]
        inp = self._word2ind(inp)
        # target = self._letter2vec(target)
        target = self.all_letters.find(target)
        return inp, target

    def __len__(self):
        return self.num_sample

if __name__ == "__main__":
    corpus_path = "../../data/shakespeare/all_data_niid_0_keep_0_test_9.json"
    with open(corpus_path) as f:
        corpus = json.load(f)
    dataset = ShakespeareAllDataset(corpus)
    inp, target = dataset[0]

    corpus_path = "../../data/shakespeare/all_data_niid_0_keep_0_train_9.json"
    with open(corpus_path) as f:
        corpus = json.load(f)
    dataset = ShakespeareDataset(corpus, 0)
    inp, target = dataset[0]
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=10,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
        drop_last=False
    )
    for inputs, targets in dataloader:
        pass