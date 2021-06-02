import random
import itertools

import numpy as np
import copy
import torch
from tqdm import tqdm
from collections import OrderedDict


class GNByzantineClusteredFederatedLearning:
    def __init__(self, model, clients, criterion, dataloader):
        self.device = "cpu"
        self.cos_fn = torch.nn.CosineSimilarity(dim=0)

        self.model = model
        self.clients = clients
        self.model.to(self.device)
        self.dataloader = dataloader
        self.criterion = criterion

        self.benign = np.array([i for i in range(len(clients))]) # 良性クライアント集合
        self.adv = np.array([], dtype=np.int64) # 悪性クライアント集合
        
        self.global_local_similarity = None


    def aggregate(self):
        total_size = sum([len(self.clients[k]) for k in self.benign])
        update_state = OrderedDict()
        for k in self.benign:
            client = self.clients[k]
            local_state = client.model.state_dict()
            weight = len(client) / total_size
            for key, value in local_state.items():
                if not key in update_state.keys():
                    update_state[key] = value * weight
                else:
                    update_state[key] += value * weight

        # NOTE: delta_global_param用 ------
        prev_global_param = []
        for p in self.model.parameters():
            prev_global_param.append(p.view(-1))
        prev_global_param = torch.cat(prev_global_param)
        # ------

        self.model.load_state_dict(update_state)
    
        # NOTE: delta_global_param用 ------
        cur_global_param = []
        for p in self.model.parameters():
            cur_global_param.append(p.view(-1))
        cur_global_param = torch.cat(cur_global_param)

        self.delta_global_param = cur_global_param - prev_global_param
        # ------

    def eval(self):
        self.model.eval()
        avg_loss = 0
        avg_accuracy = 0
        with torch.no_grad():
            for img, label in self.dataloader:
                label = label.to(self.device)
                img = img.to(self.device)
                logits = self.model(img)
                loss = self.criterion(logits, label)
                avg_loss += loss.item()

                pred = logits.argmax(dim=1, keepdim=True)
                avg_accuracy += pred.eq(label.view_as(pred)).sum().item()

        avg_loss = avg_loss / len(self.dataloader)
        avg_accuracy = 100. * avg_accuracy / len(self.dataloader.dataset)
        return avg_loss, avg_accuracy

    def to(self, device):
        self.device = device
        self.model.to(device)

    def sampling(self, num_sample):
        return random.sample(range(len(self.clients)), num_sample)

    def partition(self, cluster, simmat):
        # & : np.intersect1d
        # | : np.union1d
        # - : np.setdiff1d

        M = len(cluster)
        simmat = simmat[cluster][:, cluster]
        s = np.argsort(-simmat.reshape(-1)) # 類似度が高い順
        C = [np.array([i]) for i in range(M)]
        for i in range(M**2):
            i1 = s[i] // M
            i2 = s[i] % M
            c_tmp = np.array([], dtype=np.int64)
            for cid, c in enumerate(C):
                if (i1 in c) or (i2 in c):
                    c_tmp = np.union1d(c_tmp, c)
                    del C[cid] # 正常に機能してる（forのcidは削除後のCの参照してるので）
            C.append(c_tmp)
            if len(C) <= 8:
                if len(C) == 2:
                    return C

                # 解析解
                minsim = 1.1
                mincls = None
                for l in range(len(C) // 2 + len(C) % 2):
                    for v in itertools.combinations(C, l+1):
                        if len(v) == 1:
                            cls1 = v[0]
                        else:
                            cls1 = np.unique(np.concatenate(v)) # union
                        cls2 = np.setdiff1d(np.arange(M), cls1)

                        cross_sim = simmat[cls2][:, cls1].max()
                        if minsim > cross_sim:
                            minsim = cross_sim
                            mincls = [cluster[cls1], cluster[cls2]]
                
                return mincls

    def clustering(self, thresh=0.02, alpha=0.):
        M = len(self.clients)
        if M < 2:
            return

        simmat = np.eye(M)

        # === praram ===
        local_params = []
        for client in self.clients:
            param = []
            for p in client.model.parameters():
                param.append(p.view(-1))
            local_params.append(torch.cat(param))
        local_params = torch.stack(local_params)

        global_param = []
        for p in self.model.parameters():
            global_param.append(p.view(-1))
        global_param = torch.cat(global_param)

        # === similarity ===
        for i in range(M-1):
            for j in range(i+1, M):
                simmat[i, j] = simmat[j, i] = self.cos_fn(local_params[i] - global_param, local_params[j] - global_param).item()

        gls = np.zeros((M,))
        for i in range(M):
            gls[i] = self.cos_fn(local_params[i], global_param).item()

        regmat = np.zeros_like(simmat)
        for i, j in itertools.product(range(M), range(M)):
            regmat[i, j] = abs(gls[i] - gls[j])

        simmat_with_reg = simmat - alpha * regmat

        # === partition ===

        # partition
        C = self.partition(self.benign, simmat_with_reg)

        cls1, cls2 = C[0], C[1]
        if len(cls1) < len(cls2):
            cls1, cls2 = cls2, cls1

        xsim = simmat[cls2][:, cls1].max()

        if xsim < thresh:
            new_adv = self.benign[cls2]
            self.benign = np.setdiff1d(self.benign, new_adv)
            self.adv = np.union1d(self.adv, new_adv)
        
        return xsim, simmat, cls1, cls2