import torch
from torch import nn
import math


class MemoryMoCo(nn.Module):
    """Fixed-size queue with momentum encoder"""
    def __init__(self, feature_dim, queue_size, temperature=0.07):
        super(MemoryMoCo, self).__init__()
        self.queue_size = queue_size
        self.temperature = temperature
        self.index = 0

        # noinspection PyCallingNonCallable
        self.register_buffer('params', torch.tensor([-1]))
        stdv = 1. / math.sqrt(feature_dim / 3)
        memory = torch.rand(self.queue_size, feature_dim, requires_grad=False).mul_(2 * stdv).add_(-stdv)
        self.register_buffer('memory', memory)

    def forward(self, q, k, k_all):
        k = k.detach()

        l_pos = (q * k).sum(dim=-1, keepdim=True)  # shape: (batchSize, 1)
        # TODO: remove clone. need update memory in backwards
        l_neg = torch.mm(q, self.memory.clone().detach().t())
        out = torch.cat((l_pos, l_neg), dim=1)
        out = torch.div(out, self.temperature).contiguous()

        # update memory
        with torch.no_grad():
            all_size = k_all.shape[0]
            out_ids = torch.fmod(torch.arange(all_size, dtype=torch.long).cuda() + self.index, self.queue_size)
            self.memory.index_copy_(0, out_ids, k_all)
            self.index = (self.index + all_size) % self.queue_size

        return out


class ClassOracleMemoryMoCo(MemoryMoCo):
    def __init__(self, n_classes, feature_dim, queue_size, temperature=0.07):
        super().__init__(feature_dim, queue_size, temperature=temperature)
        self.n_classes = n_classes
        self.index = [0]*n_classes

        # queues
        stdv = 1. / math.sqrt(feature_dim / 3)
        for i in range(n_classes):
            memory = torch.rand(self.queue_size, feature_dim, requires_grad=False).mul_(2 * stdv).add_(-stdv)
            self.register_buffer('memory_{}'.format(i), memory)
        # print("----- BUFFERS -----")
        # print(self._buffers.keys())
        self.register_buffer('memory', torch.tensor([0]))  # "free" up the original memory buffer, not used here

    def forward(self, q, k, k_all, q_labels):
        k = k.detach()

        l_pos = (q * k).sum(dim=-1, keepdim=True)  # shape: (batchSize, 1)
        # TODO: remove clone. need update memory in backwards
        # TODO (silvio):    iterate over classes instead of over samples? use indexing, but need to change labels.
        #                   probably it wouldn't get much faster though
        l_neg = []
        for sample, label in zip(q, q_labels):
            l_neg.append(torch.mm(sample.unsqueeze(0), getattr(self, 'memory_{}'.format(label)).clone().detach().t()))
        l_neg = torch.cat(l_neg, dim=0)
        out = torch.cat((l_pos, l_neg), dim=1)
        out = torch.div(out, self.temperature).contiguous()

        # print("---------------------------------------------")
        # print("** {}".format(self.memory_0.mean(dim=-1)))

        # update memory
        with torch.no_grad():
            # print("\n---- DBG mem update ----")
            for i in range(self.n_classes):
                k_all_idx = k_all[q_labels == i, :]
                all_size = k_all_idx.shape[0]
                out_ids = torch.fmod(torch.arange(all_size, dtype=torch.long).cuda() + self.index[i], self.queue_size)
                getattr(self, 'memory_{}'.format(i)).index_copy_(0, out_ids, k_all_idx)
                self.index[i] = (self.index[i] + all_size) % self.queue_size

            #     if i == 0:
            #         print(k_all_idx.mean(dim=-1))
            #         print(out_ids)
            # print("** {}".format(self.memory_0.mean(dim=-1)))

        return out

