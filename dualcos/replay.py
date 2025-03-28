from collections import deque

from my_utils import perm, perm_b
import torch
import numpy as np
import copy
from sklearn import preprocessing
import random

class ReplayMemory(object):
    """Experience replay class interface"""
    def __init__(self):
        """Initialize experience replay"""
        pass

    def update(self, fake, t_logit):
        """Update experience replay with batch of features and labels"""
        pass

    def sample(self):
        """Return batch of features and labels"""
        pass

    def new_epoch(self):
        """Update epoch. This is only relevant for experience replay that has some concept of aging"""
        pass


class NoReplayMemory(ReplayMemory):
    def __init__(self):
        pass


class ClassicalMemory(ReplayMemory):
    """Circular FIFO buffer based experiment replay. Returns uniform random samples when queried."""
    def __init__(self, device, length, batch_size):
        """Initialize replay memory"""
        self.device = device
        self.max_length = length
        self.fakes = None
        self.logits = None
        self.batch_size = batch_size
        self.size = 0
        self.head = 0
        self.used_list = []

    def update(self, fake, t_logit):
        """Update memory with batch of features and labels"""
        fake = fake.cpu()
        t_logit = t_logit.cpu()
        if self.fakes is None:  # Initialize circular buffer if it hasn't already been.
            self.fakes = torch.zeros((self.max_length, fake.shape[1], fake.shape[2], fake.shape[3]))
            self.logits = torch.zeros((self.max_length, t_logit.shape[1]))
        tail = self.head + fake.shape[0]
        if tail <= self.max_length:  # Buffer is not full
            self.fakes[self.head:tail] = fake
            self.logits[self.head:tail] = t_logit
            if self.size < self.max_length:
                self.size = tail
        else:  # Buffer is full
            n = self.max_length - self.head
            self.fakes[self.head:tail] = fake[:n]
            self.logits[self.head:tail] = t_logit[:n]
            tail = tail % self.max_length
            self.fakes[:tail] = fake[n:]
            self.logits[:tail] = t_logit[n:]
            self.size = self.max_length
        self.head = tail % self.max_length

    def sample(self):
        """Return samples uniformly at random from memory"""
        assert self.fakes is not None  # Only sample after having stored samples
        assert self.size >= self.batch_size  # Only sample if we have stored a full batch of samples
        idx = perm(self.size, self.batch_size, self.device).cpu()

        """Different sampling strategies for sample buffer reuse"""
        # # add code
        # if self.size < 3*self.batch_size:
        #     idx = perm(self.size, self.batch_size, self.device).cpu()
        # else:
        #     idx_list = list(range(self.size))
        #     idx_all = torch.LongTensor(idx_list).to(self.device)
        #     new_size = self.size - i*self.batch_size
        #     idx = idx_all[new_size-self.batch_size:new_size]

        # # add code
        # if self.size <= 1000:
        #     idx_t = perm(self.size, self.size, self.device).cpu()
        #     # print(idx_t)
        #     # print(type(idx_t))
        #     # print(idx_t.dtype)
        #     # print(idx_t.tolist())
        #     # self.used_list.extend(idx_t.tolist())
        #     # print(self.used_list)
        #     # ss
        # else:
        #     idx_t = perm(self.size, 1000, self.device).cpu()
        # prob = torch.nn.functional.softmax(self.logits[idx_t], dim=-1)

        # # entropy = []
        # # for i in range(prob.size(0)):
        # #     entropy.append((-torch.sum(prob[i] * torch.log(prob[i]))).item())
        # # t = copy.deepcopy(entropy)
        # # max_number = []
        # # max_index = []
        # # for _ in range(self.batch_size):
        # #     number = max(t)
        # #     index = t.index(number)
        # #     t[index] = 0
        # #     max_number.append(number)
        # #     max_index.append(index)
        # # t = []
        # # # print(max_number)
        # # # print(max_index)
        # # # a = torch.tensor(max_index)
        # # b = [i for i in idx_t]
        # # idx = [b[j] for j in max_index]
        # # idx = [i.item() for i in idx]
        # # idx = torch.LongTensor(idx)

        # prob_max = torch.max(prob, dim=1)
        # a = torch.topk(prob_max[0], self.batch_size, largest = True)
        # b = [i for i in idx_t]
        # idx = [b[j] for j in (a[1].numpy().tolist())]
        # idx = [i.item() for i in idx]
        # idx = torch.LongTensor(idx)


        return self.fakes[idx].to(self.device), self.logits[idx].to(self.device)

    # Improved sampling function
    def sample_boost(self, selected_idx):
        """Return samples uniformly at random from memory"""
        assert self.fakes is not None  # Only sample after having stored samples
        assert self.size >= self.batch_size  # Only sample if we have stored a full batch of samples
        # if len(self.used_list) == self.size:
        if (len(selected_idx) - len(self.used_list)) < self.batch_size:
            self.used_list = []
        idx_b = perm_b(self.size, self.batch_size, self.used_list, selected_idx, self.device).cpu()
        # self.used_list.extend(idx_b.tolist())
        
        """Different sampling strategies for sample buffer reuse"""
        # add code
        # prob = torch.nn.functional.softmax(self.logits[idx_b], dim=-1)
        # prob_max = torch.max(prob, dim=1)
        # a = torch.topk(prob_max[0], self.batch_size, largest = False)
        # b = [i for i in idx_b]
        # idx = [b[j] for j in (a[1].numpy().tolist())]
        # idx = [i.item() for i in idx]
        # idx = torch.LongTensor(idx)
        idx = idx_b
        self.used_list.extend(idx.tolist())

        return self.fakes[idx].to(self.device), self.logits[idx].to(self.device)
    
    # Data reduction process
    def select_idx(self, idx_num, idx_batch_size):
        batch_num = int(self.size / idx_batch_size)
        idx_each_batch = int(idx_num / batch_num)
        print(batch_num, idx_each_batch)

        data_compression = []
        for i in range(batch_num):
            images = []
            outputs = []
            for j in range(i*idx_batch_size, (i+1)*idx_batch_size):
                img = np.array(self.fakes[j]).flatten()
                # print(img.shape)
                # sys.exit()
                output = np.array(self.logits[j].cpu())
                # print(output)
                # print(output.shape)
                # sys.exit()
                img = img.reshape(1,-1)
                # print(img.shape)
                # print(preprocessing.normalize(img,norm='l2'))
                # print(preprocessing.normalize(img,norm='l2').squeeze())
                # sys.exit()
                images.append(preprocessing.normalize(img,norm='l2').squeeze())
                # sys.exit()
                output = output.reshape(1,-1)
                # print(preprocessing.normalize(output,norm='l2'))
                # print(preprocessing.normalize(output,norm='l2').squeeze())
                # sys.exit()
                outputs.append(preprocessing.normalize(output,norm='l2').squeeze())
            images = np.array(images)
            # print(images.shape)
            outputs = np.array(outputs)
            # print(outputs.shape)
            # sys.exit()

            data_num = images.shape[0]
            # print(data_num)
            images_sim = np.dot(images,images.transpose())
            # print(images_sim)
            # sys.exit()
            outputs_sim = np.dot(outputs,outputs.transpose())
            co_sim = np.multiply(images_sim, outputs_sim)
            # print(co_sim)
            # sys.exit()
            
            max_num = idx_each_batch
            n_selected = 0
            index = random.randint(0,data_num-1)
            # print(index)

            while n_selected < max_num:
                index = np.argmin(co_sim[index])
                # print(index)
                data_compression.append(i*idx_batch_size+index)
                n_selected += 1
                co_sim[:,index] = 1
                # print(batch_n, index)
        # print(data_compression)
        # print(type(data_compression), type(data_compression[1]))
        data_compression.sort()
        # print("the sorted list:", data_compression)
        print("the idx_list length:", len(data_compression))
        
        return data_compression


    def new_epoch(self):
        """Does nothing for this type of experience replay"""
        pass

    def __len__(self):
        """Returns number of stored samples"""
        return self.size


def init_replay_memory(args):
    if args.replay == "Off":
        return NoReplayMemory()
    if args.replay == "Classic":
        return ClassicalMemory(args.device, args.replay_size, args.batch_size)
    raise ValueError(f"Unknown replay parameter {args.replay}")