import random
import numpy as np
import torch
import os
from collections import OrderedDict
import resampy


def read_fids(fid_list_f):
    with open(fid_list_f, 'r') as f:
        fids = [l.strip().split()[0] for l in f if l.strip()]
    return fids   

class VqBnfDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        meta_file: str,
        ppg_dir: str,
        ppg_labels_dir: str,
        ppg_file_ext: str = "npy",
    ):
        self.fid_list = read_fids(meta_file)
        self.ppg_dir = ppg_dir
        self.ppg_labels_dir = ppg_labels_dir
        self.ppg_file_ext = ppg_file_ext
        
        random.seed(1234)
        random.shuffle(self.fid_list)
        print(f'[INFO] Got {len(self.fid_list)} samples.')
        
    def __len__(self):
        return len(self.fid_list)

    def get_ppg_input(self, fid): #ppg-ERMS-arctic_a0343.npy
        sprf , wfle, skemb = fid.split('/')
        ppg = np.load(f"{self.ppg_dir}/ppg-{sprf}-{wfle}.{self.ppg_file_ext}")
        return ppg

    def get_ppg_reference(self, fid): #ppg-ERMS-arctic_a0343.npy
        sprf , wfle, skemb = fid.split('/')
        ppg = np.load(f"{self.ppg_dir}/ppg-BDL-{wfle}.{self.ppg_file_ext}")
        labels = np.load(f"{self.ppg_labels_dir}/ppg-BDL-{wfle}.{self.ppg_file_ext}")
        return ppg, labels 

    def __getitem__(self, index):
        fid = self.fid_list[index]
        
        # 1. Load features
        sprf , wfle, skemb = fid.split('/')
        
        ppg = self.get_ppg_input(fid)
        #remove repeatations of same frames
        selection = np.ones(len(ppg), dtype=bool)
        for idx in range(len(ppg)-1):
            if np.array_equal(ppg[idx], ppg[idx+1]):
                selection[idx+1] = False

        # idx = ppg[1:, :] != ppg[:-1, :]
        # print(selection.shape)
        ppg = ppg[selection]
        # print(ppg.shape)

        ppg_ref, ref_labels = self.get_ppg_reference(fid)
        #remove repeatations of same frames
        selection = np.ones(len(ppg_ref), dtype=bool)
        for idx in range(len(ppg_ref)-1):
            if np.array_equal(ppg_ref[idx], ppg_ref[idx+1]):
                selection[idx+1] = False

        ppg_ref = ppg_ref[selection]

        # print(ref_labels.shape)
        selection = np.ones(len(ref_labels), dtype=bool)
        selection[1:] = ref_labels[1:] != ref_labels[:-1]
        ref_labels = ref_labels[selection]

        assert len(ref_labels) == len(ppg_ref)

        
        # 2. Convert numpy array to torch.tensor
        ppg = torch.from_numpy(ppg)
        ppg_ref = torch.from_numpy(ppg_ref)
        ref_labels = torch.from_numpy(ref_labels)
        return (ppg, ppg_ref, ref_labels)

class VqBnfCollate():
    """Zero-pads model inputs and targets based on number of frames per step
    """
    # def __init__(self):
    #     self.give_uttids = give_uttids

    def __call__(self, batch):
        batch_size = len(batch)              
        # Prepare different features 
        # Input = (ppg, ppg_ref, ref_labels)

        ppgs = [x[0] for x in batch]
        ppgs_ref = [x[1] for x in batch]
        labels = [x[2] for x in batch]

        # Pad features into chunk
        ppg_lengths = [x.shape[0] for x in ppgs]
        ppg_ref_lengths = [x.shape[0] for x in ppgs_ref]
        
        max_ppg_len = max(ppg_lengths)
        max_ppg_ref_len = max(ppg_ref_lengths)
        
        ppg_dim = ppgs[0].shape[1]
        # mel_dim = mel[0].shape[1]


        ppgs_padded = torch.FloatTensor(batch_size, max_ppg_len, ppg_dim).zero_()
        ppgs_ref_padded = torch.FloatTensor(batch_size, max_ppg_ref_len, ppg_dim).zero_()
        labels_padded = torch.LongTensor(batch_size, max_ppg_ref_len).zero_()
        stop_tokens = torch.FloatTensor(batch_size, max_ppg_ref_len).zero_()

        for i in range(batch_size):
            cur_ppg_len = ppgs[i].shape[0]
            cur_ppg_ref_len = ppgs_ref[i].shape[0]
            ppgs_padded[i, :cur_ppg_len, :] = ppgs[i]
            ppgs_ref_padded[i, :cur_ppg_ref_len, :] = ppgs_ref[i]
            labels_padded[i, :cur_ppg_ref_len] = labels[i]
            stop_tokens[i, cur_ppg_ref_len:] = 1
            

        ret_tup = (ppgs_padded, ppgs_ref_padded, labels_padded, torch.LongTensor(ppg_lengths), \
                torch.LongTensor(ppg_ref_lengths), stop_tokens)
        
        return ret_tup