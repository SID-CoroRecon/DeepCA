import numpy as np
import os
import torch

ab_path = os.getcwd() + '/datasets/'

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs):
        'Initialization'
        self.list_IDs = list_IDs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        data_file = ab_path + 'CCTA_BP/recon_' + str(ID) + '.npy'
        data = np.transpose(np.load(data_file)[:,:,:,np.newaxis])
        label_file = ab_path + 'CCTA_GT/' + str(ID) + '.npy'
        label = np.transpose(np.load(label_file)[:,:,:,np.newaxis])

        return torch.from_numpy(data), torch.from_numpy(label)