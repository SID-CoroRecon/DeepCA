import numpy as np
import torch

class Dataset(torch.utils.data.Dataset):
      'Characterizes a dataset for PyTorch'
      def __init__(self, data_path, list_IDs):
            'Initialization'
            self.list_IDs = list_IDs
            self.data_path = data_path

      def __len__(self):
            'Denotes the total number of samples'
            return len(self.list_IDs)

      def __getitem__(self, index):
            'Generates one sample of data'
            # Select sample
            ID = self.list_IDs[index]

            # Load data and get label
            data_file = self.data_path + 'CCTA_BP/recon_' + str(ID) + '.npy'
            data = np.transpose(np.load(data_file)[:,:,:,np.newaxis])
            label_file = self.data_path + 'split_two/' + str(ID) + '.npy'
            label = np.transpose(np.load(label_file)[:,:,:,np.newaxis])

            return torch.from_numpy(data), torch.from_numpy(label)