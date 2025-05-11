import numpy as np
import torch

class Dataset(torch.utils.data.Dataset):
      'Characterizes a dataset for PyTorch'
      def __init__(self, BP_path, GT_path, list_IDs, linux=False):
            'Initialization'
            self.list_IDs = list_IDs
            self.BP_path = BP_path
            self.GT_path = GT_path
            self.linux = linux

      def __len__(self):
            'Denotes the total number of samples'
            return len(self.list_IDs)

      def __getitem__(self, index):
            'Generates one sample of data'
            # Select sample
            ID = self.list_IDs[index]
            if self.linux:
                  BP_file = self.BP_path + '/recon_' + str(ID) + '.npy'
                  GT_file = self.GT_path + '/' + str(ID) + '.npy'
            else:
                  BP_file = self.BP_path + '\\recon_' + str(ID) + '.npy'
                  GT_file = self.GT_path + '\\' + str(ID) + '.npy'
            # Load data and get label
            BP = np.transpose(np.load(BP_file)[:,:,:,np.newaxis])
            GT = np.transpose(np.load(GT_file)[:,:,:,np.newaxis])

            return torch.from_numpy(BP), torch.from_numpy(GT)