import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

class Dataset(Dataset):
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
      

def get_samples_parameters(label_dir, val_ratio=0.15, test_ratio=0.10, seed=1):
    """
    Generate sample parameters for dataset splitting.
    
    Args:
        data_path (str): Path to the dataset directory
        val_ratio (float): Ratio of validation data (default: 0.15)
        test_ratio (float): Ratio of test data (default: 0.10)
        seed (int): Random seed for reproducibility (default: 1)
    
    Returns:
        dict: Dictionary containing sample parameters including:
            - num_phantoms: Total number of samples
            - train_index: List of training sample IDs
            - validation_index: List of validation sample IDs
            - test_index: List of test sample IDs
            - num_train_data: Number of training samples
            - num_validation_data: Number of validation samples
            - num_test_data: Number of test samples
    """
    np.random.seed(seed)
    
    SAMPLES_PARA = {}
    
    # Get all file names and extract IDs
    all_files = os.listdir(label_dir)
    # Extract IDs from filenames (assuming format is "ID.npy")
    all_ids = [int(os.path.splitext(f)[0]) for f in all_files]
    all_ids.sort()  # Sort IDs to ensure consistent ordering
    
    SAMPLES_PARA['num_phantoms'] = len(all_ids)
    
    # Calculate split sizes
    val_test_size = int(SAMPLES_PARA['num_phantoms'] * (val_ratio + test_ratio))
    val_size = int(SAMPLES_PARA['num_phantoms'] * val_ratio)
    test_size = int(SAMPLES_PARA['num_phantoms'] * test_ratio)
    
    # Generate random indices for validation and test
    random_indices = np.random.choice(
        len(all_ids),
        val_test_size,
        False
    ).tolist()
    
    # Split into validation and test
    val_indices = random_indices[:val_size]
    test_indices = random_indices[val_size:val_size + test_size]
    
    # Get actual IDs for each split
    SAMPLES_PARA['validation_index'] = [all_ids[i] for i in val_indices]
    SAMPLES_PARA['test_index'] = [all_ids[i] for i in test_indices]
    
    # Get training indices (remaining indices)
    train_indices = [i for i in range(len(all_ids)) 
                    if i not in val_indices and i not in test_indices]
    SAMPLES_PARA['train_index'] = [all_ids[i] for i in train_indices]
    
    # Calculate sizes
    SAMPLES_PARA['num_train_data'] = len(SAMPLES_PARA['train_index'])
    SAMPLES_PARA['num_validation_data'] = len(SAMPLES_PARA['validation_index'])
    SAMPLES_PARA['num_test_data'] = len(SAMPLES_PARA['test_index'])
    
    return SAMPLES_PARA


def get_data_loader(BP_path, GT_path, batch_size, linux=True):
    """
    Create a DataLoader for the dataset.
    
    Args:
        BP_path (str): Path to the BP data
        GT_path (str): Path to the GT data
        list_IDs (list): List of IDs for the dataset    
        batch_size (int): Batch size for the DataLoader
        linux (bool): Whether the system is Linux (default: True)
    
    Returns:
        torch.utils.data.DataLoader: DataLoader for the dataset
    """
    SAMPLES_PARA = get_samples_parameters(GT_path)
    train_dataset = Dataset(BP_path, GT_path, SAMPLES_PARA['train_index'], linux)
    val_dataset = Dataset(BP_path, GT_path, SAMPLES_PARA['validation_index'], linux)
    test_dataset = Dataset(BP_path, GT_path, SAMPLES_PARA['test_index'], linux)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=4, drop_last=True, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=4, drop_last=True, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, drop_last=True, pin_memory=True, persistent_workers=True)

    return train_loader, val_loader, test_loader