import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

class Dataset(Dataset):
      'Characterizes a dataset for PyTorch'
      def __init__(self, BP_path, GT_path, list_IDs, linux=False, one_view=False):
            'Initialization'
            self.list_IDs = list_IDs
            self.BP_path = BP_path
            self.GT_path = GT_path
            self.linux = linux
            self.one_view = one_view

      def __len__(self):
            'Denotes the total number of samples'
            return len(self.list_IDs)

      def __getitem__(self, index):
            'Generates one sample of data'
            # Select sample
            ID = self.list_IDs[index]
            
            # Find BP file by searching for files containing the ID
            bp_files = [f for f in os.listdir(self.BP_path) if f.endswith(f"_{str(ID)}.npy")]
            if not bp_files:
                raise FileNotFoundError(f"No BP file found for ID {ID}")
            BP_file = os.path.join(self.BP_path, bp_files[0])
            
            # Extract label number from BP filename (format: recon_label_number_ID.npy)
            parts = BP_file.replace('.npy', '').split('_')
            if self.one_view:
                label_number = int(parts[-2])
            else:
                 label_number = int(parts[-1])
            GT_file = os.path.join(self.GT_path, f"{label_number}.npy")
            
            # Load data and get label
            BP = np.transpose(np.load(BP_file)[:,:,:,np.newaxis])
            GT = np.transpose(np.load(GT_file)[:,:,:,np.newaxis])

            return torch.from_numpy(BP), torch.from_numpy(GT)
      

def get_samples_parameters(BP_dir):
    """
    Generate sample parameters for dataset splitting.
    Train: IDs 1-899, Test: IDs 900-1000
    
    Args:
        BP_dir (str): Path to the dataset directory
    
    Returns:
        dict: Dictionary containing sample parameters including:
            - num_phantoms: Total number of samples
            - train_index: List of training sample IDs (1-899)
            - test_index: List of test sample IDs (900-1000)
            - num_train_data: Number of training samples
            - num_test_data: Number of test samples
    """
    SAMPLES_PARA = {}
    
    # Get all file names and extract IDs
    all_files = os.listdir(BP_dir)
    # Extract IDs from filenames (assuming format is "ID.npy" or "recon_*_ID.npy")
    all_ids = [int(''.join(filter(str.isdigit, os.path.splitext(f)[0].split('_')[-1]))) for f in all_files]
    all_ids.sort()  # Sort IDs to ensure consistent ordering
    
    # Split data: train (1-899), test (900-1000)
    SAMPLES_PARA['train_index'] = [id for id in all_ids if 1 <= id <= 899]
    SAMPLES_PARA['test_index'] = [id for id in all_ids if 900 <= id <= 1000]
    
    # Calculate sizes
    SAMPLES_PARA['num_phantoms'] = len(all_ids)
    SAMPLES_PARA['num_train_data'] = len(SAMPLES_PARA['train_index'])
    SAMPLES_PARA['num_test_data'] = len(SAMPLES_PARA['test_index'])
    
    return SAMPLES_PARA


def get_data_loader(BP_path, GT_path, batch_size, linux=True, rank=None, world_size=None):
    """
    Create a DataLoader for the dataset with distributed training support.
    
    Args:
        BP_path (str): Path to the BP data
        GT_path (str): Path to the GT data
        batch_size (int): Batch size for the DataLoader
        linux (bool): Whether the system is Linux (default: True)
        rank (int): Rank of the current process
        world_size (int): Total number of processes
    
    Returns:
        tuple: (train_loader, test_loader)
    """
    SAMPLES_PARA = get_samples_parameters(BP_path)
    train_dataset = Dataset(BP_path, GT_path, SAMPLES_PARA['train_index'], linux)
    test_dataset = Dataset(BP_path, GT_path, SAMPLES_PARA['test_index'], linux)

    # Create distributed samplers if rank and world_size are provided
    if rank is not None and world_size is not None:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
    else:
        train_sampler = None
        test_sampler = None

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=(test_sampler is None),
        sampler=test_sampler,
        num_workers=4,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, test_loader