import numpy as np
import os

def get_samples_parameters(data_path, val_ratio=0.15, test_ratio=0.10, seed=1):
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
    label_dir = os.path.join(data_path, 'split_two')
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