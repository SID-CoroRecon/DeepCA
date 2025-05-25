import torch
import torch.optim as optim
import numpy as np
import os
import gc
import sys
from scipy.spatial import KDTree

# Install required packages
# !pip install trimesh scikit-image numpy-stl
from skimage.measure import marching_cubes
from stl import mesh as stl_mesh

sys.path.append('/kaggle/input/deepca_2/pytorch/default/1')

from generator import Generator
from discriminator import Discriminator
from load_volume_data_RCA import Dataset
from samples_parameters import get_samples_parameters
from utils import *

output_dir = '/kaggle/working/outputs_results/'
label_dir = '/kaggle/input/deepca-preprocessed-dataset/data/split_two'
BP_dir = '/kaggle/input/deepca-preprocessed-dataset/data/CCTA_BP'

os.makedirs(os.path.join(output_dir, '3d_models'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'numpy_models'), exist_ok=True)
SAMPLES_PARA = get_samples_parameters(label_dir)


def save_3d_model(volume, filename):
    """Save 3D volume in multiple formats with robust error handling"""
    try:
        # Convert to numpy array if it's a tensor
        if torch.is_tensor(volume):
            volume = volume.squeeze().cpu().detach().numpy()
        volume = volume.squeeze()
        
        threshold = 0.5
        
        # Create mesh using marching cubes
        vertices, faces, _, _ = marching_cubes(volume, level=threshold)
        
        # Save as STL (most widely supported)
        stl_mesh_obj = stl_mesh.Mesh(np.zeros(faces.shape[0], dtype=stl_mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                stl_mesh_obj.vectors[i][j] = vertices[f[j]]
        stl_mesh_obj.save(filename + '.stl')
        
        return True
    except Exception as e:
        print(f"3D model save failed: {str(e)}")
        return False


def save_test_results(pred, gt, filename):
    np.save(os.path.join(output_dir, 'numpy_models', f'pred_{filename}.npy'), pred)
    np.save(os.path.join(output_dir, 'numpy_models', f'gt_{filename}.npy'), gt)
    save_3d_model(pred, os.path.join(output_dir, '3d_models', f'pred_{filename}'))
    save_3d_model(gt, os.path.join(output_dir, '3d_models', f'gt_{filename}'))


def compute_overlap_metric(
    label: np.ndarray,
    output: np.ndarray,
    d: float = 0,
    voxel_spacing: tuple = (0.37695312, 0.37695312, 0.5),
    threshold_label: float = 0.5,
    threshold_output: float = 0.5
):
    """
    Computes the Overlap Metric (Ot(d)) between a binary label and a predicted output.
    
    Args:
        label (numpy.ndarray): Ground truth binary array (2D or 3D).
        output (numpy.ndarray): Predicted binary array (same shape as label).
        d (float): Distance threshold in mm (0 for Dice, 1 or 2 for relaxed overlap).
        voxel_spacing (tuple): Physical voxel spacing in mm (x,y,z). Default matches CCTA data.
        threshold_label (float): Binarization threshold for label (default: 0.5).
        threshold_output (float): Binarization threshold for output (default: 0.5).
    
    Returns:
        float: Overlap score Ot(d) ∈ [0, 1].
    """
    # Binarize inputs
    label_bin = (label >= threshold_label).astype(np.uint8)
    output_bin = (output >= threshold_output).astype(np.uint8)
    
    # If d=0, compute Dice score (Ot(0))
    if d == 0:
        intersection = np.sum(label_bin & output_bin)
        union = np.sum(label_bin) + np.sum(output_bin)
        return (2 * intersection) / union if union != 0 else 0.0
    
    # Convert mm distance to voxel units using the smallest spacing dimension
    d_voxels = d / min(voxel_spacing)
    
    # Get coordinates of foreground points in label and output
    label_points = np.argwhere(label_bin > 0)
    output_points = np.argwhere(output_bin > 0)
    
    # If either set is empty, return 0 (no overlap)
    if len(label_points) == 0 or len(output_points) == 0:
        return 0.0
    
    # Build KDTree for the output points (prediction)
    kdtree = KDTree(output_points)
    
    # Find TPR(d): Label points within distance d_voxels of any output point
    dist_label_to_output, _ = kdtree.query(label_points, distance_upper_bound=d_voxels)
    tpr_d = np.sum(dist_label_to_output <= d_voxels)
    fn_d = len(label_points) - tpr_d
    
    # Build KDTree for the label points (ground truth)
    kdtree_label = KDTree(label_points)
    
    # Find TPM(d): Output points within distance d_voxels of any label point
    dist_output_to_label, _ = kdtree_label.query(output_points, distance_upper_bound=d_voxels)
    tpm_d = np.sum(dist_output_to_label <= d_voxels)
    fp_d = len(output_points) - tpm_d
    
    # Compute Ot(d)
    ot_d = (tpm_d + tpr_d) / (tpm_d + tpr_d + fn_d + fp_d)
    return ot_d


def compute_chamfer_distance(
    label: np.ndarray,
    output: np.ndarray,
    voxel_spacing: tuple = (0.37695312, 0.37695312, 0.5),
    threshold_label: float = 0.5,
    threshold_output: float = 0.5,
    physical_units: bool = True
) -> float:
    """
    Computes the symmetric Chamfer Distance (L2) between two binary volumes.
    
    Args:
        label (np.ndarray): Ground truth binary array (2D/3D).
        output (np.ndarray): Predicted binary array (same shape as label).
        voxel_spacing (tuple): Physical voxel spacing in mm (x,y,z). Default matches CCTA data.
        threshold_label (float): Binarization threshold for label. Default: 0.5.
        threshold_output (float): Binarization threshold for output. Default: 0.5.
        physical_units (bool): If True, returns distance in mm. If False, in voxels.
    
    Returns:
        float: Chamfer Distance (L2) in mm or voxels.
    """
    # Remove batch and channel dimensions (assuming batch_size=1)
    label = label.squeeze(0).squeeze(0)  # Now (x,y,z)
    output = output.squeeze(0).squeeze(0)

    # Binarize inputs
    label_bin = (label >= threshold_label).astype(np.uint8)
    output_bin = (output >= threshold_output).astype(np.uint8)
    
    # Get coordinates of foreground points
    label_points = np.argwhere(label_bin > 0)
    output_points = np.argwhere(output_bin > 0)
    
    # Handle empty cases
    if len(label_points) == 0 or len(output_points) == 0:
        return np.inf  # No overlap
    
    # Scale coordinates to physical units if requested
    if physical_units:
        label_points = label_points * np.array(voxel_spacing)
        output_points = output_points * np.array(voxel_spacing)
    
    # Build KDTrees
    kdtree_label = KDTree(label_points)
    kdtree_output = KDTree(output_points)
    
    # Label → Output distances
    dist_label_to_output, _ = kdtree_output.query(label_points)
    term1 = np.mean(dist_label_to_output ** 2)
    
    # Output → Label distances
    dist_output_to_label, _ = kdtree_label.query(output_points)
    term2 = np.mean(dist_output_to_label ** 2)
    
    # Symmetric Chamfer Distance (ℓ2)
    chamfer_distance = (term1 + term2) ** 0.5
    return chamfer_distance


def main():
    print("Starting main...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = Generator(in_channels=1, num_filters=64, class_num=1).to(device)
    discriminator = Discriminator(device, 2).to(device)
    
    gc.collect()
    torch.cuda.empty_cache()

    test_set = Dataset(BP_dir, label_dir, SAMPLES_PARA['test_index'], linux=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=1, 
                                          shuffle=True, num_workers=4, 
                                          drop_last=True, pin_memory=True)
    print("Dataset setup complete.")

    LEARNING_RATE = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.9))
    D_optimizer = optim.Adam(discriminator.parameters(), 
                                    lr=1e-4, betas=(0.5, 0.9))

    print("Loading checkpoint...")
    checkpoint_path = '/kaggle/input/deepca-training-18/outputs_results/checkpoints/Epoch_198.tar'
    _ = load_checkpoint(model, discriminator, optimizer, D_optimizer, checkpoint_path)

    model.eval()

    ot_1mm_list = []
    ot_2mm_list = []
    chamfer_distance_list = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for i, data in enumerate(testloader):
            inputs, labels = data[0].float().to(device), data[1].float().to(device)
            outputs = model(inputs)

            outputs = outputs.cpu().numpy().astype(np.int8)
            labels = labels.cpu().numpy().astype(np.int8)

            # Compute Ot(d) for d=1 & d=2
            ot_1mm = compute_overlap_metric(labels, outputs, d=1)
            ot_2mm = compute_overlap_metric(labels, outputs, d=2)

            # Compute Chamfer Distance
            chamfer_distance = compute_chamfer_distance(labels, outputs)

            ot_1mm_list.append(ot_1mm)
            ot_2mm_list.append(ot_2mm)
            chamfer_distance_list.append(chamfer_distance)

            print(f"Sample {i}: Ot(1mm)={ot_1mm*100:.2f}, Ot(2mm)={ot_2mm*100:.2f}, CD(L2)={chamfer_distance:.2f}")

            save_test_results(outputs, labels, f'{i}')
            print(f'{i} / {len(testloader)} samples done')
    
    print(f"Ot(1mm): {np.mean(ot_1mm_list)*100:.2f}, Ot(2mm): {np.mean(ot_2mm_list)*100:.2f}, CD(L2): {np.mean(chamfer_distance_list):.2f}")

if __name__ == "__main__":
    main()
    print("Done")