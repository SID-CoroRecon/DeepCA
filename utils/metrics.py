import numpy as np
from scipy.spatial import KDTree


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