import numpy as np
import matplotlib.pyplot as plt

def plot_3d_voxels(path, threshold=0.5):
    """
    Loads a 3D binary mask from a .npy file and shows a 3D scatter of non-zero voxels.

    Parameters:
    - path (str): Path to the .npy file containing a 3D volume.
    - threshold (float): Minimum value to consider as a voxel (useful for binarization).
    """
    vol = np.load(path)
    if vol.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {vol.shape}")

    coords = np.where(vol > threshold)
    if coords[0].size == 0:
        print("No voxels found above threshold.")
        return

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(coords[0], coords[1], coords[2], s=1, marker='.', alpha=0.6)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("3D Voxel Visualization")
    plt.show()

plot_3d_voxels("data/CCTA_BP/recon_1_569.npy")