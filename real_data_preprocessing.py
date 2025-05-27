import os
import pydicom
import numpy as np
import tigre
from tigre.algorithms import sirt
from tqdm import tqdm
import matplotlib.pyplot as plt

def reconstruct_from_angiograms(
    dicom_dir: str,
    output_dir: str = '/kaggle/working',
    show_plot: bool = False
) -> None:
    """
    Reconstruct 3D volume from angiograms (DICOM) using original parameters.
    Combines all reconstructions into one 3D model.
    """
    os.makedirs(os.path.join(output_dir, 'CCTA_BP'), exist_ok=True)
    
    # Load DICOM files
    dicom_files = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
    combined_recon = None
    
    for dcm_file in tqdm(dicom_files):
        dcm = pydicom.dcmread(os.path.join(dicom_dir, dcm_file))
        
        # Initialize geometry PER ANGIOGRAM (no shared assumptions)
        geo = tigre.geometry()
        geo.mode = "cone"
        geo.offDetector = np.array([0, 0])
        geo.accuracy = 1
        geo.COR = 0
        geo.rotDetector = np.array([0, 0, 0])
        
        # Detector params from DICOM
        geo.nDetector = np.array([dcm.Rows, dcm.Columns])
        geo.dDetector = np.array([float(dcm.PixelSpacing[0]), 
                                 float(dcm.PixelSpacing[1])])  # mm
        geo.sDetector = geo.nDetector * geo.dDetector
        
        # Fixed volume params (original values)
        geo.nVoxel = np.array([128, 128, 128])
        v_size = 90  # Original fixed size (mm)
        geo.sVoxel = np.array([v_size, v_size, v_size])
        geo.dVoxel = geo.sVoxel / geo.nVoxel
        
        # Geometry from DICOM (fallback to defaults)
        try:
            geo.DSD = float(dcm.DistanceSourceToDetector)  # mm
            geo.DSO = float(dcm.DistanceSourceToPatient)  # mm
        except:
            geo.DSD = 990  # Original default
            geo.DSO = 765
        
        # Get angles (default to 0° if missing)
        try:
            angle_pri = float(dcm.PositionerPrimaryAngle)
            angle_sec = float(dcm.PositionerSecondaryAngle)
        except:
            raise ValueError(f"Missing angle information in {dcm_file}")

        angles = np.array([[angle_pri, angle_sec, 0]]) * (np.pi / 180)
        
        # Projection data (normalize to [0, 1])
        projection = dcm.pixel_array.astype(np.float32)
        projection = (projection - projection.min()) / (projection.max() - projection.min())
        
        # Reconstruct with 1 SIRT iteration (original behavior)
        imgSIRT = sirt(projection[np.newaxis, ...], geo, angles, 1)
        imgSIRT_bin = (imgSIRT > 0).astype(np.int8)  # Binarize
        
        # Combine with previous reconstructions
        if combined_recon is None:
            combined_recon = imgSIRT_bin
        else:
            combined_recon += imgSIRT_bin
        
        # Optional plot per projection
        if show_plot:
            plt.imshow(projection, cmap='gray')
            plt.title(f'Angle: {angle_pri:.1f}°, {angle_sec:.1f}°')
            plt.show()
    
    # Save final combined reconstruction
    file_number = "final"  # Or use a counter if needed
    np.save(os.path.join(output_dir, 'CCTA_BP', f'recon_{file_number}.npy'), combined_recon)
    
    # 3D plot (original behavior)
    if show_plot:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        xyzs = np.where(combined_recon > 0.5)
        ax.scatter(xyzs[0], xyzs[1], xyzs[2], marker='.', s=1)
        plt.show()