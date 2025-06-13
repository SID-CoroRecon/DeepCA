import tigre
import numpy as np
import tigre.algorithms as algs
import matplotlib.pyplot as plt
import os
import random
import nibabel as nib
from scipy.ndimage import zoom
from tqdm import tqdm

def CCTA_split(file_path, output_dir='/kaggle/working', show_plot=False):
    # Create output directories if they don't exist
    os.makedirs(os.path.join(output_dir, 'split_one'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'split_two'), exist_ok=True)
    
    # Extract file number from the path
    file_name = os.path.basename(file_path)
    file_number = file_name.split('.')[0]  # This will get "1" from "1.label.nii.gz"
    
    img_nifti = nib.load(file_path)  # load the nifti file
    voxels_space = img_nifti.header['pixdim'][1:4]  # extract the voxel size
    img = img_nifti.get_fdata()  # get 3D data in numpy array
    data = np.array(img)

    # instead of having 512x512x(no. of slices) and most of the voxels are 0, we zoom the 3D data to the voxel size
    data = zoom(data, (voxels_space[0], voxels_space[1], voxels_space[2]), order=0, mode='nearest') > 0
    pos = np.where(data>0.5)  # get the coordinates of the non-zero voxels
    xyzs = [pos[0], pos[1], pos[2]]  # create a list of the coordinates

    # x-axis normalization
    v_min = np.min(xyzs[0])
    v_max = np.max(xyzs[0])
    xyzs[0] = xyzs[0] - v_min
    x_diff = v_max - v_min  # the actualwidth (x-axis)
    # print(x_diff)

    # y-axis normalization
    v_min = np.min(xyzs[1])
    v_max = np.max(xyzs[1])
    xyzs[1] = xyzs[1] - v_min
    y_diff = v_max - v_min  # the actual height (y-axis)
    # print(y_diff)

    # z-axis normalization
    v_min = np.min(xyzs[2])
    v_max = np.max(xyzs[2])
    xyzs[2] = xyzs[2] - v_min
    z_diff = v_max - v_min  # the actual depth (z-axis)
    # print(z_diff)

    # check if the size of the object is less than 128 and center the object within 128x128x128
    if x_diff < 128 and y_diff < 128 and z_diff < 128:

        # calculate the gap between the object and the edge of the 128x128x128 cube
        x_gap = 128 - (x_diff + 1)
        y_gap = 128 - (y_diff + 1)
        z_gap = 128 - (z_diff + 1)

        # center the object within 128x128x128
        xyzs[0] = xyzs[0] + int(x_gap/2)    # creating offset from x-axis
        xyzs[1] = xyzs[1] + int(y_gap/2)    # creating offset from y-axis
        xyzs[2] = xyzs[2] + int(z_gap/2)    # creating offset from z-axis

        # create 3D binary mask
        data = np.zeros((128,128,128))
        data[xyzs[0],xyzs[1],xyzs[2]] = 1

        # get the first non-zero voxel coordinates (starting point)
        w, h, d = data.shape
        coords = []
        flag = False
        for i in range(w):
            if flag:
                break
            for j in range(h):
                if flag:
                    break
                for k in range(d):
                    if data[i,j,k] > 0:
                        coords.append([i,j,k])
                        flag = True
                        break

        # get the coordinates of the 3x3x3 neighborhood of the starting point
        for [x,y,z] in coords:  # start from the first non-zero voxel (starting point)
            for cx in [x-1,x,x+1]:  # check the 3x3x3 neighborhood
                for cy in [y-1,y,y+1]:
                    for cz in [z-1,z,z+1]:
                        c_coord = [cx,cy,cz]
                        if not (c_coord in coords): # skip if the coordinate is already in the list
                            if cx > -1 and cx < w:
                                if cy > -1 and cy < h:
                                    if cz > -1 and cz < d:
                                        if data[cx,cy,cz] > 0:
                                            coords.append(c_coord)  # dynamically expand the list

        # Isolates the Right Coronary Artery (RCA) from Left Anterior Descending (LAD) in segmented CCTA data
        # Contains Left Anterior Descending (LAD) (original minus largest component)
        coords = np.transpose(np.array(coords))
        data[coords[0],coords[1],coords[2]] = 0     # Removes RCA
        np.save(output_dir + "/split_one/" + file_number, data.astype('int8'))

        if show_plot:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            xyzs = np.where(data>0.5)
            ax.scatter(xyzs[0], xyzs[1], xyzs[2], marker='.')
            plt.show()

        # Contains Right Coronary Artery (RCA) (largest connected component)
        data = data*0
        data[coords[0],coords[1],coords[2]] = 1
        np.save(output_dir + "/split_two/" + file_number, data.astype('int8'))   

        if show_plot:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            xyzs = np.where(data>0.5)
            ax.scatter(xyzs[0], xyzs[1], xyzs[2], marker='.')
            plt.show()

    else:
        print('Failed to split for this data (size is more than 128): ' + file_number)


def generate_deformed_projections_RCA(phantoms_dir, output_dir='/kaggle/working', one_view=False, show_plot=False, save_plot=False):
    # Create output directories if they don't exist
    if save_plot:
        os.makedirs(os.path.join(output_dir, 'CCTA_first_proj'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'CCTA_second_proj'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'CCTA_BP'), exist_ok=True)
    
    geo = tigre.geometry()  # Creates a CT scanner geometry object
    geo.offDetector = np.array([0, 0])  # Detector offset (mm)
    geo.accuracy = 1  # Accuracy level (affects computations)
    geo.COR = 0  # Center-of-rotation offset (y-direction)
    geo.rotDetector = np.array([0, 0, 0])  # Detector rotation angles (degrees)
    geo.mode = "cone"  # Cone-beam CT geometry (vs. parallel-beam)

    ID = 0
    for file_name in tqdm(os.listdir(phantoms_dir)):
        # Extract file number from the filename
        label_number = file_name.split('.')[0]  # This will get "1" from "1.npy"
        
        geo.nDetector = np.array([512, 512])  # Detector resolution (px)
        d_spacing = 0.2779 + 0.001*np.random.rand()  # Pixel spacing (mm, randomized)
        geo.dDetector = np.array([d_spacing, d_spacing])  # Pixel size (mm)
        geo.sDetector = geo.nDetector * geo.dDetector  # Total detector size (mm)

        geo.nVoxel = np.array([128, 128, 128])  # Volume dimensions (voxels)
        v_size = 90 + 15*np.random.rand()  # Physical volume size (mm, randomized)
        geo.sVoxel = np.array([v_size, v_size, v_size])  # Total volume size (mm)
        geo.dVoxel = geo.sVoxel / geo.nVoxel  # Voxel size (mm)

        geo.DSD = 990 + 20*np.random.rand()*random.choice((-1, 1))  # Source-detector distance (mm)
        geo.DSO = 765 + 20*np.random.rand()*random.choice((-1, 1))  # Source-origin distance (mm)
        geo.offOrigin = np.array([0, 0, 0])  # Volume offset (mm)

        angle_one_pri = 30 + 12*np.random.rand()*random.choice((-1, 1))  # Primary angle (degrees)
        angle_one_sec = 0 + 8*np.random.rand()*random.choice((-1, 1))  # Secondary angle (degrees)
        angles = np.array([[angle_one_pri, angle_one_sec, 0]])  # Combine angles
        angles = angles/180*np.pi  # Convert to radians

        ## Get Image
        phantom = np.load(os.path.join(phantoms_dir, file_name)).astype(np.float32)

        ## Project
        projections = tigre.Ax(phantom.copy(), geo, angles) #array
        projections = projections > 0

        if show_plot or save_plot:
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.imshow(projections[0], cmap=plt.get_cmap('Greys'))
            if save_plot:
                plt.savefig(os.path.join(output_dir, 'CCTA_first_proj', f'{label_number}.png'))
            if show_plot:
                plt.show()
            plt.close(fig)
        
        imgSIRT = algs.sirt(projections, geo, angles, 1)  # Reconstruct using SIRT
        imgSIRT_one = imgSIRT > 0  # Binarize

        #-8 to 8 mm translation; -10 to 10degrees
        #############################

        geo.DSD = 1060 + 10*np.random.rand()*random.choice((-1, 1))  # Change source-detector distance
        geo.DSO = geo.DSO + 3*np.random.rand()*random.choice((-1, 1))  # Change source-origin distance
        geo.offOrigin = np.array([8*np.random.rand()*random.choice((-1, 1)), 8*np.random.rand()*random.choice((-1, 1)), 0])  # Random offset

        angle_two_pri = 0 + 8*np.random.rand()*random.choice((-1, 1))  # New primary angle
        angle_two_sec = 30 + 12*np.random.rand()*random.choice((-1, 1))  # New secondary angle
        angles = np.array([[angle_two_pri + 10*np.random.rand()*random.choice((-1, 1)), angle_two_sec + 10*np.random.rand()*random.choice((-1, 1)), 0]])
        angles = angles/180*np.pi  # Convert to radians

        projections = tigre.Ax(phantom.copy(), geo, angles)  # Second projection
        projections = projections > 0  # Binarize

        if show_plot or save_plot:
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.imshow(projections[0], cmap=plt.get_cmap('Greys'))
            if save_plot:
                plt.savefig(os.path.join(output_dir, 'CCTA_second_proj', f'{label_number}.png'))
            if show_plot:
                plt.show()
            plt.close(fig)
        
        # Reconstruct:
        geo.offOrigin = np.array([0, 0, 0])
        angles = np.array([[angle_two_pri, angle_two_sec,0]])
        angles = angles/180*np.pi
        imgSIRT = algs.sirt(projections, geo, angles, 1)  # Reconstruct again
        imgSIRT_two = imgSIRT > 0  # Binarize

        if one_view:
            recon_one = imgSIRT_one.astype(np.int8)
            recon_two = imgSIRT_two.astype(np.int8)
            np.save(output_dir + "/CCTA_BP/recon_"+ label_number + "_" + str(ID), recon_one.astype(np.int8))
            ID += 1
            np.save(output_dir + "/CCTA_BP/recon_"+ label_number + "_" + str(ID), recon_two.astype(np.int8))
            ID += 1
        else:
            recon = imgSIRT_one.astype(np.int8) + imgSIRT_two.astype(np.int8)  # Combine both reconstructions
            np.save(output_dir + "/CCTA_BP/recon_" + label_number + "_" + str(ID), recon.astype(np.int8))    # Save combined volume
            ID += 1
        print("save ill_posed: " + label_number)

        if show_plot:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            xyzs = np.where(recon>0.5)
            ax.scatter(xyzs[0], xyzs[1], xyzs[2], marker='.')
            plt.show()

if __name__ == '__main__':
    base_dir = 'data/RAW_DATA'
    directories = ['1-200', '201-400', '401-600', '601-800', '801-1000']

    for dir_name in tqdm(directories, desc="Processing directories"):
        current_dir = os.path.join(base_dir, dir_name)
        for file_name in tqdm(os.listdir(current_dir), desc=f"Processing files in {dir_name}", leave=False):
            if file_name.endswith('.label.nii.gz'):
                file_path = os.path.join(current_dir, file_name)
                CCTA_split(file_path, 'data')
    print("CCTA splits completed")
    print("Number of files split: ", len(os.listdir('data/split_two/')))
    
    # After all CCTA splits are done, run the projection generation
    generate_deformed_projections_RCA('data/split_two/', 'data', save_plot=True)  # path to the phantom images
    print("Projection generation completed")
    print("Number of files projected: ", len(os.listdir('data/CCTA_BP/')))