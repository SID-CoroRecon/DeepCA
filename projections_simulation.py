## DEMO 18: Arbitrary axis of rotation
#
#
#
# Some modenr CT geometires are starting to be a bit more complex, one of
# the common things being arbitrary axis of rotation i.e. the detector and the
# source can move not in a circular path, but in a "spherical" path.
#
# In TIGRE this has been implemented by defining the rotation with 3
# angles, specifically the ZYZ configuration of Euler angles.
#
#  This demo shows how to use it.
#
# --------------------------------------------------------------------------
# ---------------------------------------------5-----------------------------
# This file is part of the TIGRE Toolbox
# # Copyright (c) 2015, University of Bath and
#                     CERN-European Organization for Nuclear Research
#                     All rights reserved.
#
# License:            Open Source under BSD.
#                     See the full license at
#                     https://github.com/CERN/TIGRE/blob/master/LICENSE
#
# Contact:            tigre.toolbox@gmail.com
# Codes:              https://github.com/CERN/TIGRE/
# Coded by:           Ander Biguri
# --------------------------------------------------------------------------
#%%Initialize
import tigre
import numpy as np
import tigre.algorithms as algs
import matplotlib.pyplot as plt
import os
import random

def generate_deformed_projections_RCA():
    # Lets create a geomtry object
    geo = tigre.geometry()
    # Offsets
    geo.offDetector = np.array([0, 0])  # Offset of Detector            (mm)
    # Auxiliary
    geo.accuracy = 1  # Variable to define accuracy of
    geo.COR = 0  # y direction displacement for
    geo.rotDetector = np.array([0, 0, 0])  # Rotation of the detector, by
    geo.mode = "cone"  # Or 'parallel'. Geometry type.
 
    phantoms_dir = './CCTA_GT/'
    num_phantoms = len(os.listdir(phantoms_dir))
    for i in range(num_phantoms):
        # Detector parameters
        geo.nDetector = np.array([512,512])  # number of pixels              (px)
        d_spacing = 0.2779 + 0.001*np.random.rand()
        geo.dDetector = np.array([d_spacing,d_spacing])  # size of each pixel            (mm)
        geo.sDetector = geo.nDetector * geo.dDetector  # total size of the detector    (mm)
        # Image parameters
        geo.nVoxel = np.array([128,128,128]) # number of voxels              (vx)
        v_size = 90 + 15*np.random.rand()
        geo.sVoxel = np.array([v_size,v_size,v_size]) # total size of the image       (mm)
        geo.dVoxel = geo.sVoxel / geo.nVoxel  # size of each voxel            (mm)

        # Distances
        geo.DSD = 990 + 20*np.random.rand()*random.choice((-1, 1)) # Distance Source Detector      (mm)
        geo.DSO = 765 + 20*np.random.rand()*random.choice((-1, 1)) # Distance Source Origin        (mm)
        geo.offOrigin = np.array([0, 0, 0])  # Offset of image from origin   (mm) #detector view:-z,x,y 

        angle_one_pri = 30 + 12*np.random.rand()*random.choice((-1, 1)) 
        angle_one_sec = 0 + 8*np.random.rand()*random.choice((-1, 1))
      
        angles = np.array([[angle_one_pri,angle_one_sec,0]])
        angles = angles/180*np.pi

        ## Get Image
        file_name = str(i+1) + '.npy'
        phantom = np.load(phantoms_dir + file_name).astype(np.float32)

        ## Project
        projections = tigre.Ax(phantom.copy(), geo, angles) #array
        projections = projections > 0

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.imshow(projections[0], cmap=plt.get_cmap('Greys'))
        # plt.show()
        plt.savefig('./CCTA_first_proj/' + str(i+1) + '.png')
        plt.close()
        
        ## Reconstruct:
        imgSIRT = algs.sirt(projections, geo, angles, 1)
        imgSIRT_one = imgSIRT > 0

        #-8 to 8 mm translation; -10 to 10degrees
        #############################
        # Distances
        geo.DSD = 1060 + 10*np.random.rand()*random.choice((-1, 1)) # Distance Source Detector      (mm)
        geo.DSO = geo.DSO + 3*np.random.rand()*random.choice((-1, 1)) # Distance Source Origin        (mm)
        geo.offOrigin = np.array([8*np.random.rand()*random.choice((-1, 1)),8*np.random.rand()*random.choice((-1, 1)),0])

        angle_two_pri = 0 + 8*np.random.rand()*random.choice((-1, 1))
        angle_two_sec = 30 + 12*np.random.rand()*random.choice((-1, 1))
      
        angles = np.array([[angle_two_pri + 10*np.random.rand()*random.choice((-1, 1)), angle_two_sec + 10*np.random.rand()*random.choice((-1, 1)),0]])
        angles = angles/180*np.pi

        ## Project
        projections = tigre.Ax(phantom.copy(), geo, angles) #array
        projections = projections > 0

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.imshow(projections[0], cmap=plt.get_cmap('Greys'))
        # plt.show()
        plt.savefig('./CCTA_second_proj/' + str(i+1) + '.png')
        plt.close()
        
        ## Reconstruct:
        geo.offOrigin = np.array([0, 0, 0])
        angles = np.array([[angle_two_pri, angle_two_sec,0]])
        angles = angles/180*np.pi
        imgSIRT = algs.sirt(projections, geo, angles, 1)
        imgSIRT_two = imgSIRT > 0

        recon = imgSIRT_one.astype(np.int8) + imgSIRT_two.astype(np.int8)

        np.save("./CCTA_BP/recon_" + str(i+1), recon.astype(np.int8))
        print("save ill_posed " + str(i+1))

        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # xyzs = np.where(recon>0.5)
        # ax.scatter(xyzs[0], xyzs[1], xyzs[2], marker='.')
        # plt.show()

if __name__ == '__main__':
    generate_deformed_projections_RCA()

