import torch
import random
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
from skimage.measure import marching_cubes
from stl import mesh as stl_mesh


def set_random_seed(seed, deterministic=True):
    """
    Set the random seed for the experiment.
    """
    torch.cuda.empty_cache()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def generation_eval(outputs,labels):
    l1_criterion = nn.L1Loss()
    l1_loss = l1_criterion(outputs, labels)
    return l1_loss


def do_evaluation(dataloader, model, device, discriminator):
    """
    Evaluate the model on the test set.
    """
    model.eval()
    discriminator.eval()
    
    l1_losses = []
    G_losses = []
    
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data[0].float().to(device), data[1].float().to(device)
            # calculate outputs by running images through the network
            outputs = model(inputs)

            DG_score = discriminator(torch.cat((inputs, F.sigmoid(outputs)), 1)).mean() # D(G(z))
            G_loss = -DG_score
            G_losses.append(G_loss.item())

            l1_loss = generation_eval(outputs,labels)
            l1_losses.append(l1_loss.item())
            
    return np.mean(G_losses), np.mean(l1_losses)


def load_checkpoint(model, discriminator, optimizer, D_optimizer, checkpoint_path):
    """Load model checkpoint if it exists.
    
    Args:
        model: Generator model
        discriminator: Discriminator model
        optimizer: Generator optimizer
        D_optimizer: Discriminator optimizer
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        start_epoch: The epoch to start training from
    """
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['network'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        D_optimizer.load_state_dict(checkpoint['D_optimizer'])
        # Extract epoch number from checkpoint filename
        start_epoch = int(checkpoint_path.split('_')[-1].split('.')[0])
        print(f"Resuming training from epoch {start_epoch}")
        return start_epoch
    return 0

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