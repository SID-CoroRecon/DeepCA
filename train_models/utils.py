import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import os


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


def calculate_gradient_penalty(real_images, fake_images, discriminator, device, batch_size, lambda_term):
    """
    Calculate the gradient penalty for the discriminator.
    """
    eta = torch.FloatTensor(batch_size,2,1,1,1).uniform_(0,1).to(device)
    eta = eta.expand(batch_size, real_images.size(1), real_images.size(2), real_images.size(3), real_images.size(4))

    interpolated = eta * fake_images + ((1 - eta) * real_images)

    # define it to calculate gradient
    interpolated = Variable(interpolated, requires_grad=True)

    # calculate probability of interpolated examples
    prob_interpolated = discriminator(interpolated)

    # calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(
                               prob_interpolated.size()).to(device),
                           create_graph=True, retain_graph=True)[0]

    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_term
    return grad_penalty


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