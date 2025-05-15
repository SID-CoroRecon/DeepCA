import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import autograd
import numpy as np
import random
import copy
import os
import gc
import sys

sys.path.append('/kaggle/input/deepca/pytorch/default/1')     # TODO: change this to the path of the project

from generator import Generator
from discriminator import Discriminator
from load_volume_data_RCA import Dataset
from samples_parameters import get_samples_parameters

output_dir = '/kaggle/working/'  # TODO: change this to the path of the project
label_dir = '/kaggle/input/deepca-preprocessed-dataset/data/split_two'     # TODO: change this to the path of the data
BP_dir = '/kaggle/input/deepca-preprocessed-dataset/data/CCTA_BP'       # TODO: change this to the path of the data

# Create output directories if they don't exist
os.makedirs(os.path.join(output_dir, 'outputs_results/checkpoints'), exist_ok=True)

# Get dataset parameters
SAMPLES_PARA = get_samples_parameters(label_dir)

LEARNING_RATE = 1e-4
EXTRA_EPOCHS = 10

BATCH_SIZE = 1

def set_random_seed(seed, deterministic=True):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    torch.cuda.empty_cache()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def calculate_gradient_penalty(real_images, fake_images, discriminator, device, batch_size=BATCH_SIZE):
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

    lambda_term = 10
    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_term
    return grad_penalty

def generation_eval(outputs,labels):
    l1_criterion = nn.L1Loss() #nn.MSELoss()

    l1_loss = l1_criterion(outputs, labels)

    return l1_loss

def calculate_dice(pred, target, smooth=1.0):
    """Calculate DICE coefficient for 3D medical images.
    
    Args:
        pred: Predicted segmentation mask (B, C, D, H, W)
        target: Ground truth segmentation mask (B, C, D, H, W)
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        dice: DICE coefficient
    """
    # Apply sigmoid to get probabilities
    pred = F.sigmoid(pred)
    
    # Convert to binary mask
    pred = (pred > 0.5).float()
    
    # Calculate intersection and union
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice.item()


def do_evaluation(dataloader, model, device, discriminator):
    model.eval()
    discriminator.eval()
    
    l1_losses = []
    G_losses = []
    dice_scores = []
    
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
            
            # Calculate DICE score
            dice_score = calculate_dice(outputs, labels)
            dice_scores.append(dice_score)
            
    return np.mean(G_losses), np.mean(l1_losses), np.mean(dice_scores)

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

def main():
    print("Starting main...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = Generator(in_channels=1, num_filters=64, class_num=1).to(device)
    discriminator = Discriminator(device, 2).to(device)

    # Clear memory before starting
    gc.collect()
    torch.cuda.empty_cache()

    batch_size = BATCH_SIZE

    #Dataset setup
    training_set = Dataset(BP_dir, label_dir, SAMPLES_PARA['train_index'], linux=True)
    trainloader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    val_set = Dataset(BP_dir, label_dir, SAMPLES_PARA['validation_index'], linux=True)
    validationloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    test_set = Dataset(BP_dir, label_dir, SAMPLES_PARA['test_index'], linux=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    print("Dataset setup complete.")

    #G and D optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.9))
    D_optimizer = optim.Adam(discriminator.parameters(), 
                                    lr=1e-4, betas=(0.5, 0.9))

    # Try to load the latest checkpoint
    checkpoint_dir = '/kaggle/input/deepca-training/outputs_results/checkpoints'
    latest_checkpoint = None
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.tar')]
        if checkpoints:
            # Extract epoch numbers and find the maximum
            epoch_numbers = [int(f.split('_')[-1].split('.')[0]) for f in checkpoints]
            max_epoch = max(epoch_numbers)
            latest_checkpoint = os.path.join(checkpoint_dir, f'Epoch_{max_epoch}.tar')
            print(f"Found checkpoints from epochs: {epoch_numbers}")
            print(f"Loading checkpoint from epoch {max_epoch}")
    
    start_epoch = load_checkpoint(model, discriminator, optimizer, D_optimizer, latest_checkpoint) if latest_checkpoint else 0

    best_validation_loss = np.Inf

    best_model_state = None
    best_D_model_state = None

    early_stop_count_val = 0
    num_critics = 2

    print("Starting training...")

    for epoch in range(start_epoch, start_epoch + EXTRA_EPOCHS):
        gc.collect()
        torch.cuda.empty_cache()

        model.train()
        discriminator.train()
        l1_losses = []
        D_losses = []
        D_losses_cur = []
        G_losses = []
        combined_losses = []
        Wasserstein_Ds = []
        Wasserstein_Ds_cur = []

        for i, data in enumerate(trainloader, 0):
            # Clear memory at the start of each batch
            gc.collect()
            torch.cuda.empty_cache()

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].float().to(device), data[1].float().to(device)

            ######################## CCTA/VG training
            ####### adversarial loss
            # Requires grad, Generator requires_grad = False
            for p in discriminator.parameters():
                p.requires_grad = True
            for p in model.parameters():
                p.requires_grad = False
              
            # Clear memory after parameter updates
            gc.collect()
            torch.cuda.empty_cache()

            D_optimizer.zero_grad()
            outputs = model(inputs)

            # Clear memory after forward pass
            gc.collect()
            torch.cuda.empty_cache()

            # Classify the generated and real batch images
            DX_score = discriminator(torch.cat((inputs, labels), 1)).mean() # D(x)
            DG_score = discriminator(torch.cat((inputs, outputs), 1).detach()).mean() # D(G(z))

            # Train with gradient penalty
            gradient_penalty = calculate_gradient_penalty(torch.cat((inputs, labels), 1), torch.cat((inputs, outputs), 1).detach(), discriminator, device)

            D_loss = (DG_score - DX_score + gradient_penalty)
            Wasserstein_D = DX_score - DG_score

            # Update parameters
            D_loss.backward()
            D_optimizer.step()
            
            # Clear memory after backward pass
            gc.collect()
            torch.cuda.empty_cache()

            D_losses.append(D_loss.detach().item())
            D_losses_cur.append(D_loss.detach().item())
            Wasserstein_Ds.append(Wasserstein_D.detach().item())
            Wasserstein_Ds_cur.append(Wasserstein_D.detach().item())

            # Clear memory after storing losses
            gc.collect()
            torch.cuda.empty_cache()

            ###### generator loss
            # Generator update
            if (i+1) % num_critics == 0:
                for p in discriminator.parameters():
                    p.requires_grad = False  # to avoid computation
                for p in model.parameters():
                    p.requires_grad = True
                gc.collect()
                torch.cuda.empty_cache()

                optimizer.zero_grad()
                outputs = model(inputs)

                DG_score = discriminator(torch.cat((inputs, outputs), 1)).mean() # D(G(z))
                G_loss = -DG_score
                G_losses.append(G_loss.detach().item())

                l1_loss = generation_eval(outputs,labels)
                l1_losses.append(l1_loss.detach().item())

                ###################
                combined_loss = G_loss + l1_loss*100
                combined_losses.append(combined_loss.detach().item())

                # update parameters
                combined_loss.backward()
                optimizer.step()

                D_losses_cur = []
                Wasserstein_Ds_cur = []

                # Print losses for this batch
                print(f'Batch {i+1}/{len(trainloader)} - G_loss: {G_loss.item():.4f}, D_loss: {D_loss.item():.4f}, L1_loss: {l1_loss.item():.4f}, Combined_loss: {combined_loss.item():.4f}')

        #do validation
        G_loss_val, l1_loss_val, dice_val = do_evaluation(validationloader, model, device, discriminator)
        combined_loss_val = G_loss_val + l1_loss_val*100
        validation_loss = l1_loss_val

        # Print epoch summary
        print(f'\nEpoch {epoch+1}/{start_epoch + EXTRA_EPOCHS} Summary:')
        print(f'Training - Avg G_loss: {np.mean(G_losses):.4f}, Avg D_loss: {np.mean(D_losses):.4f}, Avg L1_loss: {np.mean(l1_losses):.4f}')
        print(f'Validation - G_loss: {G_loss_val:.4f}, L1_loss: {l1_loss_val:.4f}, Combined_loss: {combined_loss_val:.4f}, DICE: {dice_val:.4f}\n')

        if (epoch + 1) % 10 == 0:
            model.eval()
            torch.save(
                    {
                        "network": model.state_dict(),
                        "discriminator": discriminator.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "D_optimizer": D_optimizer.state_dict(),
                    },
                    output_dir + 'outputs_results/checkpoints/Epoch_' + str(epoch+1) + '.tar',
                )

        # early stopping if validation loss is increasing or staying the same after five epoches
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            early_stop_count_val = 0

            # # Save a checkpoint of the best validation loss model so far
            # # print("saving this best validation loss model so far!")
            best_model_state = copy.deepcopy(model.state_dict())
            best_D_model_state = copy.deepcopy(discriminator.state_dict())
        else:
            early_stop_count_val += 1
            # print('no improvement on validation at this epoch, continue training...')

        if early_stop_count_val >= 20:
            print('early stopping validation!!!')
            break
            
    # evaluate on test set
    print('\n############################### testing evaluation on best trained model so far')
    model.load_state_dict(best_model_state)
    discriminator.load_state_dict(best_D_model_state)
    G_loss_test, l1_loss_test, dice_test = do_evaluation(testloader, model, device, discriminator)
    test_loss = G_loss_test + l1_loss_test*100

    print('Testdataset Evaluation - test loss: {0:3.8f}, G loss: {1:3.8f}, l1 loss: {2:3.8f}, DICE: {3:3.8f}'
                .format(test_loss, G_loss_test.item(), l1_loss_test.item(), dice_test))

if __name__ == '__main__':
    set_random_seed(1, False)
    main()