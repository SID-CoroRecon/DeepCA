import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import autograd
import numpy as np
import random
import copy
import gc
import os
import sys
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# Add current directory to path for imports
sys.path.append(r'D:\AI\Internships\SID Research\models\DeepCA')     # TODO: change this to the path of the project

from networks.generator import Generator
from networks.discriminator import Discriminator
from load_volume_data_RCA import Dataset
from samples_parameters import get_samples_parameters

output_dir = r'D:\AI\Internships\SID Research\models\DeepCA\outputs_results'  # TODO: change this to the path of the project
label_dir = r'D:\AI\Internships\SID Research\models\DeepCA\datasets\CCTA_GT'     # TODO: change this to the path of the data
BP_dir = r'D:\AI\Internships\SID Research\models\DeepCA\datasets\CCTA_BP'       # TODO: change this to the path of the data

# Create output directories if they don't exist
os.makedirs(os.path.join(output_dir, 'outputs_results/checkpoints'), exist_ok=True)

# Get dataset parameters
SAMPLES_PARA = get_samples_parameters(label_dir)

LEARNING_RATE = 1e-5
MAX_EPOCHS = 15
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

def do_evaluation(dataloader, model, device, discriminator):
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

            # Clean up
            del inputs, labels, outputs, DG_score, G_loss, l1_loss
            torch.cuda.empty_cache()

    return np.mean(G_losses), np.mean(l1_losses)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    # Initialize models
    model = Generator(in_channels=1, num_filters=64, class_num=1).to(device)
    discriminator = Discriminator(device, 2).to(device)

    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()

    batch_size = BATCH_SIZE

    #Dataset setup
    training_set = Dataset(BP_dir, label_dir, SAMPLES_PARA['train_index'], linux=False)
    trainloader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True, pin_memory=True)

    val_set = Dataset(BP_dir, label_dir, SAMPLES_PARA['validation_index'], linux=False)
    validationloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True, pin_memory=True)

    test_set = Dataset(BP_dir, label_dir, SAMPLES_PARA['test_index'], linux=False)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True, pin_memory=True)

    #G and D optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.9))
    D_optimizer = optim.Adam(discriminator.parameters(), 
                                    lr=LEARNING_RATE, betas=(0.5, 0.9))

    best_validation_loss = np.Inf
    best_model_state = None
    best_D_model_state = None
    early_stop_count_val = 0
    num_critics = 2

    # Create progress bar for epochs
    epoch_pbar = tqdm(range(MAX_EPOCHS), desc='Training Progress', position=0)
    
    for epoch in epoch_pbar:
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
        Wasserstein_Ds_cur= []

        # Create progress bar for batches
        batch_pbar = tqdm(enumerate(trainloader), total=len(trainloader), 
                         desc=f'Epoch {epoch+1}/{MAX_EPOCHS}', position=1, leave=False)

        for i, data in batch_pbar:
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
            gc.collect()
            torch.cuda.empty_cache()

            D_optimizer.zero_grad()
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(inputs)
                DX_score = discriminator(torch.cat((inputs, labels), 1)).mean() # D(x)
                DG_score = discriminator(torch.cat((inputs, outputs), 1).detach()).mean() # D(G(z))

                # Train with gradient penalty
                gradient_penalty = calculate_gradient_penalty(torch.cat((inputs, labels), 1), torch.cat((inputs, outputs), 1).detach(), discriminator, device)

                D_loss = (DG_score - DX_score + gradient_penalty)
                Wasserstein_D = DX_score - DG_score

            scaler.scale(D_loss).backward()
            scaler.step(D_optimizer)
            scaler.update()

            D_losses.append(D_loss.detach().item())
            D_losses_cur.append(D_loss.detach().item())
            Wasserstein_Ds.append(Wasserstein_D.detach().item())
            Wasserstein_Ds_cur.append(Wasserstein_D.detach().item())

            ####################            

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
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = model(inputs)
                    DG_score = discriminator(torch.cat((inputs, outputs), 1)).mean() # D(G(z))
                    G_loss = -DG_score
                    l1_loss = generation_eval(outputs,labels)
                    combined_loss = G_loss + l1_loss*100

                scaler.scale(combined_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                G_losses.append(G_loss.detach().item())
                l1_losses.append(l1_loss.detach().item())
                combined_losses.append(combined_loss.detach().item())

                # Update batch progress bar with current metrics
                batch_pbar.set_postfix({
                    'D_loss': f'{np.mean(D_losses_cur):.4f}',
                    'G_loss': f'{G_loss.detach().item():.4f}',
                    'L1_loss': f'{l1_loss.detach().item():.4f}',
                    'W_D': f'{np.mean(Wasserstein_Ds_cur):.4f}'
                })

                D_losses_cur = []
                Wasserstein_Ds_cur = []

            # Clean up
            del inputs, labels, outputs, DX_score, DG_score, gradient_penalty, D_loss, Wasserstein_D
            if (i+1) % num_critics == 0:
                del G_loss, l1_loss, combined_loss
            torch.cuda.empty_cache()

        #do validation
        G_loss_val, l1_loss_val = do_evaluation(validationloader, model, device, discriminator)
        combined_loss_val = G_loss_val + l1_loss_val*100
        validation_loss = l1_loss_val

        # Update epoch progress bar with validation metrics
        epoch_pbar.set_postfix({
            'Val_L1': f'{l1_loss_val:.4f}',
            'Val_G': f'{G_loss_val:.4f}',
            'Train_D': f'{np.mean(D_losses):.4f}',
            'Train_G': f'{np.mean(G_losses):.4f}',
            'Train_L1': f'{np.mean(l1_losses):.4f}'
        })

        if validation_loss < best_validation_loss:

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
            best_model_state = copy.deepcopy(model.state_dict())
            best_D_model_state = copy.deepcopy(discriminator.state_dict())
        else:
            early_stop_count_val += 1

        if early_stop_count_val >= 20:
            print('\nEarly stopping triggered!')
            break
            
    # evaluate on test set
    print('\n############################### Testing evaluation on best trained model')
    model.load_state_dict(best_model_state)
    discriminator.load_state_dict(best_D_model_state)
    G_loss_test, l1_loss_test = do_evaluation(testloader, model, device, discriminator)
    test_loss = G_loss_test + l1_loss_test*100

    print('Test Results:')
    print(f'Test Loss: {test_loss:.8f}')
    print(f'G Loss: {G_loss_test:.8f}')
    print(f'L1 Loss: {l1_loss_test:.8f}')

if __name__ == '__main__':
    set_random_seed(1, False)
    main()