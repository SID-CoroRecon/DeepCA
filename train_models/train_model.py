import torch
import torch.optim as optim
import numpy as np
import copy
import os
import gc
import sys

sys.path.append('/kaggle/input/deepca/pytorch/default/1')     # TODO: change this to the path of the project

try:
    from generator import Generator
    from discriminator import Discriminator
except:
    from networks.generator import Generator
    from networks.discriminator import Discriminator
from load_volume_data_RCA import Dataset
from samples_parameters import get_samples_parameters
from utils import *


output_dir = '/kaggle/working/'  # TODO: change this to the path of the project
label_dir = '/kaggle/input/deepca-preprocessed-dataset/data/split_two'     # TODO: change this to the path of the data
BP_dir = '/kaggle/input/deepca-preprocessed-dataset/data/CCTA_BP'       # TODO: change this to the path of the data

# Create output directories if they don't exist
os.makedirs(os.path.join(output_dir, 'outputs_results/checkpoints'), exist_ok=True)

# Get dataset parameters
SAMPLES_PARA = get_samples_parameters(label_dir)

LEARNING_RATE = 1e-4
EXTRA_EPOCHS = 10
LAMBDA_TERM = 10
BATCH_SIZE = 1


def main():
    print("Starting main...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = Generator(in_channels=1, num_filters=64, class_num=1).to(device)
    discriminator = Discriminator(device, 2).to(device)

    # Clear memory before starting
    gc.collect()
    torch.cuda.empty_cache()

    #Dataset setup
    training_set = Dataset(BP_dir, label_dir, SAMPLES_PARA['train_index'], linux=True)
    trainloader = torch.utils.data.DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    val_set = Dataset(BP_dir, label_dir, SAMPLES_PARA['validation_index'], linux=True)
    validationloader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    test_set = Dataset(BP_dir, label_dir, SAMPLES_PARA['test_index'], linux=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    print("Dataset setup complete.")

    #G and D optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.9))
    D_optimizer = optim.Adam(discriminator.parameters(), 
                                    lr=1e-4, betas=(0.5, 0.9))

    # Try to load the latest checkpoint
    checkpoint_dir = '/kaggle/input/deepca-training-6/outputs_results/checkpoints'
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

        # Determine which batches to use for generator training in this epoch
        # Alternate between even and odd batches each epoch
        generator_batches = set(range(epoch % 2, len(trainloader), 2))
        
        for i, data in enumerate(trainloader, 0):
            # Clear memory at the start of each batch
            gc.collect()
            torch.cuda.empty_cache()

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].float().to(device), data[1].float().to(device)

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
            gradient_penalty = calculate_gradient_penalty(torch.cat((inputs, labels), 1), torch.cat((inputs, outputs), 1).detach(), discriminator, device, BATCH_SIZE, LAMBDA_TERM)

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
            # Generator update - only on predetermined batches for this epoch
            if i in generator_batches:
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

                combined_loss = G_loss + l1_loss*100
                combined_losses.append(combined_loss.detach().item())

                # update parameters
                combined_loss.backward()
                optimizer.step()

                D_losses_cur = []
                Wasserstein_Ds_cur = []

                # Print losses for this batch
                print(f'Batch {i+1}/{len(trainloader)} - G_loss: {G_loss.item():.4f}, D_loss: {D_loss.item():.4f}, L1_loss: {l1_loss.item():.4f}, Combined_loss: {combined_loss.item():.4f}')

        # Print generator training coverage for this epoch
        print(f'Epoch {epoch+1}: Generator trained on {len(generator_batches)}/{len(trainloader)} batches')
        print(f'Generator batches: {sorted(list(generator_batches))}')

        #do validation
        G_loss_val, l1_loss_val = do_evaluation(validationloader, model, device, discriminator)
        combined_loss_val = G_loss_val + l1_loss_val*100
        validation_loss = l1_loss_val

        # Print epoch summary
        print(f'\nEpoch {epoch+1}/{start_epoch + EXTRA_EPOCHS} Summary:')
        print(f'Training - Avg G_loss: {np.mean(G_losses):.4f}, Avg D_loss: {np.mean(D_losses):.4f}, Avg L1_loss: {np.mean(l1_losses):.4f}')
        print(f'Validation - G_loss: {G_loss_val:.4f}, L1_loss: {l1_loss_val:.4f}, Combined_loss: {combined_loss_val:.4f}\n')

        if (epoch + 1) == start_epoch + EXTRA_EPOCHS:
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
    G_loss_test, l1_loss_test = do_evaluation(testloader, model, device, discriminator)
    test_loss = G_loss_test + l1_loss_test*100

    print('Testdataset Evaluation - test loss: {0:3.8f}, G loss: {1:3.8f}, l1 loss: {2:3.8f}'
                .format(test_loss, G_loss_test.item(), l1_loss_test.item()))


if __name__ == '__main__':
    set_random_seed(1, False)
    main()