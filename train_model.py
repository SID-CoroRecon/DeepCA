import torch
import torch.optim as optim
import numpy as np
import gc
import sys
import yaml

def load_config():
    with open('/kaggle/working/DeepCA/config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config
config = load_config()

sys.path.append(config['path']['model_dir'])

from networks.generator import Generator
from networks.discriminator import Discriminator
from utils.data import get_data_loader
from utils.helpers import *
from utils.losses import *

output_dir = config['path']['output_dir']
label_dir = config['path']['label_dir']
BP_dir = config['path']['BP_dir']

# Hyperparameters
LEARNING_RATE = config['hyperparameters']['learning_rate']
EXTRA_EPOCHS = config['hyperparameters']['extra_epochs']
LAMBDA_TERM = config['hyperparameters']['lambda_term']
BATCH_SIZE = config['hyperparameters']['batch_size']
BETAS = config['hyperparameters']['betas']


def main():
    print("Starting main...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = Generator(in_channels=1, num_filters=64, class_num=1).to(device)
    discriminator = Discriminator(device, 2).to(device)

    # Clear memory before starting
    gc.collect()
    torch.cuda.empty_cache()

    #Dataset setup
    train_loader, val_loader, test_loader = get_data_loader(BP_dir, label_dir, BATCH_SIZE)
    print("Dataset setup complete.")

    #G and D optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=BETAS)
    D_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=BETAS)

    print("Loading checkpoint...")
    checkpoint_path = config['path']['checkpoint_path']
    start_epoch = load_checkpoint(model, discriminator, optimizer, D_optimizer, checkpoint_path)

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
        generator_batches = set(range(epoch % 2, len(train_loader), 2))
        
        for i, data in enumerate(train_loader, 0):
            # Clear memory at the start of each batch
            gc.collect()
            torch.cuda.empty_cache()

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].float().to(device), data[1].float().to(device)

            # Discriminator update
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
            gradient_penalty = calculate_gradient_penalty(torch.cat((inputs, labels), 1), torch.cat((inputs, outputs), 1).detach(), 
                                                          discriminator, device, BATCH_SIZE, LAMBDA_TERM)

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
                print(f'Batch {i+1}/{len(train_loader)} - G_loss: {G_loss.item():.4f}, D_loss: {D_loss.item():.4f}, '
                      f'L1_loss: {l1_loss.item():.4f}, Combined_loss: {combined_loss.item():.4f}')

        # Print generator training coverage for this epoch
        print(f'Epoch {epoch+1}: Generator trained on {len(generator_batches)}/{len(train_loader)} batches')
        print(f'Generator batches: {sorted(list(generator_batches))}')

        # Evaluate on validation set
        G_loss_val, l1_loss_val = do_evaluation(val_loader, model, device, discriminator)
        combined_loss_val = G_loss_val + l1_loss_val*100

        # Print epoch summary
        print(f'\nEpoch {epoch+1}/{start_epoch + EXTRA_EPOCHS} Summary:')
        print(f'Training - Avg G_loss: {np.mean(G_losses):.4f}, Avg D_loss: {np.mean(D_losses):.4f}, Avg L1_loss: {np.mean(l1_losses):.4f}')
        print(f'Validation - G_loss: {G_loss_val:.4f}, L1_loss: {l1_loss_val:.4f}, Combined_loss: {combined_loss_val:.4f}\n')

        # Save checkpoint in the last epoch
        if (epoch + 1) == start_epoch + EXTRA_EPOCHS:
            model.eval()
            torch.save(
                    {
                        "network": model.state_dict(),
                        "discriminator": discriminator.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "D_optimizer": D_optimizer.state_dict(),
                    },
                    output_dir + '/checkpoints/Epoch_' + str(epoch+1) + '.tar',
                )
            
    # evaluate on test set
    print('Evaluating on test set...')
    G_loss_test, l1_loss_test = do_evaluation(test_loader, model, device, discriminator)
    test_loss = G_loss_test + l1_loss_test*100
    print(f'Test loss: {test_loss:.4f}, G_loss: {G_loss_test:.4f}, l1_loss: {l1_loss_test:.4f}')

if __name__ == '__main__':
    set_random_seed(1, False)
    main()