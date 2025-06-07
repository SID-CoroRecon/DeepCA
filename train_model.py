import os
import sys
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import gc
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

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

# Configuration
output_dir = config['path']['output_dir']
label_dir = config['path']['label_dir']
BP_dir = config['path']['BP_dir']
LEARNING_RATE = config['hyperparameters']['learning_rate']
EXTRA_EPOCHS = config['hyperparameters']['extra_epochs']
LAMBDA_TERM = config['hyperparameters']['lambda_term']
BATCH_SIZE = config['hyperparameters']['batch_size']
BETAS = config['hyperparameters']['betas']
CHECKPOINT_PATH = config['path']['checkpoint_path']

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "5554"
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def ddp_main(rank, world_size):
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu') 

    model = Generator(in_channels=1, num_filters=64, class_num=1).to(device)
    discriminator = Discriminator(device, 2).to(device)

    model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None, find_unused_parameters=True)
    discriminator = DDP(discriminator, device_ids=[rank] if torch.cuda.is_available() else None, find_unused_parameters=True)

    train_loader, val_loader, test_loader = get_data_loader(
        BP_dir, label_dir, BATCH_SIZE, rank=rank, world_size=world_size
    )

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=BETAS)
    D_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=BETAS)

    start_epoch = load_checkpoint(model.module, discriminator.module, optimizer, D_optimizer, CHECKPOINT_PATH)

    for epoch in range(start_epoch, start_epoch + EXTRA_EPOCHS):
        gc.collect()
        torch.cuda.empty_cache()

        train_loader.sampler.set_epoch(epoch)
        val_loader.sampler.set_epoch(epoch)

        model.train()
        discriminator.train()
        l1_losses, D_losses, G_losses, combined_losses, Wasserstein_Ds = [], [], [], [], []

        generator_batches = set(range(epoch % 2, len(train_loader), 2))

        for i, data in enumerate(train_loader):
            inputs, labels = data[0].float().to(device, non_blocking=True), data[1].float().to(device, non_blocking=True)

            # === Train Discriminator ===
            for p in discriminator.parameters(): p.requires_grad = True
            for p in model.parameters(): p.requires_grad = False

            D_optimizer.zero_grad()
            with torch.no_grad():
                outputs = model(inputs)

            real_pair = torch.cat((inputs, labels), 1)
            fake_pair = torch.cat((inputs, outputs), 1)

            DX_score = discriminator(real_pair).mean()
            DG_score = discriminator(fake_pair.detach()).mean()

            gradient_penalty = calculate_gradient_penalty(
                real_pair, fake_pair.detach(), discriminator, device, BATCH_SIZE, LAMBDA_TERM
            )
            D_loss = (DG_score - DX_score + gradient_penalty)
            Wasserstein_D = DX_score - DG_score

            D_loss.backward()
            D_optimizer.step()

            D_losses.append(D_loss.item())
            Wasserstein_Ds.append(Wasserstein_D.item())

            # === Train Generator ===
            if i in generator_batches:
                for p in discriminator.parameters(): p.requires_grad = False
                for p in model.parameters(): p.requires_grad = True

                optimizer.zero_grad()
                outputs = model(inputs)
                DG_score = discriminator(torch.cat((inputs, outputs), 1)).mean()
                G_loss = -DG_score
                G_losses.append(G_loss.item())

                l1_loss = generation_eval(outputs, labels)
                l1_losses.append(l1_loss.item())

                combined_loss = G_loss + l1_loss * 100
                combined_losses.append(combined_loss.item())

                combined_loss.backward()
                optimizer.step()

                print(f'[RANK {rank}] Batch {i+1}/{len(train_loader)} - G_loss: {G_loss.item():.4f}, '
                      f'D_loss: {D_loss.item():.4f}, L1_loss: {l1_loss.item():.4f}, '
                      f'Combined_loss: {combined_loss.item():.4f}')

        if rank == 0:
            print(f'[RANK 0] Epoch {epoch+1} Summary: '
                  f'Avg G_loss: {np.mean(G_losses):.4f}, Avg D_loss: {np.mean(D_losses):.4f}, '
                  f'Avg L1_loss: {np.mean(l1_losses):.4f}')

        G_loss_val, l1_loss_val = do_evaluation(val_loader, model.module, device, discriminator.module)
        combined_loss_val = G_loss_val + l1_loss_val * 100

        if rank == 0:
            print(f'[RANK 0] Validation - G_loss: {G_loss_val:.4f}, '
                  f'L1_loss: {l1_loss_val:.4f}, Combined_loss: {combined_loss_val:.4f}')

        if (epoch + 1) == start_epoch + EXTRA_EPOCHS and rank == 0:
            model.eval()
            torch.save({
                "network": model.module.state_dict(),
                "discriminator": discriminator.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "D_optimizer": D_optimizer.state_dict(),
            }, output_dir + f'/checkpoints/Epoch_{epoch+1}.tar')

    if rank == 0:
        print('[RANK 0] Evaluating on test set...')
        G_loss_test, l1_loss_test = do_evaluation(test_loader, model.module, device, discriminator.module)
        test_loss = G_loss_test + l1_loss_test * 100
        print(f'Test loss: {test_loss:.4f}, G_loss: {G_loss_test:.4f}, L1_loss: {l1_loss_test:.4f}')

    cleanup()

def run_ddp():
    n_gpus = torch.cuda.device_count()
    if n_gpus < 1:
        raise RuntimeError("At least one GPU is required.")
    elif n_gpus == 1:
        print("Warning: Only one GPU detected. DDP will work but without real parallelism.")
    mp.spawn(ddp_main, args=(n_gpus,), nprocs=n_gpus, join=True)

if __name__ == '__main__':
    set_random_seed(1, False)
    run_ddp()
