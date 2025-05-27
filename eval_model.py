import torch
import torch.optim as optim
import numpy as np
import os
import gc
import sys
import yaml

def load_config():
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config
config = load_config()

sys.path.append(config['path']['model_dir'])

from networks.generator import Generator
from networks.discriminator import Discriminator
from utils.data import get_data_loader
from utils.helpers import *
from utils.metrics import *

output_dir = config['path']['output_dir']
label_dir = config['path']['label_dir']
BP_dir = config['path']['BP_dir']

os.makedirs(os.path.join(output_dir, '3d_models'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'numpy_models'), exist_ok=True)


def save_test_results(pred, gt, filename):
    np.save(os.path.join(output_dir, 'numpy_models', f'pred_{filename}.npy'), pred)
    np.save(os.path.join(output_dir, 'numpy_models', f'gt_{filename}.npy'), gt)
    save_3d_model(pred, os.path.join(output_dir, '3d_models', f'pred_{filename}'))
    save_3d_model(gt, os.path.join(output_dir, '3d_models', f'gt_{filename}'))


def main():
    print("Starting main...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = Generator(in_channels=1, num_filters=64, class_num=1).to(device)
    discriminator = Discriminator(device, 2).to(device)
    
    gc.collect()
    torch.cuda.empty_cache()

    _ , _ , test_loader = get_data_loader(BP_dir, label_dir, 1)
    print("Dataset setup complete.")

    LEARNING_RATE = config['hyperparameters']['learning_rate']
    BETAS = config['hyperparameters']['betas']
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=BETAS)
    D_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=BETAS)

    print("Loading checkpoint...")
    checkpoint_path = config['path']['checkpoint_dir'] + '/Epoch_198.tar'
    _ = load_checkpoint(model, discriminator, optimizer, D_optimizer, checkpoint_path)

    model.eval()

    ot_1mm_list = []
    ot_2mm_list = []
    chamfer_distance_list = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data[0].float().to(device), data[1].float().to(device)
            outputs = model(inputs)

            outputs = outputs.cpu().numpy()
            labels = labels.cpu().numpy()

            print(f'Processing {i} / {len(test_loader)} ...')
            save_test_results(outputs, labels, f'{i}')

            # Compute Ot(d) for d=1 & d=2
            ot_1mm = compute_overlap_metric(labels, outputs, d=1)
            ot_2mm = compute_overlap_metric(labels, outputs, d=2)

            # Compute Chamfer Distance
            chamfer_distance = compute_chamfer_distance(labels, outputs)

            ot_1mm_list.append(ot_1mm)
            ot_2mm_list.append(ot_2mm)
            chamfer_distance_list.append(chamfer_distance)

            print(f"Sample {i}: Ot(1mm)={ot_1mm*100:.2f}, Ot(2mm)={ot_2mm*100:.2f}, CD(L2)={chamfer_distance:.2f}")
    
    print(f"Ot(1mm): {np.mean(ot_1mm_list)*100:.2f}, Ot(2mm): {np.mean(ot_2mm_list)*100:.2f}, CD(L2): {np.mean(chamfer_distance_list):.2f}")

if __name__ == "__main__":
    main()
    print("Done")