import sys
import os
import torch
import numpy as np
sys.path.append('../..')
import definitions as D
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Optional
from src.models import VQVAE
from torch.utils.data import DataLoader
from src.training.vqvae import create_loaders,create_models,Config
from src.utils import load_json,get_last_checkpoint,seed_everything
from tqdm.auto import tqdm

@dataclass
class Args:
    experiment: str
    batch_size: int = 64
    num_workers: int = 4
    prefetch_factor: Optional[int] = 2
    save_dir: Optional[str] = None

def feature_extract(
    model : VQVAE,
    loaders : dict[str,DataLoader],
    device : torch.device,
    save_dir : str
) -> None:
    
    model.to(device)
    model.eval()

    with torch.inference_mode():

        for split,loader in loaders.items():

            os.makedirs(os.path.join(save_dir,split),exist_ok=True)

            for data in tqdm(loader):

                # Move data to device
                data['image'] = data['image'].to(device)

                # Forward pass
                enc_out = model.encoder(data['image'])
                quant_out = model.quantize(enc_out)
                latents = quant_out['quant_out'].detach().cpu().numpy()

                for idx,latent in zip(data['id'],latents):
                    idx = idx.long().item()
                    image_id = loader.dataset.splits_df.iloc[idx]['image_id']
                    image_id = os.path.splitext(image_id)[0]
                    np.save(os.path.join(save_dir,split,f'{image_id}.npy'),latent)

def main(args: Args):
    
    ### *** Load configuration *** ###
    EXPERIMENTS_DIR = D.EXPERIMENTS_DIR
    EXPERIMENT_DIR = os.path.join(EXPERIMENTS_DIR, args.experiment)
    CHECKPOINTS_DIR = os.path.join(EXPERIMENT_DIR, 'checkpoints')
    CONFIG_PATH = os.path.join(EXPERIMENT_DIR, 'config.json')
    config = Config(**load_json(CONFIG_PATH))

    ### *** Update configuration *** ###
    config.batch_size = args.batch_size
    config.num_workers = args.num_workers
    config.prefetch_factor = args.prefetch_factor

    ### *** Seed everything *** ###
    seed_everything(config.seed)

    ### *** create loaders *** ###
    loaders = create_loaders(config)

    ### *** create models *** ###
    vqvae,_,_ = create_models(config)

    ### *** Load last checkpoint *** ###
    last_checkpoint = get_last_checkpoint(CHECKPOINTS_DIR)

    if last_checkpoint is None:
       raise ValueError('No checkpoint found')
    
    vqvae.load_state_dict(last_checkpoint['vqvae'])

    ### *** Feature extraction *** ###
    save_dir = args.save_dir or os.path.join(D.DATASETS[config.dataset].root, 'features')
    os.makedirs(save_dir,exist_ok=True)

    feature_extract(
        model=vqvae,
        loaders={
            'train': loaders[0],
            'val': loaders[1],
            'test': loaders[2]
        },
        device=config.device,
        save_dir=save_dir
    )

    print('Goodbye!')

if __name__ == '__main__':
    
    parser = ArgumentParser()

    parser.add_argument('--experiment', type=str, help='Name of the experiment')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--num-workers', type=int, help='Number of workers')
    parser.add_argument('--prefetch-factor', type=int, help='Prefetch factor')
    parser.add_argument('--save-dir', type=str, help='Directory to save features')

    args = parser.parse_args()
    args = Args(**vars(args))

    main(args)