import torch
import sys
import pandas as pd
import logging
import os
from logging import Logger
from torch import nn, optim
from argparse import ArgumentParser
sys.path.append('../..')
import definitions as D
from src.utils import seed_everything, load_json
from dataclasses import dataclass, field
from src.datasets import NumpyDirectory
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
from src.transforms import NoiseScheduler
from src.models import UNet
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

@dataclass
class Config:

    ### *** Reproducibility *** ###
    seed : int = 42

    ### *** Device *** ###
    device : str = "cuda" if torch.cuda.is_available() else "cpu"

    ### *** Architecture *** ###
    in_channels : int = 3
    out_channels : int = 32
    down_channels : list[int] = field(default_factory=lambda:[64,96,128,192])
    mid_channels : list[int] = field(default_factory=lambda:[192,128])
    num_layers : int = 2
    norm_channels : int = 32
    num_heads : int = 8
    t_emb_dim : int = 512
    output_activation : str = 'linear'

    ### *** Training *** ###
    batch_size : int = 16
    learning_rate : float = 1e-4
    total_epochs : int = 100
    timesteps : int = 1000
    beta_start : float = 0.0015
    beta_end : float = 0.0195

    ### *** Data Loading *** ###
    num_workers : int = 4
    prefetch_factor : int = 2

    ### *** Data Directory *** ###
    data_dir : str = ''

@dataclass
class Args:
    experiment : str

def create_datasets(config: Config) -> tuple[NumpyDirectory,NumpyDirectory,NumpyDirectory]:

    data_dir = os.path.join(D.EXPERIMENTS_DIR,config.data_dir,'features')

    train_data = NumpyDirectory(
        root = data_dir,
        transform = T.ToDtype(dtype=torch.float32,scale=False),
        split = 'train',
    )

    val_data = NumpyDirectory(
        root = data_dir,
        transform = T.ToDtype(dtype=torch.float32,scale=False),
        split = 'val',
    )

    test_data = NumpyDirectory(
        root = data_dir,
        transform = T.ToDtype(dtype=torch.float32,scale=False),
        split = 'test',
    )

    return train_data,val_data,test_data

def create_dataloaders(config : Config) -> tuple[DataLoader,DataLoader,DataLoader]:

    train_data,val_data,test_data = create_datasets(config)

    train_loader = DataLoader(
        train_data,
        batch_size = config.batch_size,
        shuffle = True,
        num_workers = config.num_workers,
        prefetch_factor = config.prefetch_factor,
    )

    val_loader = DataLoader(
        val_data,
        batch_size = config.batch_size,
        shuffle = False,
        num_workers = config.num_workers,
        prefetch_factor = config.prefetch_factor,
    )

    test_loader = DataLoader(
        test_data,
        batch_size = config.batch_size,
        shuffle = False,
        num_workers = config.num_workers,
        prefetch_factor = config.prefetch_factor,
    )

    return train_loader,val_loader,test_loader

def create_scheduler(config: Config) -> NoiseScheduler:

    return NoiseScheduler(
        timesteps = config.timesteps,
        beta_start = config.beta_start,
        beta_end = config.beta_end,
    ).to(config.device)

def create_model(config: Config) -> UNet:

    return UNet(
        in_channels = config.in_channels,
        out_channels = config.out_channels,
        down_channels = config.down_channels,
        mid_channels = config.mid_channels,
        num_layers = config.num_layers,
        norm_channels = config.norm_channels,
        num_heads = config.num_heads,
        t_emb_dim = config.t_emb_dim,
        output_activation = config.output_activation,
    ).to(config.device)


def train(
    model : UNet,
    train_loader : DataLoader,
    val_loader : DataLoader,
    scheduler : NoiseScheduler,
    loss_fn : nn.Module,
    optimizer : optim.Optimizer,
    epochs : int,
    device : str
) -> pd.DataFrame:
    
    writer = SummaryWriter()
    
    history = {
        'epoch' : [],
        'split' : [],
        'loss' : [],
    }

    iteration = 0.0

    model = model.to(device)
    scheduler = scheduler.to(device)
    
    for epoch in range(epochs):

        for phase in ['train','val']:

            training = phase == 'train'

            loader = train_loader if training else val_loader
            iterator = tqdm(loader,desc=f'Epoch {epoch+1}/{epochs} [{phase}]')

            ### *** Set model mode *** ###
            model.train(training)

            ### *** Update history *** ###
            history['epoch'].append(epoch)
            history['split'].append(phase)
            history['loss'].append(0.0)

            for data in iterator:
                
                ### *** Zero the gradients *** ###
                if training:
                    optimizer.zero_grad()

                ### *** Move data to device *** ###
                data['latent'] = data['latent'].to(device)

                ### *** Noising step *** ###
                noised,noise,t = scheduler(data['latent'])

                with torch.set_grad_enabled(training):
                    
                    ### *** Forward pass *** ###
                    output = model(noised,t)

                    ### *** Calculate loss *** ###
                    loss = loss_fn(output,noise)

                    if training:

                        ### *** Backward pass *** ###
                        loss.backward()

                        ### *** Update weights *** ###
                        optimizer.step()

                        ### *** Update iteration *** ###
                        iteration += 1

                        ### *** Log to Tensorboard *** ###
                        writer.add_scalar(f'loss',loss.item(),iteration)

                ### *** Update history *** ###
                history['loss'][-1] += loss.item() / len(loader)

            msg = f'Epoch {epoch+1}/{epochs} [{phase}] Loss: {history["loss"][-1]:.4f}'

            print(f"\n{msg}\n")

    return pd.DataFrame(history)

def main(args: Args) -> None:

    ### *** Logger *** ###
    logger = Logger(name="Train VQVAE",level=logging.INFO)

    ### ***** Load Config ***** ###
    logger.info(f'Loading Config for Experiment {args.experiment}')
    experiment_dir = os.path.join(D.EXPERIMENTS_DIR,args.experiment)

    if not os.path.exists(experiment_dir):
        raise FileNotFoundError(f'Experiment {args.experiment} not found')
    
    config_path = os.path.join(experiment_dir,'config.json')
    config = Config(**load_json(config_path))
    checkpoints_dir = os.path.join(experiment_dir,'checkpoints')
    os.makedirs(checkpoints_dir,exist_ok=True)        

    ### ***** Seed Everything ***** ###
    seed_everything(config.seed)

    ### ***** Load Dataset ***** ###
    logger.info('Loading Dataset')
    train_dataloader,val_dataloader,_ = create_dataloaders(config)

    ### ***** Initialize Models ***** ###
    logger.info('Initializing Models')
    model = create_model(config)
    scheduler = create_scheduler(config)

    ### ***** Initialize Optimizers ***** ###
    logger.info('Initializing Optimizers')
    optimizer = optim.Adam(model.parameters(),lr=config.learning_rate)

    ### ***** Initialize Criterias ***** ###
    logger.info('Initializing Criterias')
    loss_fn = nn.MSELoss()

    ### ***** Train ***** ###
    logger.info('Starting Training')

    history = train(
        model = model,
        train_loader = train_dataloader,
        val_loader = val_dataloader,
        scheduler = scheduler,
        loss_fn = loss_fn,
        optimizer = optimizer,
        epochs = config.total_epochs,
        device = config.device
    )

    logger.info('Training Complete')

    ### ***** Save History ***** ###
    logger.info('Saving History')
    history_path = os.path.join(experiment_dir,'history.csv')
    history.to_csv(history_path,index=False)

    ### ***** Save Checkpoint ***** ###
    logger.info('Saving Checkpoint')
    checkpoint_path = os.path.join(checkpoints_dir,'checkpoint.pth')  
    torch.save(model.state_dict(),checkpoint_path)

    logger.info('Goodbye!')

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--experiment',type=str,required=True)

    args = parser.parse_args()
    args = Args(**vars(args))

    main(args)