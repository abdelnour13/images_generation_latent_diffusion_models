import torch
import sys
import pandas as pd
import logging
import os
import numpy as np
from logging import Logger
from torch import nn, optim
from argparse import ArgumentParser
import imageio
sys.path.append('../..')
import definitions as D
from src.utils import seed_everything, load_json, make_grid, get_last_checkpoint, move_data_to_device
from dataclasses import dataclass, field
from src.datasets import ImageDirectory
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
from src.models import LatentDiffusion,LatentDiffusionConfig
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter   

@dataclass
class Config:

    ### *** Reproducibility *** ###
    seed : int = 42

    ### *** Device *** ###
    device : str = "cuda" if torch.cuda.is_available() else "cpu"

    ### *** Architecture *** ###
    model_config : LatentDiffusionConfig = field(default_factory=LatentDiffusionConfig)
    vqvae_exp : str = 'vqvae'

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
    dataset : str = next(iter(D.DATASETS.keys()))

    ### *** GIF Generation *** ###
    grid_size : int = 4
    save_image_every : int = 1000
    noise_size : tuple[int,int] = (16,16)

    def __post_init__(self):

        if isinstance(self.model_config,dict):
            self.model_config = LatentDiffusionConfig(**self.model_config)

@dataclass
class Args:
    experiment : str

def create_transforms(config: Config) -> T.Compose:

    if config.model_config.input_type == 'image':

        return T.Compose([
            T.ToImage(),
            T.ToDtype(dtype=torch.float32,scale=True),
            T.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]),
        ])
    
    else:
        
        return T.Compose([
            T.ToDtype(dtype=torch.float32,scale=False),
        ])

def create_dataset(config: Config,split : str) -> ImageDirectory:

    dataset = ImageDirectory(
        dataset = config.dataset,
        transform = create_transforms(config),
        split = split,
        type=config.model_config.input_type,
        return_metadata=(config.model_config.metadata_cond is not None),
    )

    return dataset

def create_datasets(config: Config) -> tuple[ImageDirectory,ImageDirectory,ImageDirectory]:

    train_data = create_dataset(config,'train')
    val_data = create_dataset(config,'val')
    test_data = create_dataset(config,'test')

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


def create_model(config: Config) -> LatentDiffusion:

    model =  LatentDiffusion(config.model_config).to(config.device)

    checkpoints_dir = os.path.join(D.EXPERIMENTS_DIR,config.vqvae_exp,'checkpoints')
    checkpoint = get_last_checkpoint(checkpoints_dir)['vqvae']
    model.vqvae.load_state_dict(checkpoint,strict=False)
    
    return model


def train(
    config : Config,
    *,
    model : LatentDiffusion,
    train_loader : DataLoader,
    val_loader : DataLoader,
    loss_fn : nn.Module,
    optimizer : optim.Optimizer,
) -> tuple[pd.DataFrame,list[np.ndarray]]:
    
    writer = SummaryWriter()
    
    history = {
        'epoch' : [],
        'split' : [],
        'loss' : [],
    }

    iteration = 0

    gif_noise = torch.randn(
        config.grid_size**2,
        config.model_config.vqvae.z_dim,
        *config.noise_size,
        device=config.device
    )

    images = []

    model = model.to(config.device)
    
    for epoch in range(config.total_epochs):

        for phase in ['train','val']:

            training = phase == 'train'

            loader = train_loader if training else val_loader
            iterator = tqdm(loader,desc=f'Epoch {epoch+1}/{config.total_epochs} [{phase}]')

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
                data = move_data_to_device(data,config.device)

                with torch.set_grad_enabled(training):
                    
                    ### *** Forward pass *** ###
                    y_hat,y,t = model.forward(data['data'],data.get('metadata',None))

                    ### *** Calculate loss *** ###
                    loss = loss_fn(y_hat,y)

                    if training:

                        ### *** Backward pass *** ###
                        loss.backward()

                        ### *** Update weights *** ###
                        optimizer.step()

                        ### *** Log to Tensorboard *** ###
                        writer.add_scalar('Loss',loss.item(),iteration)
                        
                        if iteration % config.save_image_every == 0:

                            model.eval()

                            fake_images,_ = model.generate(gif_noise,progress=False)
                            fake_images = fake_images.detach().cpu()

                            model.train()
                            
                            writer.add_images('Generated Images',fake_images,iteration // config.save_image_every)

                            fake_images = make_grid(fake_images,h=config.grid_size,w=config.grid_size,gap=4,gap_value=1.0) \
                                .mul(255) \
                                .type(torch.uint8) \
                                .numpy()
                            
                            images.append(fake_images)

                        ### *** Update iteration *** ###
                        iteration += 1

                    ### *** Update history *** ###
                    history['loss'][-1] += loss.item() / len(loader)

            msg = f'Epoch {epoch+1}/{config.total_epochs} [{phase}] Loss: {history["loss"][-1]:.4f}'

            print(f"\n{msg}\n")

    return pd.DataFrame(history),images

def main(args: Args) -> None:

    ### *** Logger *** ###
    logger = Logger(name="Main",level=logging.INFO)

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

    ### ***** Initialize Optimizers ***** ###
    logger.info('Initializing Optimizers')
    optimizer = optim.Adam(model.parameters(),lr=config.learning_rate)

    ### ***** Initialize Criterias ***** ###
    logger.info('Initializing Criterias')
    loss_fn = nn.MSELoss()

    ### ***** Train ***** ###
    logger.info('Starting Training')

    history,images = train(
        config = config,
        model = model,
        train_loader = train_dataloader,
        val_loader = val_dataloader,
        loss_fn = loss_fn,
        optimizer = optimizer,
    )

    logger.info('Training Complete')

    ### ***** Save GIF ***** ###
    logger.info('Saving GIF')
    gif_path = os.path.join(experiment_dir,'images.gif')
    imageio.mimsave(gif_path,images)

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