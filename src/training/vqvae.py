import sys
sys.path.append('../..')
import os
import torch
import pandas as pd
import definitions as D
import warnings
import logging
import numpy as np
import imageio
from torch import nn,optim
from torch.utils.data import DataLoader
from src.datasets import ImageDirectory
from src.models import VQVAE, Discriminator
from src.utils import seed_everything,load_json
from torchvision import transforms as T
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Optional
from tqdm.auto import tqdm
from lpips import LPIPS
from logging import Logger
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings('ignore',category=UserWarning)

@dataclass
class Config:

    ### *** Reproducibility *** ###
    seed : int = 42

    ### *** Data *** ###
    device : str = 'cuda'

    ### *** Models *** ###
    vqvae_args : Optional[dict] = None
    discriminator_args : Optional[dict] = None
    lpips_net : str = 'squeeze' # 'squeeze','vgg' or 'alex'

    ### *** Training *** ###
    batch_size : int = 32
    learning_rate : float = 1e-4
    total_epochs : int = 20
    disc_start_iter : int = 8000
    percptual_loss_weight : float = 1.0
    commitment_loss_weight : float = 1.0
    codebook_loss_weight : float = 0.2
    adversarial_loss_weight : float = 0.5

    discriminator_loss : str = 'bce' # 'bce' or 'mse'

    ### *** Preprocessing *** ###
    image_size : Optional[tuple[int,int]] = None

    ### *** Data Loading *** ###
    num_workers : int = 4
    prefetch_factor : int = 2

    ### *** GIF Generation *** ###
    save_image_every : int = 1000
    grid_size : int = 4
    max_frames : int = 50

    ### *** Dataset *** ###
    dataset : str = next(iter(D.DATASETS.keys()))

@dataclass
class Args:
    experiment : str

def create_transforms(config : Config) -> tuple[T.Compose,T.Compose,T.Compose]:

    train_transforms = T.Compose([
        T.Resize(config.image_size) if config.image_size is not None else T.Lambda(lambda x: x),
        T.ToTensor(),
        T.Lambda(lambda x : 2 * x - 1)
    ])

    val_transforms = T.Compose([
        T.Resize(config.image_size) if config.image_size is not None else T.Lambda(lambda x: x),
        T.ToTensor(),
        T.Lambda(lambda x : 2 * x - 1)
    ])

    test_transforms = T.Compose([
        T.Resize(config.image_size) if config.image_size is not None else T.Lambda(lambda x: x),
        T.ToTensor(),
        T.Lambda(lambda x : 2 * x - 1)
    ])

    return train_transforms,val_transforms,test_transforms

def create_datasets(config : Config) -> tuple[ImageDirectory,ImageDirectory,ImageDirectory]:

    train_transforms,val_transforms,test_transforms = create_transforms(config)

    train_dataset = ImageDirectory(
        dataset = config.dataset,
        split = 'train',
        transform = train_transforms
    )

    val_dataset = ImageDirectory(
        dataset = config.dataset,
        split = 'val',
        transform = val_transforms
    )

    test_dataset = ImageDirectory(
        dataset = config.dataset,
        split = 'test',
        transform = test_transforms
    )

    return train_dataset,val_dataset,test_dataset

def create_loaders(config : Config) -> tuple[DataLoader,DataLoader,DataLoader]:

    train_dataset,val_dataset,test_dataset = create_datasets(config)

    train_loader = DataLoader(
        train_dataset,
        batch_size = config.batch_size,
        shuffle = True,
        num_workers = config.num_workers,
        prefetch_factor = config.prefetch_factor,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size = config.batch_size,
        shuffle = False,
        num_workers = config.num_workers,
        prefetch_factor = config.prefetch_factor,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size = config.batch_size,
        shuffle = False,
        num_workers = config.num_workers,
        prefetch_factor = config.prefetch_factor,
    )

    return train_loader,val_loader,test_loader

def create_models(config : Config) -> tuple[nn.Module,nn.Module,nn.Module]:

    vqvae = VQVAE(**config.vqvae_args)
    discriminator = Discriminator(**config.discriminator_args)
    lpip_net = LPIPS(net = config.lpips_net)

    vqvae = vqvae.to(config.device)
    discriminator = discriminator.to(config.device)
    lpip_net = lpip_net.to(config.device).eval()

    return vqvae,discriminator,lpip_net

def create_optimizers(config : Config,vqvae : nn.Module,discriminator : nn.Module) -> tuple[optim.Optimizer,optim.Optimizer]:

    vqvae_opt = optim.Adam(vqvae.parameters(),lr=config.learning_rate,betas=(0.5,0.999))
    disc_opt = optim.Adam(discriminator.parameters(),lr=config.learning_rate,betas=(0.5,0.999))

    return vqvae_opt,disc_opt

def create_criterias(config : Config) -> tuple[nn.Module,nn.Module]:

    d_loss = nn.BCEWithLogitsLoss() if config.discriminator_loss == 'bce' else nn.MSELoss()
    rec_loss = nn.MSELoss()

    return d_loss,rec_loss

def train(
    config : Config,
    *,
    discriminator : nn.Module,
    vqvae : nn.Module,
    lpip_net : nn.Module,
    d_loss : nn.Module,
    rec_loss : nn.Module,
    vqvae_opt : optim.Optimizer,
    d_opt : optim.Optimizer,
    train_loader : DataLoader,
    val_loader : DataLoader,
) -> tuple[pd.DataFrame,list[np.ndarray]]:

    lpip_net.eval()

    iteration_count = 0

    losses = [
        'd_real_loss',
        'd_fake_loss',
        'd_loss',
        'vqvae_loss',
        'rec_loss',
        'commitment_loss',
        'codebook_loss',
        'adversarial_loss',
        'lpips_loss',
    ]

    history = {
        'epoch': [],
        'split': [],
        **{loss: [] for loss in losses}
    }

    images = []

    writer = SummaryWriter()

    example_images = []

    for i in range(config.grid_size ** 2):
        example_images.append(val_loader.dataset[i]['image'].unsqueeze(0))

    example_images = torch.cat(example_images,dim=0).to(config.device)
                
    for epoch in range(config.total_epochs):

        for phase in ['train','val']:
            
            ### *** Setup *** ###
            loader = train_loader if phase == 'train' else val_loader
            training = phase == 'train'

            discriminator.train(training)
            vqvae.train(training)

            ### *** History *** ###

            history['epoch'].append(epoch)
            history['split'].append(phase)
            
            for loss in losses:
                history[loss].append(0.0)

            iterator = tqdm(enumerate(loader),total=len(loader),desc=f'Epoch {epoch+1}/{config.total_epochs} - {phase}')

            with torch.set_grad_enabled(training):

                for i,data in iterator:

                    # Move data to device
                    data['image'] = data['image'].to(config.device)
                        
                    ############################
                    # Discriminator Training
                    ############################

                    d_real_loss = torch.tensor(0.0).to(config.device)
                    d_fake_loss = torch.tensor(0.0).to(config.device)

                    ### *** Phase 01 : train with real data *** ###

                    if iteration_count >= config.disc_start_iter:

                        # Zero the gradients
                        if training:
                            discriminator.zero_grad()

                        # Forward pass
                        output = discriminator(data['image'])

                        # Calculate loss
                        label = torch.ones_like(output)
                        d_real_loss = d_loss(output,label)

                        # Backward pass
                        if training:
                            d_real_loss.backward()

                    ### *** Phase 02 : train with fake data *** ###

                    # Generate fake data
                    vqvae_out = vqvae(data['image'])

                    if iteration_count >= config.disc_start_iter:
                        # Forward pass
                        output = discriminator(vqvae_out['dec_out'].detach())

                        # Calculate loss
                        label = torch.zeros_like(output)
                        d_fake_loss = d_loss(output,label)

                        # Backward pass
                        if training:
                            d_fake_loss.backward()

                        # Update weights
                        if training:
                            d_opt.step()

                    ############################
                    # VQVAE Training
                    ############################

                    ### *** Adversarial Loss *** ###

                    # Zero the gradients
                    if training:
                        vqvae.zero_grad()

                    adv_loss = torch.tensor(0.0).to(config.device)

                    if iteration_count >= config.disc_start_iter:

                        # Forward pass
                        output = discriminator(vqvae_out['dec_out']) if training else output

                        # Calculate adversarial loss
                        label = torch.ones_like(output)
                        adv_loss = d_loss(output,label)

                    # Calculate the LPIPS loss
                    lpips_loss = lpip_net(vqvae_out['dec_out'],data['image']).mean()

                    # Calculate reconstruction loss
                    reconstruction_loss = rec_loss(vqvae_out['dec_out'],data['image'])

                    # Calculate the loss
                    vqvae_loss = (
                        reconstruction_loss +
                        config.commitment_loss_weight * vqvae_out['commitment_loss'] +
                        config.codebook_loss_weight * vqvae_out['codebook_loss'] +
                        config.adversarial_loss_weight * adv_loss +
                        config.percptual_loss_weight * lpips_loss
                    )

                    # Backward pass
                    if training:
                        vqvae_loss.backward()

                    # Update weights
                    if training:
                        vqvae_opt.step()

                    ############################
                    # Update History
                    ############################

                    disc_iterations = 1.0

                    if training:

                        if iteration_count >= config.disc_start_iter:
                            if epoch == config.disc_start_iter // len(loader):
                                disc_iterations = len(loader) - config.disc_start_iter % len(loader)
                            else:
                                disc_iterations = len(loader)

                    elif iteration_count >= config.disc_start_iter:
                        disc_iterations = len(loader)


                    history['d_real_loss'][-1] += d_real_loss.item() / disc_iterations
                    history['d_fake_loss'][-1] += d_fake_loss.item() / disc_iterations
                    history['d_loss'][-1] += (d_real_loss + d_fake_loss).item() / (2 * disc_iterations)
                    history['vqvae_loss'][-1] += vqvae_loss.item() / len(loader)
                    history['rec_loss'][-1] += reconstruction_loss.item() / len(loader)
                    history['commitment_loss'][-1] += vqvae_out['commitment_loss'].item() / len(loader)
                    history['codebook_loss'][-1] += vqvae_out['codebook_loss'].item() / len(loader)
                    history['adversarial_loss'][-1] += adv_loss.item() / len(loader)
                    history['lpips_loss'][-1] += lpips_loss.item() / len(loader)

                    ############################
                    # Update Tensorboard
                    ############################

                    if training:
                        
                        if iteration_count >= config.disc_start_iter:

                            writer.add_scalar('d_real_loss',d_real_loss.item(),iteration_count)
                            writer.add_scalar('d_fake_loss',d_fake_loss.item(),iteration_count)
                            writer.add_scalar('d_loss',(d_real_loss + d_fake_loss).item() / 2,iteration_count)
                            writer.add_scalar('adversarial_loss',adv_loss,iteration_count)

                        writer.add_scalar('vqvae_loss',vqvae_loss.item(),iteration_count)
                        writer.add_scalar('rec_loss',reconstruction_loss.item(),iteration_count)
                        writer.add_scalar('commitment_loss',vqvae_out['commitment_loss'].item(),iteration_count)
                        writer.add_scalar('codebook_loss',vqvae_out['codebook_loss'].item(),iteration_count)
                        writer.add_scalar('lpips_loss',lpips_loss.item(),iteration_count)

                    ############################
                    # Update Images
                    ############################

                    if training and (iteration_count % config.save_image_every == 0):   

                        vqvae.eval()

                        with torch.inference_mode():
                            vqvae_out = vqvae(example_images)
                            decoded_images = vqvae_out['dec_out'].detach().cpu()

                        decoded_images = (decoded_images + 1) / 2
                        decoded_images = decoded_images * 255
                        decoded_images = decoded_images.type(torch.uint8)

                        writer.add_images('decoded_images',decoded_images,iteration_count)

                        decoded_images = F.pad(decoded_images,(4,4,4,4),value=255)
                        decoded_images = decoded_images.permute(0,2,3,1)

                        decoded_images = decoded_images.reshape(config.grid_size,config.grid_size,*decoded_images.shape[1:])
                        decoded_images = decoded_images.permute(0,2,1,3,4)
                        decoded_images = decoded_images.flatten(0,1).flatten(1,2).numpy()
                        
                        images.append(decoded_images)

                        vqvae.train()
                        
                    # Update the iteration count
                    if training:
                        iteration_count += 1

            ### *** Logging *** ###

            msg = f'Epoch {epoch} - {phase} : '

            for loss in losses:
                msg += f'{loss} : {history[loss][-1]:.4f} |'

            print(f"\n{msg}\n")

    writer.close()

    return pd.DataFrame(history),images


def main(args : Args) -> None:
    
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
    train_dataloader,val_dataloader,test_dataloader = create_loaders(config)

    ### ***** Initialize Models ***** ###
    logger.info('Initializing Models')
    vqvae,discriminator,lpips = create_models(config)

    ### ***** Initialize Optimizers ***** ###
    logger.info('Initializing Optimizers')
    vqvae_optim,disc_optim = create_optimizers(config,vqvae,discriminator)

    ### ***** Initialize Criterias ***** ###
    logger.info('Initializing Criterias')
    d_loss,rec_loss = create_criterias(config)

    ### ***** Train ***** ###
    logger.info('Starting Training')

    history,images = train(
        config,
        discriminator = discriminator,
        vqvae = vqvae,
        lpip_net = lpips,
        d_loss = d_loss,
        rec_loss = rec_loss,
        vqvae_opt = vqvae_optim,
        d_opt = disc_optim,
        train_loader = train_dataloader,
        val_loader = val_dataloader
    )

    logger.info('Training Complete')

    ### ***** Save History ***** ###
    logger.info('Saving History')
    history_path = os.path.join(experiment_dir,'history.csv')
    history.to_csv(history_path,index=False)

    ### ***** Save GIF ***** ###
    logger.info('Saving GIF')
    gif_path = os.path.join(experiment_dir,'images.gif')
    imageio.mimsave(gif_path,images)

    ### ***** Save Checkpoint ***** ###
    logger.info('Saving Checkpoint')
    checkpoint_path = os.path.join(checkpoints_dir,'checkpoint.pth')  
    
    torch.save({
        'vqvae' : vqvae.state_dict(),
        'discriminator' : discriminator.state_dict()
    },checkpoint_path)

    logger.info('Goodbye!')

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--experiment', type=str, help='Name of the experiment',choices=os.listdir(D.EXPERIMENTS_DIR))

    args = parser.parse_args()

    main(Args(**vars(args)))