import torch
import sys
import pandas as pd
import logging
import os
import numpy as np
import imageio
from logging import Logger
from torch import nn, optim
from argparse import ArgumentParser
sys.path.append('../..')
import definitions as D
from src.utils import seed_everything, load_json, make_grid
from dataclasses import dataclass, field
from src.datasets import ImageDirectory
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
from src.models import Generator, Discriminator, GeneratorConfig, DiscriminatorConfig
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

@dataclass
class Config:

    ### *** Reproducibility *** ###
    seed : int = 42

    ### *** Device *** ###
    device : str = "cuda" if torch.cuda.is_available() else "cpu"

    ### *** Architecture *** ###
    gan_config : GeneratorConfig = field(default_factory=GeneratorConfig)
    discriminator_config : DiscriminatorConfig = field(default_factory=DiscriminatorConfig)

    ### *** Training *** ###
    batch_size : int = 16
    learning_rate : float = 1e-4
    total_epochs : int = 100
    noise_size : tuple[int] = field(default_factory=lambda: (16,16))
    loss_fn : str = 'mse'
    smooth : float = 0.1

    ### *** Data Loading *** ###
    num_workers : int = 4
    prefetch_factor : int = 2

    ### *** Data Directory *** ###
    dataset : str = next(iter(D.DATASETS.keys()))

    ### *** GIF generation *** ###
    save_image_every : int = 1000
    grid_size : int = 4

    def __post_init__(self):
            
        if isinstance(self.gan_config, dict):
            self.gan_config = GeneratorConfig(**self.gan_config)

        if isinstance(self.discriminator_config, dict):
            self.discriminator_config = DiscriminatorConfig(**self.discriminator_config)

@dataclass
class Args:
    experiment : str

def create_dataset(config : Config) -> ImageDirectory:

    transfroms = [
        T.ToTensor()
    ]

    if config.gan_config.output_activation == 'tanh':
        transfroms.append(T.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]))

    transfroms = T.Compose(transfroms)

    train_dataset = ImageDirectory(
        dataset = config.dataset,
        transform = transfroms
    )

    return train_dataset

def create_dataloader(config : Config) -> DataLoader:

    train_dataset = create_dataset(config)

    train_loader = DataLoader(
        train_dataset,
        batch_size = config.batch_size,
        num_workers = config.num_workers,
        prefetch_factor = config.prefetch_factor,
        shuffle = True
    )

    return train_loader

def create_models(config : Config) -> tuple[Generator, Discriminator]:

    generator = Generator(config.gan_config)
    discriminator = Discriminator(config.discriminator_config)

    return generator, discriminator

def create_optimizers(config : Config,generator : nn.Module,discriminator : nn.Module) -> tuple[optim.Optimizer,optim.Optimizer]:

    gen_opt = optim.Adam(generator.parameters(),lr=config.learning_rate,betas=(0.5,0.999))
    disc_opt = optim.Adam(discriminator.parameters(),lr=config.learning_rate,betas=(0.5,0.999))

    return gen_opt,disc_opt

def create_loss_fn(config : Config) -> nn.Module:
    
    if config.loss_fn == 'mse':
        return nn.MSELoss()
    elif config.loss_fn == 'bce':
        return nn.BCEWithLogitsLoss()
    
    raise ValueError(f"Loss function {config.loss_fn} not supported")

def train(
    config : Config,
    *,
    generator : Generator,
    discriminator : Discriminator,
    loss_fn : nn.Module,
    g_optim : optim.Optimizer,
    d_optim : optim.Optimizer,
    dataloader : DataLoader,
) -> tuple[pd.DataFrame,list[np.ndarray]]:
    
    ### *** Put models in training mode & the right device *** ###
    generator.train().to(config.device)
    discriminator.train().to(config.device)

    ### *** History *** ###
    history = {
        "epoch" : [],
        "g_loss" : [],
        "d_loss" : [],
        "d_real" : [],
        "d_fake" : []
    }

    ### *** Tensorborad *** ###
    writer = SummaryWriter()

    ### *** GIF *** ###
    total_iterations = 0
    gif_noise = torch.randn(config.grid_size**2,config.gan_config.latent_dim,1,1).to(config.device)
    images = []

    ### *** Training *** ###
    for epoch in range(config.total_epochs):

        ### *** Progress bar *** ###
        iterator = tqdm(enumerate(dataloader),desc=f"Epoch {epoch+1}/{config.total_epochs}",total=len(dataloader))

        ### *** History *** ###
        history['epoch'].append(epoch)
        history['g_loss'].append(0.0)
        history['d_loss'].append(0.0)
        history['d_real'].append(0.0)
        history['d_fake'].append(0.0)

        for _,data in iterator:
            
            ### *** Move data to device *** ###
            data['image'] = data['image'].to(config.device)

            ############################
            ### Training discriminator #
            ############################

            ### *** Phase 01 : Training on real data *** ###

            # Zero the gradients
            discriminator.zero_grad()

            # Forward pass
            real_output = discriminator(data['image'])

            # Compute loss
            real_loss = loss_fn(real_output,torch.ones_like(real_output) - config.smooth)

            # Backward pass
            real_loss.backward()

            ### *** Phase 02 : Training on fake data *** ###

            # Generate fake data
            noise = torch.randn(config.batch_size,config.gan_config.latent_dim,1,1).to(config.device)
            fake_data = generator(noise)
            # print(real_output.shape,fake_data.shape)

            # Forward pass
            fake_output = discriminator(fake_data.detach())

            # Compute loss
            fake_loss = loss_fn(fake_output,torch.zeros_like(fake_output) + config.smooth)

            # Backward pass
            fake_loss.backward()

            # Update discriminator
            d_optim.step()

            ############################
            ### Training generator #####
            ############################

            # Zero the gradients
            generator.zero_grad()

            # Forward pass
            fake_output = discriminator(fake_data)

            # Compute loss
            g_loss = loss_fn(fake_output,torch.ones_like(fake_output))

            # Backward pass
            g_loss.backward()

            # Update generator
            g_optim.step()

            ############################
            ### Update tensorboard #####
            ############################

            writer.add_scalar("Loss/Generator",g_loss.item(),total_iterations)
            writer.add_scalar("Loss/Discriminator",(real_loss.item()+fake_loss.item())/2,total_iterations)
            writer.add_scalar("Loss/Real",real_loss.item(),total_iterations)
            writer.add_scalar("Loss/Fake",fake_loss.item(),total_iterations)

            ############################
            ### Update history #########
            ############################

            history['g_loss'][-1] += g_loss.item() / len(dataloader)
            history['d_loss'][-1] += (real_loss.item() + fake_loss.item()) / (2 * len(dataloader))
            history['d_real'][-1] += real_loss.item() / len(dataloader)
            history['d_fake'][-1] += fake_loss.item() / len(dataloader)

            ############################
            ### Save images ############
            ############################

            if total_iterations % config.save_image_every == 0:

                generator.eval()

                with torch.no_grad():
    
                    fake_images = generator(gif_noise).cpu()

                    if config.gan_config.output_activation == 'tanh':
                        fake_images = (fake_images + 1) / 2

                    fake_images = fake_images * 255
                    fake_images = fake_images.type(torch.uint8)

                writer.add_images('Generated Images',fake_images,total_iterations // config.save_image_every)

                fake_images = make_grid(fake_images,h=config.grid_size,w=config.grid_size,gap=4,gap_value=255).numpy()

                images.append(fake_images)

                generator.train()
            
            total_iterations += 1

        ### *** Logging *** ###
        msg = f"Epoch {epoch+1}/{config.total_epochs} : "
        msg += f"Generator Loss : {history['g_loss'][-1]:.4f}, "
        msg += f"Discriminator Loss : {history['d_loss'][-1]:.4f}, "
        msg += f"Real Loss : {history['d_real'][-1]:.4f}, "
        msg += f"Fake Loss : {history['d_fake'][-1]:.4f}"

        print(f"\n{msg}\n")

    history = pd.DataFrame(history)

    return history,images

def main(args : Args) -> None:
    
    ### *** Logger *** ###
    logger = Logger(name="Train GAN",level=logging.INFO)

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
    dataloader = create_dataloader(config)

    ### ***** Initialize Models ***** ###
    logger.info('Initializing Models')
    generator,discriminator = create_models(config)

    ### ***** Initialize Optimizers ***** ###
    logger.info('Initializing Optimizers')
    gen_optim,disc_optim = create_optimizers(config,generator,discriminator)

    ### ***** Initialize Criterias ***** ###
    logger.info('Initializing Criterias')
    loss_fn = create_loss_fn(config)

    ### ***** Train ***** ###
    logger.info('Starting Training')

    history,images = train(
        config,
        generator=generator,
        discriminator=discriminator,
        loss_fn=loss_fn,
        g_optim=gen_optim,
        d_optim=disc_optim,
        dataloader=dataloader
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
        'generator' : generator.state_dict(),
        'discriminator' : discriminator.state_dict()
    },checkpoint_path)

    logger.info('Goodbye!')

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--experiment', type=str, help='Name of the experiment',choices=os.listdir(D.EXPERIMENTS_DIR))

    args = parser.parse_args()

    main(Args(**vars(args)))
