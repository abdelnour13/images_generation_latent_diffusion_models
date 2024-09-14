## Images Generation with Latent Diffusion models

This repository contains the code for generating images with the Latent Diffusion models, as described in the paper [Latent Diffusions for Generative Modeling](https://arxiv.org/abs/2106.06759),tested on [Anime Faces dataset](https://www.kaggle.com/datasets/soumikrakshit/anime-faces).

## VQVAE Training results

- VQVAE Output during training (every 500 iterations)

![VQVAE Output during training (every 500 iterations)](experiments/vqvae/images.gif)

## Latent diffusion model results

- Latent diffusion output during training (every 2000 iterations)

![Latent diffusion output during training (every 2000 iterations)](experiments/diffusion/images.gif)

- Latent diffusion generation process (decoded every 20 timesteps)

![ Latent diffusion generation process (decoded every 20 timesteps)](experiments/diffusion/generation_process.gif)

## Setup

```
git clone git@github.com:abdelnour13/images_generation_latent_diffusion_models.git
conda create -n latent-diffusion
conda activate latent-diffusion
pip install requirements.txt
```

## Download data

```
python3 download.py [-h] --datasets {celeb_a,anime_faces,celeb_a anime_faces}
```

## Create experiment

```
python3 create.py [-h] [--on-exists {error,overwrite}] --name NAME --type {vqvae,diffusion,gan}
```

## Train a model

- To train a VQVAE

```
cd src/training
python3 vqvae.py --experiment EXPERIMENT
```

- To train a Latent diffusion model

```
cd src/training
python3 diffusion.py --experiment EXPERIMENT
```