## Images Generation with Latent Diffusion models

This repository contains the code for generating images with the Latent Diffusion models, as described in the paper [Latent Diffusions for Generative Modeling](https://arxiv.org/abs/2006.11239),tested on the following datasets : [Anime Faces dataset](https://www.kaggle.com/datasets/soumikrakshit/anime-faces), [Google cartoon faces](https://www.kaggle.com/datasets/brendanartley/cartoon-faces-googles-cartoon-set) and [Bitmoji faces
](https://www.kaggle.com/datasets/romaingraux/bitmojis).

## VQVAE Training results

- VQVAE Output during training

![VQVAE Output during training](experiments/vqvae_cartoon/images.gif)

- Original Vs Reconstructed images

<img src="experiments/vqvae_cartoon/images.png" width="300" />


## Latent diffusion model results

- Latent diffusion output during training (every 2000 iterations)

![Latent diffusion output during training (every 2000 iterations)](experiments/diffusion_cartoon/images.gif)

- Latent diffusion generation process (decoded every 20 timesteps)

![ Latent diffusion generation process (decoded every 20 timesteps)](experiments/diffusion_cartoon/generation_process.gif)

## Setup

```
git clone git@github.com:abdelnour13/images_generation_latent_diffusion_models.git
conda create -n latent-diffusion
conda activate latent-diffusion
pip install requirements.txt
```

## Download data

```
python3 download.py [-h] --datasets {celeb_a,anime_faces,cartoon_faces,anime_faces_2,art_pictograms,bitmojie}
```

## Create an experiment

```
python3 create.py [-h] [--on-exists {error,overwrite}] --name NAME --type {vqvae,diffusion}
```

## Models training

- To train the VQVAE

```
cd src/training
python3 vqvae.py --experiment EXPERIMENT
```

- To train the Latent diffusion model

```
cd src/training
python3 diffusion.py --experiment EXPERIMENT
```