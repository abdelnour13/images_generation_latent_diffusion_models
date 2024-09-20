import kaggle
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from definitions import DATA_DIR
from argparse import ArgumentParser
from dataclasses import dataclass
from tqdm.auto import tqdm
from src.utils import seed_everything

@dataclass
class Args:
    datasets: list[str]

seed = 42
api = kaggle.KaggleApi()

seed_everything(seed)

api.authenticate()

def download_celeba():

    dataset_path = os.path.join(DATA_DIR, 'celeb_a')
    api.dataset_download_cli('jessicali9530/celeba-dataset', path=DATA_DIR, unzip=True)

    os.rename(os.path.join(DATA_DIR, 'data'), dataset_path)

def download_anime_faces():

    dataset_path = os.path.join(DATA_DIR, 'anime_faces')
    os.makedirs(dataset_path, exist_ok=True)

    api.dataset_download_cli('soumikrakshit/anime-faces', path=dataset_path, unzip=True)

    images_dir = os.path.join(dataset_path, 'images')
    os.rename(os.path.join(dataset_path, 'data'), images_dir)

    shutil.rmtree(os.path.join(images_dir, 'data'))

    images = os.listdir(images_dir)

    df = pd.DataFrame(columns=['image_id','partition'])

    train_images, val_images = train_test_split(images, test_size=0.1, random_state=seed)
    train_images, _ = train_test_split(train_images, test_size=0.1, random_state=seed)

    df['image_id'] = images
    df['partition'] = df['image_id'].apply(lambda x: 0 if x in train_images else 1 if x in val_images else 2)

    df.to_csv(os.path.join(dataset_path, 'splits.csv'), index=False)

def download_anime_faces_2():

    dataset_path = os.path.join(DATA_DIR, 'anime_faces_2')
    os.makedirs(dataset_path, exist_ok=True)

    api.dataset_download_cli('modassirafzal/anime-faces', path=dataset_path, unzip=True)

    images_dir = os.path.join(dataset_path, 'images')
    images = os.listdir(images_dir)

    df = pd.DataFrame(columns=['image_id','partition'])

    train_images, val_images = train_test_split(images, test_size=0.1, random_state=seed)
    train_images, _ = train_test_split(train_images, test_size=0.1, random_state=seed)

    df['image_id'] = images
    df['partition'] = df['image_id'].apply(lambda x: 0 if x in train_images else 1 if x in val_images else 2)

    df.to_csv(os.path.join(dataset_path, 'splits.csv'), index=False)

def download_art_pictograms():

    # kaggle datasets download -d olgabelitskaya/art-pictogram
    dataset_path = os.path.join(DATA_DIR, 'art_pictograms')
    os.makedirs(dataset_path, exist_ok=True)

    api.dataset_download_cli('olgabelitskaya/art-pictogram', path=dataset_path, unzip=True)

    images_dir = os.path.join(dataset_path, 'pictograms')
    images = os.listdir(images_dir)

    df = pd.DataFrame(columns=['image_id','partition'])

    train_images, val_images = train_test_split(images, test_size=0.1, random_state=seed)
    train_images, _ = train_test_split(train_images, test_size=0.1, random_state=seed)

    df['image_id'] = images
    df['partition'] = df['image_id'].apply(lambda x: 0 if x in train_images else 1 if x in val_images else 2)

    df.to_csv(os.path.join(dataset_path, 'splits.csv'), index=False)

def dowbload_cartoon_faces():

    dataset_path = os.path.join(DATA_DIR, 'cartoon_faces')
    os.makedirs(dataset_path, exist_ok=True)

    ### Images
    api.dataset_download_cli('brendanartley/cartoon-faces-googles-cartoon-set', path=dataset_path, unzip=True)

    images_dir = os.path.join(dataset_path, 'cartoonset100k_jpg')
    
    subfolders = os.listdir(images_dir)

    for subfolder in subfolders:

        subfolder_path = os.path.join(images_dir, subfolder)
        images = os.listdir(subfolder_path)

        for image in tqdm(images, desc=f"Moving images from {subfolder}"):
            shutil.move(os.path.join(subfolder_path, image), os.path.join(images_dir, image))

        os.rmdir(subfolder_path)
    
    images = os.listdir(images_dir)

    df = pd.DataFrame(columns=['image_id','partition'])

    train_images, val_images = train_test_split(images, test_size=0.1, random_state=seed)
    train_images, _ = train_test_split(train_images, test_size=0.1, random_state=seed)

    df['image_id'] = images
    df['partition'] = df['image_id'].apply(lambda x: 0 if x in train_images else 1 if x in val_images else 2)

    df.to_csv(os.path.join(dataset_path, 'splits.csv'), index=False)

    ### Annotations
    api.dataset_download_cli('kirkdco/google-cartoon-faces-attributes', path=dataset_path, unzip=True)

    attributes_df = pd.read_csv(os.path.join(dataset_path, 'cartoon_image_attributes.csv'))
    attributes_df['filename'] = attributes_df['filename'].apply(lambda x: x.split('/')[-1])
    attributes_df['image_id'] = attributes_df['filename']
    attributes_df['filename'].drop(columns=['filename'])
    attributes_df.to_csv(os.path.join(dataset_path, 'cartoon_image_attributes.csv'), index=False)

def download_bitmojie():

    dataset_path = os.path.join(DATA_DIR, 'bitmojie')
    os.makedirs(dataset_path, exist_ok=True)

    api.dataset_download_cli('romaingraux/bitmojis', path=dataset_path, unzip=True)

    images_dir = os.path.join(dataset_path, 'bitmojis')
    images = os.listdir(images_dir)

    df = pd.DataFrame(columns=['image_id','partition'])

    train_images, val_images = train_test_split(images, test_size=0.1, random_state=seed)
    train_images, test_images = train_test_split(train_images, test_size=0.1, random_state=seed)

    df['image_id'] = images
    df['partition'] = df['image_id'].apply(lambda x: 1 if x in val_images else 2 if x in test_images else 0)

    df.to_csv(os.path.join(dataset_path, 'splits.csv'), index=False)

def main(args : Args):
    
    datasets = {
        'celeba': download_celeba,
        'anime_faces': download_anime_faces,
        'cartoon_faces': dowbload_cartoon_faces,
        'anime_faces_2': download_anime_faces_2,
        'art_pictograms': download_art_pictograms,
        'bitmojie': download_bitmojie
    }

    for dataset in args.datasets:

        download_fn = datasets.get(dataset)
        
        if download_fn is not None:
            download_fn()
        else:
            print(f"Dataset {dataset} not found")

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--datasets', nargs='+', default=['celeba', 'anime_faces', 'cartoon_faces'])

    args = parser.parse_args()

    args = Args(**vars(args))

    main(args)