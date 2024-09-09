import kaggle
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from definitions import DATA_DIR

api = kaggle.KaggleApi()

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

    train_images, val_images = train_test_split(images, test_size=0.1, random_state=42)
    train_images, _ = train_test_split(train_images, test_size=0.1, random_state=42)

    df['image_id'] = images
    df['partition'] = df['image_id'].apply(lambda x: 0 if x in train_images else 1 if x in val_images else 2)

    df.to_csv(os.path.join(dataset_path, 'splits.csv'), index=False)

def main():
    download_celeba()
    download_anime_faces()

if __name__ == '__main__':
    main()