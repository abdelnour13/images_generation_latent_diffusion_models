import os
from dataclasses import dataclass

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(ROOT_DIR, 'data')
EXPERIMENTS_DIR = os.path.join(ROOT_DIR, 'experiments')

@dataclass
class DatasetConfig:
    root : str
    images_dir: str
    splits_file: str
    metadata_file: str | None = None
    variants_file: str | None = None

DATASETS = {
    "celeb_a" : DatasetConfig(
        root = os.path.join(DATA_DIR, 'celeb_a'),
        images_dir = os.path.join(DATA_DIR, 'celeb_a', 'img_align_celeba', 'img_align_celeba'),
        splits_file = os.path.join(DATA_DIR, 'celeb_a', 'list_eval_partition.csv'),
    ),
    "anime_faces" : DatasetConfig(
        root = os.path.join(DATA_DIR, 'anime_faces'),
        images_dir = os.path.join(DATA_DIR, 'anime_faces', 'images'),
        splits_file = os.path.join(DATA_DIR, 'anime_faces', 'splits.csv')
    ),
    'cartoon_faces' : DatasetConfig(
        root = os.path.join(DATA_DIR, 'cartoon_faces'),
        images_dir = os.path.join(DATA_DIR, 'cartoon_faces', 'cartoonset100k_jpg'),
        splits_file = os.path.join(DATA_DIR, 'cartoon_faces', 'splits.csv'),
        metadata_file = os.path.join(DATA_DIR, 'cartoon_faces', 'cartoon_image_attributes.csv'),
        variants_file = os.path.join(DATA_DIR, 'cartoon_faces', 'cartoon_attributes_variants.csv'),
    ),
    "anime_faces_2" : DatasetConfig(
        root = os.path.join(DATA_DIR, 'anime_faces_2'),
        images_dir = os.path.join(DATA_DIR, 'anime_faces_2', 'images'),
        splits_file = os.path.join(DATA_DIR, 'anime_faces_2', 'splits.csv')
    ),
    'art_pictograms' : DatasetConfig(
        root = os.path.join(DATA_DIR, 'art_pictograms'),
        images_dir = os.path.join(DATA_DIR, 'art_pictograms', 'pictograms'),
        splits_file = os.path.join(DATA_DIR, 'art_pictograms', 'splits.csv')
    ),
    'bitmojie' : DatasetConfig(
        root = os.path.join(DATA_DIR, 'bitmojie'),
        images_dir = os.path.join(DATA_DIR, 'bitmojie', 'bitmojis'),
        splits_file = os.path.join(DATA_DIR, 'bitmojie', 'splits.csv')
    ),
}