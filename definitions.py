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

DATASETS = {
    "celeb_a" : DatasetConfig(
        root = os.path.join(DATA_DIR, 'celeb_a'),
        images_dir=os.path.join(DATA_DIR, 'celeb_a', 'img_align_celeba', 'img_align_celeba'),
        splits_file=os.path.join(DATA_DIR, 'celeb_a', 'list_eval_partition.csv')
    ),
    "anime_faces" : DatasetConfig(
        root = os.path.join(DATA_DIR, 'anime_faces'),
        images_dir=os.path.join(DATA_DIR, 'anime_faces', 'images'),
        splits_file=os.path.join(DATA_DIR, 'anime_faces', 'splits.csv')
    )
}