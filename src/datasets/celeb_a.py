import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset
from typing import Optional,Callable,Any

class CelebADataset(Dataset):

    def __init__(self,
        root : str,
        img_transform : Optional[Callable] = None,
        label_transform : Optional[Callable] = None,
        split : Optional[str] = None             
    ) -> None:
        
        self.root = root
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.split = split

        self.attributes = self._get_attributes()
        self.partitions = self._get_partitions()

        if split is not None:

            partitions = ['train', 'val', 'test']
            selected_images = self.partitions[self.partitions['partition'] == partitions.index(split)]['image_id']
            self.attributes = self.attributes[self.attributes['image_id'].isin(selected_images)]

    def _get_attributes(self) -> pd.DataFrame:

        path = os.path.join(self.root, 'list_attr_celeba.csv')
        attributes = pd.read_csv(path)

        return attributes
    
    def _get_partitions(self) -> pd.DataFrame:

        path = os.path.join(self.root, 'list_eval_partition.csv')
        partition = pd.read_csv(path)

        return partition
    
    def __len__(self) -> int:
        return len(self.attributes)
    
    def __getitem__(self, idx : int) -> dict:

        row = self.attributes.iloc[idx]

        image_id = row['image_id']
        image_path = os.path.join(self.root, 'img_align_celeba','img_align_celeba', image_id)
        image = Image.open(image_path).convert('RGB')

        if self.img_transform:
            image = self.img_transform(image)

        metadata = row.drop('image_id').values.astype(int)

        if self.label_transform:
            metadata = self.label_transform(metadata)

        return {
            'id' : idx,
            'image' : image,
            'metadata' : metadata
        }