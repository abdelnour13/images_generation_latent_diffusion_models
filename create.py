import definitions as D
import os
from argparse import ArgumentParser
from src.training.vqvae import Config as VQVAEConfig
from src.training.diffusion import Config as DiffusionConfig
from src.utils import save_json
from dataclasses import asdict,dataclass
from typing import Literal

expirements = {
    'vqvae': VQVAEConfig,
    'diffusion': DiffusionConfig,
}

@dataclass
class Args:
    on_exists: Literal['error','overwrite']
    name: str
    type: str

def main(args : Args) -> None:
    
    expirement_dir = os.path.join(D.EXPERIMENTS_DIR, args.name)
    config_path = os.path.join(expirement_dir, 'config.json')

    if os.path.exists(expirement_dir) and args.on_exists == 'error':
        raise ValueError(f'Experiment {args.name} already exists')
    
    os.makedirs(expirement_dir, exist_ok=True)

    config_cls = expirements[args.type]
    config_obj = config_cls()
    config_dict = asdict(config_obj)

    save_json(config_dict, config_path)

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--on-exists', type=str, default='error', help='What to do if the experiment already exists', choices=['error','overwrite'])
    parser.add_argument('--name', type=str, required=True, help='Name of the experiment')
    parser.add_argument('--type', type=str, required=True, help='Type of the experiment folder',choices=list(expirements.keys()))

    args = parser.parse_args()
    args = Args(**vars(args))

    main(args)