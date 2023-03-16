import logging
import pickle
from pathlib import Path

from tqdm import trange

from utils.hopkins import get_H

logging.basicConfig(filename='training.log',
                    datefmt='%H:%M:%S',
                    format='%(asctime)s %(message)s',
                    encoding='utf-8',
                    level=logging.DEBUG,
                    filemode='a')

path_pkls = Path('ckp_vae_fn_2')
assert path_pkls.exists()

for epoch in trange(50):
    
    with open(path_pkls / f'{epoch}.pkl', 'rb') as f:
        zs = pickle.load(f)
    
    h = get_H(zs)
    
    logging.debug(f'epoch {epoch} H: {h:.5f}')