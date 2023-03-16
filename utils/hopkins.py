import random

import numpy as np
from scipy.spatial.distance import cdist


def get_cdist(data, sampling_ratio=0.3):
    
    sampling_ratio = min(max(sampling_ratio, 0.1), 0.5)

    n_samples = int(data.shape[0] * sampling_ratio)

    random_index = list(range(data.shape[0]))
    random.shuffle(random_index)
    random_index = random_index[:n_samples]

    sample_data = data[random_index]
    data = np.delete(data, random_index, 0)

    data_cdist = cdist(data, sample_data).min(axis=0).sum()

    return data_cdist

def get_H(data, sampling_ratio=0.3):
    
    data_cdist = get_cdist(data, sampling_ratio)
    
    ags_data = np.random.uniform(data.max(), data.min(), data.shape)
    
    ags_cdist = get_cdist(ags_data, sampling_ratio)
    
    H_value = ags_cdist / (ags_cdist + data_cdist)
    
    return H_value