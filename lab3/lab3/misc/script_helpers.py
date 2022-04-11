import numpy as np
import os 

def check_kwds(path, kwds):
    if kwds is None:
        return True
    else:
        word_in_path = [word in path for word in kwds]
        return np.all(word_in_path)

def get_sima_paths(paths, keywords=None):
    sima_paths = []
    for path in paths:
        for dirpath, _, _ in os.walk(path):
            if dirpath.endswith('.sima') and check_kwds(dirpath, keywords):
                sima_paths.append(dirpath)

    return sorted(sima_paths)