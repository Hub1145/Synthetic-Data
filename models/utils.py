import numpy as np
from sklearn.cluster import KMeans
from typing import List

def create_sliding_windows(data_array, window_size):
    T, N = data_array.shape
    if T < window_size:
        print(f"Warning: Not enough data (T={T}) to create a window of size {window_size}. Returning empty array.")
        return np.empty((0, window_size, N))

    num_windows = T - window_size + 1
    windows = []
    for start_idx in range(num_windows):
        window = data_array[start_idx : start_idx + window_size, :]
        windows.append(window)
    if not windows:
         return np.empty((0, window_size, N))
    return np.stack(windows, axis=0)

def reconstruct_from_windows(orig_array, window_size, window_recons):
    T, N = orig_array.shape
    num_windows = T - window_size + 1
    if num_windows < 1:
        if T < window_size:
             return orig_array
        else:
             return np.zeros((0, N))

    if window_recons.shape[0] != num_windows:
         raise ValueError(f"Number of reconstructed windows ({window_recons.shape[0]}) does not match expected number ({num_windows})")

    recon = np.zeros((T, N), dtype=np.float32)
    recon[:window_size-1] = orig_array[:window_size-1]
    for i in range(window_size - 1, T):
        w_index = i - window_size + 1
        if w_index < len(window_recons):
             recon[i] = window_recons[w_index, -1]
        else:
             if len(window_recons) > 0:
                 recon[i] = window_recons[-1, -1]
             else:
                 recon[i] = orig_array[i]
    return recon
