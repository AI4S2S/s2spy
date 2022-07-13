from s2s.RGDR import map_regions
import numpy as np

# create random 3-d array with shape (time, latitude, longitude) 
precur_arr = np.random.randint(0, 100, size=(10, 5, 5))
# create random 3-d array with shape (lag, latitude, longitude) and create region masks 1 and 2.
labels = np.zeros(shape=(1, 5, 5), dtype=int) ; labels[:] = np.nan
labels[:, 1:3, 1:3] = 1 ; labels[:, 4, 3:5] = 2
print(labels)

map_regions.single_split_spatial_mean_regions(precur_arr, labels, name='var', lag_names=['lag1'])