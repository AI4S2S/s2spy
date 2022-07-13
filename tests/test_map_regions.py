from s2spy.RGDR import map_regions
import numpy as np

def test_spatial_mean_regions():
    # create random 3-d array with shape (time, latitude, longitude) 
    np.random.seed(0)
    precur_arr = np.random.randint(0, 100, size=(10, 5, 5), seed=1)
    # create random 3-d array with shape (lag, latitude, longitude) and create region masks 1 and 2.
    labels = np.zeros(shape=(1, 5, 5)) ; labels[:] = np.nan
    labels[:, 1:3, 1:3] = 1 ; labels[:, 4, 3:5] = 2
    print(labels)

    df = map_regions.spatial_mean_regions(precur_arr, labels, name='var', lag_names=['lag1'])
    