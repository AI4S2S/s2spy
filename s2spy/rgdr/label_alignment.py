import numpy as np
import pandas as pd
from fractions import Fraction

def _get_split_region_dict(split_list: list, lag: int):
    ''' Function to return a dict of splits numbers as keys 
    and a list of region labels as values for one lag.'''
    
    split_labels_dict = {}
    #loop over splits
    for split_number, split in enumerate(split_list):
        #select lag
        split_lag = split.sel({'i_interval':lag}).cluster_labels
        unique_labels = np.unique(split_lag)
        unique_labels = unique_labels[unique_labels != 0] #0 are no precursor fields
        split_labels_dict[split_number] = unique_labels
    
    return split_labels_dict

def _create_overlap_df(split_list: list, lag: int):
    ''' Function to create an empty pandas dataframe from a
    list of arrays with found region labels for every split.'''

    split_label_dict = _get_split_region_dict(split_list, lag=lag)

    #create a list of tuples:
    tuple_list = []
    for split, label_list in split_label_dict.items():
        for label in label_list:
            tuple_list.append((split,label))

    #create multi index with tuple list
    multi_index = pd.MultiIndex.from_tuples(tuple_list, names=('split','label'))

    #create empty dataframe
    df = pd.DataFrame(np.nan, index = multi_index, columns = list(split_label_dict.keys()))

    #add region count column
    df['region_count'] = 0

    return df


def array_label_count(array: np.array):
    '''Function to return a dict of values in an array with their number of occurences.'''
    unique, counts = np.unique(array, return_counts = True)
    count_dict = dict(zip(unique,counts))
    
    return count_dict

def mysign(x: int):
    ''' Simple function to return sign of an int, with zero as 0.'''
    if x > 0: 
        return 1
    elif x == 0:
        return 0
    else:
        return -1

def align_overlap(split_list: list, lag: int):
    '''
    Function to create and show a dataframe that shows how many grid cells 
    of a found precursor region of a lag in a training split are also found
    in the same location in a different split at the same lag.
    Args:
        - split_list: a list of cluster field arrays
        - lag: given lag for which alignment is needed
    '''
    
    #initialize empty dataframe
    df = _create_overlap_df(split_list, lag = lag)
    
    #create split number list
    split_number_list = []
    for count, array in enumerate(split_list):
        split_number_list.append(count)

    #loop over splits in the split list
    for split_number, split in enumerate(split_list):
        
        #select lag
        split_lag = split.sel({'i_interval':lag}).cluster_labels
        #get unique labels from split for given lag
        unique_labels = np.unique(split_lag)
        unique_labels = unique_labels[unique_labels != 0] #0 are no precursor fields
        
        #loop over every label in the split lag
        for label in unique_labels:

            #mask the array per unique label
            masked_split_lag = split_lag.where(split_lag == label,0)
            #convert where equal to label to 1
            masked_split_lag = masked_split_lag.where(masked_split_lag != label, 1)
            #count number of cells in selected precursor region
            region_count = array_label_count(masked_split_lag)[1]
            #add region count to df
            df.at[(split_number,label), 'region_count'] = region_count
            
            #create other split list
            other_split_list = [other_split_number for other_split_number in 
            split_number_list if other_split_number != split_number]
            
            #loop over other splits
            for other_split_number in other_split_list:

                #select same lag in other split
                other_split_lag = split_list[other_split_number].sel({'i_interval':lag}).cluster_labels
                #mask for all labels of the same sign as label
                other_split_lag_sign = other_split_lag.where(np.sign(other_split_lag) == mysign(label))
                #set all values other than np.nan values to 1
                other_split_lag_sign_masked = other_split_lag_sign.where(np.isnan(other_split_lag_sign), 1)
                #check where regions overlap
                overlap_array = masked_split_lag.where(masked_split_lag == other_split_lag_sign_masked)
                #count the number of cells in the overlap array
                overlap_count_dict = array_label_count(overlap_array)
                if 1 not in overlap_count_dict:
                    overlap_count = 0
                else:
                    overlap_count = overlap_count_dict[1]

                #append to df
                df.at[(split_number,label), other_split_number] = overlap_count / region_count

    return df
