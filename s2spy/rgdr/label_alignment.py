import numpy as np
import pandas as pd
import xarray as xr
from fractions import Fraction

def _get_split_label_dict(split_array: xr.DataArray):
    ''' Function to return a dict of splits numbers as keys 
    and a list of region labels as values for a given lag.'''
    
    split_labels_dict = {}
    #loop over splits
    for split_number, split in enumerate(split_array):
        #select lag
        unique_labels = np.unique(split)
        unique_labels = unique_labels[unique_labels != 0] #0 are no precursor fields
        split_labels_dict[split_number] = list(unique_labels)
    
    return split_labels_dict

def array_label_count(array: np.array):
    '''Function to return a dict of values in an array with their number of occurences.'''
    unique, counts = np.unique(array, return_counts = True)
    count_dict = dict(zip(unique,counts))
    
    return count_dict

def _init_overlap_df(split_array: xr.DataArray, lag: int):
    ''' Function to create a pandas dataframe with a multi index of split 
    numbers as the first index and precursor region labels from every split 
    as the second index. The same multi index is used for the columns. 
    This df can be used to show how labels from different splits overlap. 
    The lag must be given since we can only compare found regions from 
    the same lag over splits.'''

    split_label_dict = _get_split_label_dict(split_array.sel({'i_interval':lag}))

    #create a list of tuples:
    tuple_list = []
    for split_number, label_list in split_label_dict.items():
        for label in label_list:
            tuple_list.append((split_number,label))

    #create multi index with tuple list
    multi_index = pd.MultiIndex.from_tuples(tuple_list, names=('split','label'))

    #create empty dataframe
    df = pd.DataFrame(np.nan, index = multi_index, columns = multi_index)

    #add label cell count column
    df['label_cell_count'] = 0

    #get label cell counts
    for split_number, label_list in split_label_dict.items():
        for label in label_list:
            split = split_array.sel({'split':split_number,'i_interval':lag})
            split_labelmask = split.where(split == label,0)
            label_count = array_label_count(split_labelmask)
            df.at[(split_number, label), 'label_cell_count'] = label_count[label]

    return df

def mysign(x: int):
    ''' Simple function to return sign of an int, with zero as 0.'''
    if x > 0: 
        return 1
    elif x == 0:
        return 0
    else:
        return -1

def overlap_labels(split_array: xr.DataArray, lag: int):
    '''
    Function to create a dataframe that shows how many grid cells 
    of a found precursor region of a lag in a training split overlap with
    a region found in another split at the same lag.
    Args:
        - split_array: an xr.DataArray containing the cluster labels with 
        dimensions 'lag' and 'split'
        - lag: given lag for overlap investigation
    '''
    
    #initialize empty dataframe
    df = _init_overlap_df(split_array, lag = lag)

    #get list of split numbers
    split_number_list = list(np.unique(df.index.get_level_values(0)))

    #loop over splits
    for split_number in split_number_list:
        
        #select a split, at a lag, get the cluster_labels
        split = split_array.sel({'split':split_number,'i_interval':lag})
        #get labels from split for given lag
        labels = df.loc[split_number,:].index.get_level_values(0)
        
        #loop over every label in the split lag
        for label in labels:

            #mask the array for label
            split_labelmask = split.where(split == label,0)
            #convert where equal to label to 1
            split_labelmask = split_labelmask.where(split_labelmask != label, 1)
            
            #create other split list
            other_split_number_list = [other_split_number for other_split_number in 
            split_number_list if other_split_number != split_number]
            
            #loop over other splits
            for other_split_number in other_split_number_list:

                #select same lag in other split
                other_split = split_array.sel({'split':other_split_number,'i_interval':lag})
                #get list of labels in this split
                other_split_labels = df.loc[other_split_number,:].index.get_level_values(0)
                #create list with labels in other split with same sign as label
                other_split_labels_samesign = [other_label for other_label in 
                other_split_labels if mysign(other_label) == mysign(label)]

                #loop over labels of same sign
                for other_label in other_split_labels_samesign:

                    #mask for other_label
                    other_split_labelmask = other_split.where(other_split == other_label, 0)
                    #convert where equal to label to 1
                    other_split_labelmask = other_split.where(other_split_labelmask != other_label, 1)

                    #check where regions overlap
                    overlap_array = split_labelmask.where(split_labelmask == other_split_labelmask)
                    #count the number of cells in the overlap array
                    overlap_count_dict = array_label_count(overlap_array)
                    if 1 not in overlap_count_dict:
                        overlap_count = 0
                    else:
                        overlap_count = overlap_count_dict[1]

                    #append to df
                    df.at[
                        (split_number,label), 
                        (other_split_number, other_label)
                        ] = overlap_count / df.loc[(split_number, label), 'label_cell_count']

    return df

def overlap_superlabels(split_list: list, lag: int):
    '''
    Function to create a list with intersection arrays of found 
    precursor regions between splits over the same lag
    Args:
        - split_list: a list of cluster field arrays
        - lag: given lag for which alignment is needed
    '''

    split_label_dict = _get_split_label_dict(split_list, lag=lag)
    
    #unique label list
    unique_label_list = []
    for counter, labels in enumerate(split_label_dict.values()):
        #first one
        if counter == 0:
            unique_label_list.append(labels)
        else:
            new_unique_label = set(unique_label_list) - set(labels)
            unique_label_list.append(new_unique_label)
    
    super_label_dict = {}

    for label in unique_label_list:
        for counter, split in enumerate(split_list):
            #select lag in split and get cluster labels
            cluster_labels = split.sel({'i_interval':lag}).cluster_labels
            #mask label
            cluster_labels_masked = cluster_labels.where(cluster_labels == label, 0)
            #convert where equal to label to 1
            cluster_labels_masked = cluster_labels_masked.where(cluster_labels_masked != label, 1)

            if counter == 0:
                superlabel_array = cluster_labels_masked
            else:
                superlabel_array = super_label_array * cluster_labels_masked
        
        super_label_dict[label] = superlabel_array

    return super_label_dict
