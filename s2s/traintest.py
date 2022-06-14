"""AI4S2S train/test splitting method.

Cross-validation approachess used for train/test splitting.
Methods included: (1)random stratified (2)kfold
Methods to be added: 
"""
from typing import Union
import numpy as np
import pandas as pd
from sklearn.model_selection import PredefinedSplit, StratifiedKFold
import sklearn.model_selection as sk_ms

def testgroups_to_original(
                            groups: Union[pd.DataFrame, pd.Series],
                            test_groups: np.ndarray=None, 
                            gap_prior: int=None, 
                            gap_after: int=None
                            ):
    '''
    Map test_groups or test_years to AdventCalender date intervals.

    args:
        groups: output of ``get_cv_groups``.
        test_groups: output of a cv strategy from
        ``random_stratified``, ``leave_n_out``, ``kfold``, ``one_step_ahead``, 
        Gap_prior: skip n groups/years prior to every test group/year
        Gap_after: skip n groups/years after every test group/year
    '''
    
    # n_repeats is build in for compatitability with RepeatedKFold
    testsetidx = np.zeros( (n_repeats, groups.size), dtype=int)
    testsetidx[:] = -999

    n = 0 ;
    for i, test_fold_idx in enumerate(test_groups):
        # convert idx to grouplabel (year or dateyrgroup)
        if test_yrs is None:
            test_fold = [uniqgroups[i] for i in test_fold_idx]
        else:
            test_fold = test_fold_idx
        # assign testsetidx
        if max(1,i) % kfold == 0 and kfold!=1:
            n += 1
        # if i == 2: break
        # print(test_fold)
        for j, gr in enumerate(groups):
            # if j == 21: break
            # print(gr)
            if gr in list(test_fold):
                # print(gr, i, idx, i % kfold )
                testsetidx[n,j] = i % kfold

    def gap_traintest(testsetidx, groups, gap):
        ign = np.zeros((np.unique(testsetidx).size, testsetidx.size))
        for f, i in enumerate(np.unique(testsetidx)):
            test_fold = testsetidx.squeeze()==i
            roll_account_traintest_gr = gap*groups[groups==np.unique(groups)[1]].size
            ign[f] = np.roll(test_fold, roll_account_traintest_gr).astype(float)
            ign[f] = (ign[f] - test_fold) == 1 # everything prior to test
            if np.sign(gap) == -1:
                ign[f][roll_account_traintest_gr:] = False
            elif np.sign(gap) == 1:
                ign[f][:roll_account_traintest_gr] = False
        return ign.astype(bool)

    if gap_prior is not None:
        ignprior = gap_traintest(testsetidx, groups, -gap_prior)
    if gap_after is not None:
        ignafter = gap_traintest(testsetidx, groups, gap_after)

    TrainIsTrue = []
    for n in range(n_repeats):
        for f, i in enumerate(np.unique(testsetidx)):
            # print(n, f, i)
            # if -999, No Train Test split, all True
            if method[:15].lower() == 'timeseriessplit':
                if i == -999:
                    continue
                mask = np.array(testsetidx[n] < i, dtype=int)
                mask[testsetidx[n]>i] = -1
            elif method[:5] == 'split':
                if i == -999:
                    continue
                mask = np.logical_or(testsetidx[n]!=i, testsetidx[n]==-999)
            else:
                mask = np.logical_or(testsetidx[n]!=i, testsetidx[n]==-999)
                # print(n,i,mask[mask].size)

            if gap_prior is not None:
                # if gap_prior, mask values will become -1 for unused (training) data
                mask = np.array(mask, dtype=int) ; mask[ignprior[f]] = -1
            if gap_after is not None:
                # same as above for gap_after.
                mask = np.array(mask, dtype=int) ; mask[ignafter[f]] = -1

            TrainIsTrue.append(pd.DataFrame(data=mask.T,
                                            columns=['TrainIsTrue'],
                                            index=index))
    df_TrainIsTrue = pd.concat(TrainIsTrue , axis=0, keys=range(n_repeats*kfold))
    return df_TrainIsTrue   

def kfold(
        groups: Union[pd.DataFrame, pd.Series],
        kfold: int=10, 
        seed: int=1,
        shuffle=True,
        gap_prior: int=None, 
        gap_after: int=None):
    '''
    kfold cross validator based on scikit-learn.
    args:
        groups: output of ``get_cv_groups``.
                timeseries: 1-d timeseries of which upper and lower terciles are assigned with 1 and -1, respectively.
                Subsequently, the split is created such that the 1 and -1 events are approx. evenly distribution over the train and test subset.
        kfold: number of train/test splits.
        seed: seed for random generator. 
        Gap_prior: skip n groups/years prior to every test group/year
        Gap_after: skip n groups/years after every test group/year        
    '''
    uniqgroups = np.unique(groups)
    cv = sk_ms.KFold(n_splits=kfold, shuffle=shuffle,
                                 random_state=seed)
    test_groups = [list(f[1]) for f in cv.split(uniqgroups)]
    return testgroups_to_original(groups, test_groups, gap_prior, gap_after)

def manual(test_years):

    # align import timeseries to account data not aggregated yet
    df_TrainIsTrue = test_years
    kfold = df_TrainIsTrue.index.levels[0].size ; n_repeats = 1

    df_index = traintestgroups.index ; input_index = df_TrainIsTrue.loc[0].index
    input_data = np.array(df_TrainIsTrue.values).reshape(kfold, -1)
    idx = [True if input_index[i] in df_index else False for i in range(input_index.size)]
    mask = np.repeat(np.array(idx)[None,:], kfold, axis=0)
    # .loc very slow on lots of data
    # df_TrainIsTrue = df_TrainIsTrue.loc[pd.IndexSlice[:,traintestgroups.index],:]
    # below is faster, but still slower than numpy
    # df_TrainIsTrue = df_TrainIsTrue.loc[pd.MultiIndex.from_product([np.arange(kfold), df_index])]
    df_TrainIsTrue = pd.DataFrame(input_data[mask],
                                    index=pd.MultiIndex.from_product([np.arange(kfold), df_index]),
                                    columns=['TrainIsTrue'])
    return df_TrainIsTrue

def random_stratified(
                    groups: Union[pd.DataFrame, pd.Series], 
                    timeseries: Union[pd.DataFrame, pd.Series], 
                    kfold: int=10, 
                    seed: int=1):
    '''
   
    args:
        groups: output of ``get_cv_groups``.
                timeseries: 1-d timeseries of which upper and lower terciles are assigned with 1 and -1, respectively.
                Subsequently, the split is created such that the 1 and -1 events are approx. evenly distribution over the train and test subset.
        kfold: number of train/test splits.
        seed: seed for random generator. 
    '''
    groups = groups.loc[timeseries.index] # align groups with timeseries
    cv = _get_cv_accounting_for_years(timeseries, kfold, seed, groups)
    test_groups = cv.uniqgroups
    return test_groups

def _get_cv_accounting_for_years(timeseries=pd.DataFrame, kfold: int=5,
                                seed: int=1, groups=None):
    '''
    Train-test split that gives priority to keep data of same year as blocks,
    datapoints of same year are very much not i.i.d. and should be seperated.
    Parameters
    ----------
    total_size : int
        total length of dataset.
    kfold : int
        prefered number of folds, however, if folds do not fit the number of
        years, kfold is incremented untill it does.
    seed : int, optional
        random seed. The default is 1.
    groups: pd.DataFrame, optional
        if groups is None, the years are used instead of the group label. 
        Discussion needed if we want to support this.
    Returns
    -------
    cv : sk-learn cross-validation generator
    '''
    # if dealing with subseasonal data, there is a lot of autocorrelation.
    # it is best practice to keep the groups of target dates within a year well
    # seperated, therefore:
    if groups is None and np.unique(timeseries.index.year).size != timeseries.size:
        # find where there is a gap in time, indication of seperate RV period
        gapdays = (timeseries.index[1:] - timeseries.index[:-1]).days
        adjecent_dates = gapdays > (np.median(gapdays)+gapdays/2)
        n_gr = gapdays[gapdays > (np.median(gapdays)+gapdays/2)].size + 1
        dategroupsize = np.argmax(adjecent_dates) + 1
        groups = np.repeat(np.arange(0,n_gr), dategroupsize)
        if groups.size != timeseries.size: # else revert to keeping years together
            groups = timeseries.index.year
    elif groups is None and np.unique(timeseries.index.year).size == timeseries.size:
        groups = timeseries.index.year # annual data, no autocorrelation groups
    else:
        pass

    high_normal_low = timeseries.groupby(groups).sum()
    high_normal_low[(high_normal_low > high_normal_low.quantile(q=.66)).values] = 1
    high_normal_low[(high_normal_low < high_normal_low.quantile(q=.33)).values] = -1
    high_normal_low[np.logical_and(high_normal_low!=1, high_normal_low!=-1)] = 0
    freq  = high_normal_low

    cv_strat = StratifiedKFold(n_splits=kfold, shuffle=True,
                               random_state=seed)
    test_gr = []
    for i, j in cv_strat.split(X=freq.index, y=freq.values):
        test_gr.append(j)

    label_test = np.zeros(timeseries.size , dtype=int)
    for i, test_fold in enumerate(test_gr):
        for j, gr in enumerate(groups):
            if j in list(test_fold):
                label_test[j] = i

    cv = PredefinedSplit(label_test)
    cv.uniqgroups = test_gr
    return cv


ALL_METHODS = {
    "leave_n_out": s2s.traintest.leave_n_out,
    "randstrat": s2s.traintest.rand_strat,
    "random": s2s.traintest.random,
    "split": s2s.traintest.split,
    "timeseriessplit": s2s.traintest.timeseries_split,
    "repeated_kfold": s2s.traintest.repeated_kfold,
}
