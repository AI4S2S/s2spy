#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 12:17:25 2019

@author: semvijverberg
"""
import inspect
import itertools
import os
import re
import sys
import uuid
from typing import Union

import numpy as np
import pandas as pd
import scipy
import xarray as xr
from statsmodels.sandbox.stats import multicomp

flatten = lambda l: list(itertools.chain.from_iterable(l))
from joblib import Parallel, delayed

try:
    from tigramite.independence_tests import ParCorr
except:
    pass


def correlation_map():
    '''
    Wrapper to calculate correlation/partial correlation maps, see bivariateMI_map.
    Loop over splits & lags
    '''
    raise NotImplementedError

def corr_map(field: xr.DataArray, target: xr.DataArray):
    '''
    Calculate correlation correlation maps, corr_map.
    '''
    raise NotImplementedError

def partial_corr_map(field: xr.DataArray, target: xr.DataArray, z: xr.DataArray):
    '''
    Calculate partial correlation maps, see partial_corr_map.
    '''
    raise NotImplementedError

def adjust_significance_threshold(alpha):
    '''
    Adjust mask in xr_corr with new alpha value
    '''
    raise NotImplementedError

def regression_map():
    '''
    Wrapper to do regression analysis on entire map (Linear, Ridge, Lasso).
    Loop over splits & lags
    '''
    raise NotImplementedError

def regr_map(field: xr.DataArray, target: xr.DataArray, regression_model):
    '''
    Regression analysis on entire map (Linear, Ridge, Lasso).
    '''
    raise NotImplementedError

def store_map():
    '''
    Store all relevent info for correlation map analysis, see store_netcdf
    '''
    raise NotImplementedError

def load_map():
    '''
    load all relevent info of stored correlation map analysis, see load_netcdf
    '''
    raise NotImplementedError



