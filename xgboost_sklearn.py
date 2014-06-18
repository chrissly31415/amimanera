#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Chrissly31415 August 2014

import sys
import numpy as np

# add path of xgboost python module
sys.path.append('/home/loschen/programme/xgboost-master/python')
import xgboost as xgb

from sklearn.base import BaseEstimator


class XgboostClassifier(BaseEstimator):
    """
    xgboost sklearn interface
    xgboost: 
    sklearn:
    """
    def __init__(self):
        self.alpha = 1.0

    def fit(self, X, y, sample_weight=None):
        pass

    def predict(self, X):
        y = np.empty((X.shape[0], 1), dtype=np.float64)
        return y
        
    def predict_proba(self,X)
	y = np.empty((X.shape[0], 1), dtype=np.float64)
        return y