#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np

# add path of xgboost python module
sys.path.append('/home/loschen/programme/xgboost-master/python')
import xgboost as xgb

from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt

import qsprLib
import higgs


class XgboostClassifier(BaseEstimator):
    """
    xgboost<-->sklearn interface
    Chrissly31415 July 2014
    xgboost: https://github.com/tqchen/xgboost
    sklearn: http://scikit-learn.org/stable/index.html
    based on the forum thread: https://www.kaggle.com/c/higgs-boson/forums/t/8184/public-starting-guide-to-get-above-3-60-ams-score/44691#post44691
    """
    def __init__(self,n_estimators=120,learning_rate=0.1,max_depth=6,n_jobs=1,cutoff=0.50,NA=-999.0,verbose=1):
	"""
	Constructor
	"""
	self.learning_rate = learning_rate
	self.max_depth = max_depth
	self.n_jobs= n_jobs
	self.n_estimators=n_estimators
	self.xgboost_model=None
	self.cutoff=cutoff
	self.NA = NA
	self.verbose = verbose
	
	self.param = {}
	self.param['objective'] = 'binary:logitraw'
	#self.param['objective'] = 'binary:logistic'
	self.param['eval_metric'] = 'auc'
	
	self.param['scale_pos_weight'] = 594.0 #scaling can be done also externally

    def fit(self, X, y, sample_weight=None):
	#avoid problems with pandas dataframes and DMatrix
	X = np.asarray(X)
	y = np.asarray(y)
	
	sample_weight = np.asarray(sample_weight)
	
        xgmat = xgb.DMatrix(X, label=y, missing=self.NA, weight=sample_weight)
        
        #set up parameters
        self.param['bst:eta'] =  self.learning_rate
        self.param['bst:max_depth'] = self.max_depth
        self.param['nthread'] = self.n_jobs
        self.param['silent'] = self.verbose
        
        plst = self.param.items()+[('eval_metric', 'ams@0.153')]
	watchlist = [ (xgmat,'train') ]
        self.xgboost_model = xgb.train(plst, xgmat, self.n_estimators, watchlist)

    def predict(self, X):
	y = self.predict_proba(X)[:,1]
	ones = y >= self.cutoff
	zeros = y < self.cutoff	
	y[ones]=1
	y[zeros]=0
        return y
        
    def predict_proba(self,X):
	#avoid problems with pandas dataframes and DMatrix
	X = np.asarray(X)
        xgmat_test = xgb.DMatrix(X, missing=self.NA)
        y = self.xgboost_model.predict(xgmat_test)
        #scale data to [0,1]
        y = (y - np.min(y))/(np.max(y)-np.min(y))
        
        proba = np.ones((y.shape[0], 2), dtype=np.float64)
        proba[:, 1] = y
        proba[:, 0] = proba[:, 0] - proba[:, 1]
        
        #plt.hist(y,bins=50)
	#plt.show()
        return proba
        
        
if __name__=="__main__":
    """
    test function
    """
    Xtrain,ytrain,Xtest,wtrain=higgs.prepareDatasets(1000)
    model = XgboostClassifier()
    model.fit(Xtrain,ytrain,wtrain)
    print "Prediction..."
    y = model.predict(Xtest)
    print y