#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd

# add path of xgboost python module
sys.path.append('/home/loschen/programs/xgboost/wrapper')
import xgboost as xgb

from sklearn.base import BaseEstimator

import qsprLib

class XgboostClassifier(BaseEstimator):
    """
    xgboost<-->sklearn interface
    Chrissly31415 July 2014
    xgboost: https://github.com/tqchen/xgboost
    sklearn: http://scikit-learn.org/stable/index.html
    based on the kaggle forum thread: https://www.kaggle.com/c/higgs-boson/forums/t/8184/public-starting-guide-to-get-above-3-60-ams-score/44691#post44691
    """
    def __init__(self,n_estimators=120,learning_rate=0.1,max_depth=6,subsample=1.0,objective='binary:logistic',eval_metric='auc',booster='gbtree',n_jobs=1,cutoff=0.50,NA=-999.0,alpha_L1=0.0001,lambda_L2=1,silent=1):
	"""
	Constructor
	"""
	self.learning_rate = learning_rate
	self.max_depth = max_depth
	self.n_estimators=n_estimators	
	self.cutoff=cutoff
	self.NA = NA
	self.booster=booster
	self.objective=objective
	self.eval_metric=eval_metric
	self.subsample=subsample
	self.n_jobs=n_jobs
	self.silent = silent
	self.alpha_L1 = alpha_L1
	self.lambda_L2 = lambda_L2
	
	self.isRegressor=False
	self.classes_=-1
	self.xgboost_model=None
	#self.param['scale_pos_weight'] = 1.0 #scaling can be done also externally| for AMS metric

    def fit(self, X, y, sample_weight=None):
	#avoid problems with pandas dataframes and DMatrix
	if isinstance(X,pd.DataFrame): X = np.asarray(X)
	if isinstance(y,pd.DataFrame): y = np.asarray(y)	
	self.classes_ = np.unique(y)
	
	if sample_weight is not None:
	    sample_weight = np.asarray(sample_weight)
	
        #xgmat = xgb.DMatrix(X, label=y, missing=self.NA, weight=sample_weight)#NA ??
        xgmat = xgb.DMatrix(X, label=y, missing=self.NA)#NA=0 as regulariziation->gives rubbish
        
        #plst = self.param.items()+[('eval_metric')]
        #set up parameters
	param = {}	 
	param['objective'] = self.objective #'binary:logitraw', 'binary:logistic', 'multi:softprob'
	param['eval_metric'] = self.eval_metric #'auc','mlogloss'	
	param['booster']=self.booster #gblinear
	param['subsample']=self.subsample
	param['bst:eta'] =  self.learning_rate
        param['bst:max_depth'] = self.max_depth
        param['nthread'] = self.n_jobs
        param['silent'] = self.silent
        param['num_class']=np.unique(y).shape[0] 
        
        param['alpha']=self.alpha_L1
        param['lambda']=self.lambda_L2
        
        plst = param.items()
        
	watchlist = [ (xgmat,'train') ]
        self.xgboost_model = xgb.train(plst, xgmat, self.n_estimators, watchlist)

    def predict(self, X):
	y = self.predict_proba(X)
	idx = np.argmax(y,axis=1)
        return idx
        
    def predict_proba(self,X):
	#avoid problems with pandas dataframes and DMatrix
	if isinstance(X,pd.DataFrame): X = np.asarray(X)
        xgmat_test = xgb.DMatrix(X, missing=self.NA)
        y = self.xgboost_model.predict(xgmat_test)
        return y
 
class XgboostRegressor(XgboostClassifier):    
    def __init__(self):
	super(XgboostClassifier, self).__init__()
	self.isRegressor=True

    
        
if __name__=="__main__":
    """
    test function
    """
    Xtrain,ytrain,Xtest,wtrain=higgs.prepareDatasets(1000)
    model = XgboostClassifier()
    model.fit(Xtrain,ytrain,wtrain)
    y = model.predict(Xtest)
