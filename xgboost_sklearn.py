#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd

# add path of xgboost python module
sys.path.append('/home/loschen/programs/xgboost/wrapper')
import xgboost as xgb

from sklearn.base import BaseEstimator
from sklearn import preprocessing

import qsprLib

class XgboostClassifier(BaseEstimator):
    """
    xgboost<-->sklearn interface
    Chrissly31415 July 2014
    xgboost: https://github.com/tqchen/xgboost
    sklearn: http://scikit-learn.org/stable/index.html
    based on the kaggle forum thread: https://www.kaggle.com/c/higgs-boson/forums/t/8184/public-starting-guide-to-get-above-3-60-ams-score/44691#post44691
    """
    def __init__(self,n_estimators=120,learning_rate=0.1,max_depth=6,subsample=1.0,min_child_weight=1,colsample_bytree=1.0,gamma=0,objective='binary:logistic',eval_metric='auc',booster='gbtree',n_jobs=1,cutoff=0.50,NA=-999.0,alpha_L1=0.0001,lambda_L2=1,silent=1,eval_size=0.0):
	"""
	Constructor
	Parameters: https://github.com/dmlc/xgboost/blob/d3af4e138f7cfa5b60a426f1468908f928092198/doc/parameter.md
	"""
	self.learning_rate = learning_rate
	self.max_depth = max_depth
	self.n_estimators=n_estimators
	self.min_child_weight=min_child_weight
	self.colsample_bytree = colsample_bytree
	self.gamma = gamma
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
	self.eval_size = eval_size
	
	self.isRegressor=False
	self.classes_=-1
	self.xgboost_model=None
	self.encoder= preprocessing.LabelEncoder()
	#self.param['scale_pos_weight'] = 1.0 #scaling can be done also externally| for AMS metric

    def fit(self, lX, ly, sample_weight=None):
	#avoid problems with pandas dataframes and DMatrix
	if isinstance(lX,pd.DataFrame): lX = np.asarray(lX)
	if isinstance(ly,pd.DataFrame): ly = np.asarray(ly)
	
	self.classes_ = np.unique(ly)
	ly = self.encoder.fit_transform(ly)
	
	if sample_weight is not None:
	    sample_weight = np.asarray(sample_weight)
	
	#early stopping!!
	if self.eval_size>0.0:
	    n_test = int(self.eval_size * lX.shape[0])
	    idx_test = np.random.choice(xrange(lX.shape[0]),n_test,False)
	    idx_train = [x for x in xrange(lX.shape[0]) if x not in idx_test]
	    Xeval = lX[idx_test,:]
	    yeval = ly[idx_test]
	    lX = lX[idx_train,:]
	    ly = ly[idx_train]
	    print "Xeval:",Xeval.shape
	    print "X:",lX.shape
	    deval = xgb.DMatrix(Xeval,label=yeval)
	    
	    
        #xgmat = xgb.DMatrix(X, label=y, missing=self.NA, weight=sample_weight)#NA ??
        dtrain = xgb.DMatrix(lX, label=ly, missing=self.NA)#NA=0 as regulariziation->gives rubbish
        
        #set up parameters
	param = {}	 
	param['objective'] = self.objective #'binary:logitraw', 'binary:logistic', 'multi:softprob'
	param['eval_metric'] = self.eval_metric #'auc','mlogloss'	
	param['booster']=self.booster #gblinear
	param['subsample']=self.subsample
	param['min_child_weight']=self.min_child_weight
	param['colsample_bytree']=self.colsample_bytree
	param['gamma']=self.gamma
	param['bst:eta'] =  self.learning_rate
        param['bst:max_depth'] = self.max_depth
        param['nthread'] = self.n_jobs
        param['silent'] = self.silent
        param['num_class']=np.unique(ly).shape[0]      
        param['alpha']=self.alpha_L1
        param['lambda']=self.lambda_L2
        
        plst = param.items()
        
	#watchlist = [ (dtrain,'train') ]
	if self.eval_size>0.0:
	    watchlist  = [(dtrain,'train'),(deval,'eval')]
	    self.xgboost_model = xgb.train(plst, dtrain, num_boost_round=self.n_estimators, evals=watchlist, early_stopping_rounds=None)
	else:
	    watchlist  = [(dtrain,'train')]
	    #self.xgboost_model = xgb.train(plst, dtrain, num_boost_round=self.n_estimators,evals=watchlist)
	    self.xgboost_model = xgb.train(plst, dtrain, num_boost_round=self.n_estimators)

    def predict(self, lX):
	ly = self.predict_proba(lX)
	if 'multi:softprob' in self.objective:
	    ly = np.argmax(ly)

        return self.encoder.inverse_transform(ly.astype(int))
        
    def predict_proba(self,lX):
	#avoid problems with pandas dataframes and DMatrix
	if isinstance(lX,pd.DataFrame): lX = np.asarray(lX)
        xgmat_test = xgb.DMatrix(lX, missing=self.NA)
        ly = self.xgboost_model.predict(xgmat_test)
        return ly
    
    #def set_params(random_state=None):
    #    print "Agugu"
        
    
 
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
