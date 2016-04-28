#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ensemble helper tools

Chrissly31415
October,September 2014

using stacking for ensemble building
for stacking versus blending: see:

http://mlwave.com/kaggle-ensembling-guide/


"""

from home_depot import *
from FullModel import *

import sys,os
import itertools
from random import randint

from scipy.optimize import fmin, fmin_cobyla, minimize

from sklearn.externals.joblib import Parallel, delayed, logger
from sklearn.base import clone
from sklearn import preprocessing


def createModels():

    global idx
    ensemble = []
    """
     #creating best DF -> './data/store_db1b.h5'
    Xtest, Xtrain, ytrain, idx, sample_weight, Xval, yval = prepareDataset(quickload=False,#'./data/store_db1b.h5',
                                                                           seed=42,
                                                                           nsamples=-1,
                                                                           #holdout='holdout_chrisslz.csv',
                                                                           holdout=False,#use holdout=False for full Xtrain
                                                                           keepFeatures=None,
                                                                           dropFeatures=['product_title','product_description','product_info'],
                                                                           labelEncode=['search_term']+[u'MFG Brand Name'],
                                                                           stemmData=None,
                                                                           createCommonWords=True,
                                                                           useAttributes=[u'MFG Brand Name'],
                                                                           createVerticalFeatures=True,
                                                                           computeSim={'columns': ['product_info','search_term'],'doSVD': 250, 'vectorizer': TfidfVectorizer(min_df=10,  max_features=None, strip_accents='unicode', analyzer='word',ngram_range=(1, 3), use_idf=True,smooth_idf=True,sublinear_tf=True,stop_words = stop_words,token_pattern=r'\w{1,}')},
                                                                           doTFIDF=['search_term','product_title'],
                                                                           n_components=[50,50],
                                                                           cleanData=True,
                                                                           merge_product_infos = True,
                                                                           spellchecker = True,
                                                                           word2vecFeates= True,
                                                                           computeAddFeates=True)

    os.rename('./data/store.h5','./data/store_db1b.h5')
    Xtrain.to_csv("Xtrain_chrisslz_db1b.csv",index=False)
    Xtest.to_csv("Xtest_chrisslz_db1b.csv",index=False)


     #creating best DF -> './data/store_db1c.h5'
    Xtest, Xtrain, ytrain, idx, sample_weight, Xval, yval = prepareDataset(quickload=False,#'./data/store_db1c.h5',
                                                                           seed=42,
                                                                           nsamples=-1,
                                                                           use_new_spellchecker=True,
                                                                           holdout='holdout_chrisslz.csv',
                                                                           #holdout=False,#use holdout=False for full Xtrain
                                                                           keepFeatures=None,
                                                                           dropFeatures=['product_title','product_description','product_info'],
                                                                           labelEncode=['search_term']+[u'MFG Brand Name'],
                                                                           stemmData=None,
                                                                           createCommonWords=True,
                                                                           useAttributes=[u'MFG Brand Name'],
                                                                           createVerticalFeatures=True,
                                                                           computeSim={'columns': ['product_info','search_term'],'doSVD': 250, 'vectorizer': TfidfVectorizer(min_df=10,  max_features=None, strip_accents='unicode', analyzer='word',ngram_range=(1, 3), use_idf=True,smooth_idf=True,sublinear_tf=True,stop_words = stop_words,token_pattern=r'\w{1,}')},
                                                                           doTFIDF=['search_term','product_title'],
                                                                           n_components=[50,50],
                                                                           cleanData=True,
                                                                           merge_product_infos = True,
                                                                           spellchecker = True,
                                                                           word2vecFeates= True,
                                                                           computeAddFeates=True)

    os.rename('./data/store.h5','./data/store_db1c.h5')
    #Xtrain.to_csv("Xtrain_chrisslz_spellcheck.csv",index=False)
    #Xtest.to_csv("Xtest_chrisslz_spellcheck.csv",index=False)



    #XGB1b RMSE = 0.457 EVAL = ? PL = 0.4548 !
    Xtest, Xtrain, ytrain, idx, sample_weight, Xval, yval = prepareDataset('./data/store_db1b.h5')
    model = XgboostRegressor(n_estimators=2000,learning_rate=0.005,max_depth=10, NA=0,subsample=.5,colsample_bytree=0.75,min_child_weight=5,n_jobs=4,objective='reg:linear',eval_metric='rmse',booster='gbtree',silent=1,eval_size=0.0)
    xmodel = XModel("xgb1b_f8r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,Xval=Xval,yval=yval,cv_labels=cv_labels,bag_mode=False)
    ensemble.append(xmodel)

    #XGB1c RMSE = 0.457 EVAL = ? PL = 0.4548 !
    Xtest, Xtrain, ytrain, idx, sample_weight, Xval, yval = prepareDataset('./data/store_db1c.h5')
    model = XgboostRegressor(n_estimators=2000,learning_rate=0.005,max_depth=10, NA=0,subsample=.5,colsample_bytree=0.75,min_child_weight=5,n_jobs=4,objective='reg:linear',eval_metric='rmse',booster='gbtree',silent=1,eval_size=0.0)
    xmodel = XModel("xgb1c_f8r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,Xval=Xval,yval=yval,cv_labels=cv_labels,bag_mode=False)
    ensemble.append(xmodel)

    #XRF3 RMSE = 0.459 EVAL=0.461
    Xtest, Xtrain, ytrain, idx, sample_weight, Xval, yval = prepareDataset('./data/store_db1b.h5')
    model =  ExtraTreesRegressor(n_estimators=250,max_depth=None,min_samples_leaf=5,n_jobs=2, max_features=3*Xtrain.shape[1]/4)
    xmodel = XModel("xrf3_f8r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,Xval=Xval,yval=yval,cv_labels=cv_labels,bag_mode=False)
    ensemble.append(xmodel)


    #XRF4 RMSE = 0.459 EVAL=0.462
    Xtest, Xtrain, ytrain, idx, sample_weight, Xval, yval = prepareDataset('./data/store_db1c.h5')
    model =  ExtraTreesRegressor(n_estimators=250,max_depth=None,min_samples_leaf=5,n_jobs=2, max_features=3*Xtrain.shape[1]/4)
    xmodel = XModel("xrf4_f8r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,Xval=Xval,yval=yval,cv_labels=cv_labels,bag_mode=False)
    ensemble.append(xmodel)


    #XGB0 RMSE = 0.4645
    Xtest, Xtrain, ytrain, idx, sample_weight, Xval, yval = prepareDataset('./data/store_db1b.h5')
    model = XgboostRegressor(n_estimators=200,learning_rate=0.05,max_depth=10, NA=0,subsample=.75,colsample_bytree=0.75,min_child_weight=5,n_jobs=4,objective='reg:linear',eval_metric='rmse',booster='gbtree',silent=1,eval_size=0.0)
    xmodel = XModel("xgb0_f8r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,Xval=Xval,yval=yval,cv_labels=cv_labels,bag_mode=False)
    ensemble.append(xmodel)


    #bagxgb1 RMSE = 0.456 EVAL = 0.457  time for n_est = 10 ->
    Xtest, Xtrain, ytrain, idx, sample_weight, Xval, yval = prepareDataset('./data/store_db1b.h5')
    model = XgboostRegressor(n_estimators=800,learning_rate=0.01,max_depth=10, NA=0,subsample=.5,colsample_bytree=0.75,min_child_weight=5,n_jobs=4,objective='reg:linear',eval_metric='rmse',booster='gbtree',silent=1,eval_size=0.0)
    model = BaggingRegressor(model,n_estimators=10, max_samples=1.0, max_features=0.9, bootstrap=True,verbose=1)
    xmodel = XModel("bagxgb1_f8r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,Xval=Xval,yval=yval,cv_labels=cv_labels,bag_mode=False)
    ensemble.append(xmodel)


    #NN1
    Xtest, Xtrain, ytrain, idx, sample_weight, Xval, yval = prepareDataset('./data/store_db1b.h5')
    model  = KerasNN(dims=Xtrain.shape[1],nb_classes=1,nb_epoch=35,learning_rate=1E-4,validation_split=0.0,batch_size=128,verbose=1,activation='sigmoid', layers=[260,480], dropout=[0.17,0.16],loss='mse')
    model = Pipeline([('scaler', StandardScaler()), ('nn',model)])
    #model = BaggingRegressor(model,n_estimators=10, max_samples=1.0, max_features=1.0)
    xmodel = XModel("nn1_f8r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,Xval=Xval,yval=yval,cv_labels=cv_labels,bag_mode=False)
    ensemble.append(xmodel)

    #NN2
    Xtest, Xtrain, ytrain, idx, sample_weight, Xval, yval = prepareDataset('./data/store_db1b.h5')
    model  = KerasNN(dims=Xtrain.shape[1],nb_classes=1,nb_epoch=20,learning_rate=1E-4,validation_split=0.0,batch_size=512,verbose=1,activation='sigmoid', layers=[256,256], dropout=[0.1,0.0],loss='mse') # best
    model = Pipeline([('scaler', StandardScaler()), ('nn',model)])
    #model = BaggingRegressor(model,n_estimators=10, max_samples=1.0, max_features=1.0)
    xmodel = XModel("nn2_f8r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,Xval=Xval,yval=yval,cv_labels=cv_labels,bag_mode=False)
    ensemble.append(xmodel)

    #NN1 bag
    Xtest, Xtrain, ytrain, idx, sample_weight, Xval, yval = prepareDataset('./data/store_db1b.h5')
    model  = KerasNN(dims=Xtrain.shape[1],nb_classes=1,nb_epoch=35,learning_rate=1E-4,validation_split=0.0,batch_size=128,verbose=1,activation='sigmoid', layers=[260,480], dropout=[0.17,0.16],loss='mse')
    model = Pipeline([('scaler', StandardScaler()), ('nn',model)])
    model = BaggingRegressor(model,n_estimators=10, max_samples=1.0, max_features=1.0)
    xmodel = XModel("bagnn1_f8r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,Xval=Xval,yval=yval,cv_labels=cv_labels,bag_mode=False)
    ensemble.append(xmodel)

    #XGB2b RMSE =   EVAL= PL=
    Xtest, Xtrain, ytrain, idx, sample_weight, Xval, yval = prepareDataset(quickload=False,
                                                                           seed=42,
                                                                           nsamples=-1, holdout='holdout_chrisslz.csv',
                                                                           keepFeatures=None,
                                                                           dropFeatures=['product_title','product_description','product_info'],
                                                                           labelEncode=['search_term'],
                                                                           stemmData=None,
                                                                           createCommonWords=True,
                                                                           useAttributes=None,
                                                                           computeSim={'columns': ['product_info','search_term'],'doSVD': 250, 'vectorizer': CountVectorizer(binary=False, min_df=3,  max_features=None, lowercase=True,analyzer="word",ngram_range=(1,2),stop_words=stop_words,strip_accents='unicode') },
                                                                           doTFIDF=['search_term','product_title'],
                                                                           n_components=[50,50],
                                                                           cleanData=True,
                                                                           merge_product_infos = True,
                                                                           computeAddFeates=True,
                                                                           spellchecker=False)

    model = XgboostRegressor(n_estimators=800,learning_rate=0.01,max_depth=10, NA=0,subsample=.5,colsample_bytree=0.75,min_child_weight=5,n_jobs=4,objective='reg:linear',eval_metric='rmse',booster='gbtree',silent=1,eval_size=0.0)
    xmodel = XModel("xgb2b_f8r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,Xval=Xval,yval=yval,cv_labels=cv_labels,bag_mode=False)
    ensemble.append(xmodel)



    #XGB3 RMSE = 0.467
    Xtest, Xtrain, ytrain, idx, sample_weight, Xval, yval = prepareDataset(quickload=False,#'./data/store_db2.h5',
                                                                           seed=42,
                                                                           nsamples=-1, holdout='holdout_chrisslz.csv',
                                                                           keepFeatures=None,
                                                                           dropFeatures=['product_title','product_description','product_info'],
                                                                           labelEncode=['search_term','MFG Brand Name','Material'],
                                                                           stemmData=None,
                                                                           createCommonWords=False,
                                                                           computeSim={'columns': ['product_description','search_term'],'doSVD': 300, 'vectorizer': CountVectorizer(binary=False, min_df=3,  max_features=None, lowercase=True,analyzer="word",ngram_range=(1,2),stop_words=stop_words,strip_accents='unicode')},
                                                                           doTFIDF= None,#['search_term','product_description'],
                                                                           n_components=None,#[75,75],
                                                                           cleanData=False,
                                                                           merge_product_infos = False,
                                                                           computeAddFeates=False,
                                                                           spellchecker = True,
                                                                           useAttributes = ['MFG Brand Name','Material'],
                                                                           word2vecFeates = True,
                                                                           removeCorr = True)

    model = XgboostRegressor(n_estimators=800,learning_rate=0.01,max_depth=10, NA=0,subsample=.5,colsample_bytree=0.75,min_child_weight=5,n_jobs=4,objective='reg:linear',eval_metric='rmse',booster='gbtree',silent=1,eval_size=0.0)
    xmodel = XModel("xgb3_f8r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,Xval=Xval,yval=yval,cv_labels=cv_labels,bag_mode=False)
    ensemble.append(xmodel)


    #XGB4 RMSE = 0.47
    attribute_list = [u'MFG Brand Name']
    #attribute_list = [u'MFG Brand Name',u'Material',u'Product Width (in.)',u'Color Family',u'Product Height (in.)',u'Product Depth (in.)',u'Product Weight (lb.)',u'Color/Finish',u'Certifications and Listings']
    Xtest, Xtrain, ytrain, idx, sample_weight, Xval, yval = prepareDataset(quickload=False,#'./data/store_large.h5',
                                                                           seed=42,
                                                                           nsamples=-1, holdout='holdout_chrisslz.csv',
                                                                           keepFeatures=None,
                                                                           dropFeatures=[u'product_title',u'product_description',u'product_info']+ attribute_list,
                                                                           labelEncode=['search_term'],
                                                                           stemmData=None,
                                                                           createCommonWords=True,
                                                                           computeSim = None,#{'columns': ['product_info','search_term'],'doSVD': 200, 'vectorizer': TfidfVectorizer(min_df=10,  max_features=None, strip_accents='unicode', analyzer='word',ngram_range=(1, 3), use_idf=True,smooth_idf=True,sublinear_tf=True,stop_words = stop_words,token_pattern=r'\w{1,}')},
                                                                           doTFIDF=['search_term','product_info'],
                                                                           n_components=[200,400],
                                                                           cleanData=True,
                                                                           merge_product_infos = [u'product_title',u'product_description'] +attribute_list,
                                                                           spellchecker = True,
                                                                           useAttributes = attribute_list,
                                                                           word2vecFeates = True,
                                                                           removeCorr = True,
                                                                           computeAddFeates = False,
                                                                           computeAddFeates_new=False,
                                                                           query_correction=False)

    model = XgboostRegressor(n_estimators=800,learning_rate=0.01,max_depth=10, NA=0,subsample=.5,colsample_bytree=0.75,min_child_weight=5,n_jobs=4,objective='reg:linear',eval_metric='rmse',booster='gbtree',silent=1,eval_size=0.0)
    xmodel = XModel("xgb4_f8r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,Xval=Xval,yval=yval,cv_labels=cv_labels,bag_mode=False)
    ensemble.append(xmodel)



    #XGB5 RMSE = ? somehow is doe not run through??
    Xtest, Xtrain, ytrain, idx, sample_weight, Xval, yval = prepareDataset(quickload=False,
                                                                           seed=42,
                                                                           nsamples=-1, holdout='holdout_chrisslz.csv',
                                                                           keepFeatures=None,
                                                                           dropFeatures=[u'product_title',u'product_description',u'product_info'],
                                                                           labelEncode=['search_term'],
                                                                           stemmData=True,
                                                                           createCommonWords=True,
                                                                           oneHotenc = None,
                                                                           computeSim = None,#{'columns': ['product_info','search_term'],'doSVD': 500, 'vectorizer': TfidfVectorizer(min_df=10,  max_features=None, strip_accents='unicode', analyzer='word',ngram_range=(1, 3), use_idf=True,smooth_idf=True,sublinear_tf=True,stop_words = stop_words,token_pattern=r'\w{1,}')},
                                                                           doTFIDF={'columns': ['search_term','product_title'],'vectorizer': TfidfVectorizer(min_df=100,  max_features=None, strip_accents='unicode', analyzer='char',ngram_range=(1, 2), use_idf=True,smooth_idf=True,sublinear_tf=True,stop_words = None,token_pattern=r'\w{1,}')},
                                                                           n_components=[75,300],
                                                                           cleanData=True,
                                                                           spellchecker = True,
                                                                           useAttributes = None,
                                                                           removeCorr = False,
                                                                           computeAddFeates = False,
                                                                           computeAddFeates_new=False,
                                                                           query_correction=False)

    model = XgboostRegressor(n_estimators=800,learning_rate=0.01,max_depth=10, NA=0,subsample=.5,colsample_bytree=0.75,min_child_weight=5,n_jobs=4,objective='reg:linear',eval_metric='rmse',booster='gbtree',silent=1,eval_size=0.0)
    xmodel = XModel("xgb5_f8r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,Xval=Xval,yval=yval,cv_labels=cv_labels,bag_mode=False)
    ensemble.append(xmodel)


    #XGB6 RMSE =
    Xtest, Xtrain, ytrain, idx, sample_weight, Xval, yval = prepareDataset(quickload=False,#'./data/store_db5.h5',
                                                                           seed=42,
                                                                           nsamples=-1, holdout='holdout_chrisslz.csv',
                                                                           keepFeatures=None,
                                                                           dropFeatures=[u'product_title',u'product_description',u'product_info'],
                                                                           labelEncode=['search_term'],
                                                                           stemmData=True,
                                                                           createCommonWords=True,
                                                                           oneHotenc = None,
                                                                           computeSim = {'columns': ['product_title','search_term'],'doSVD': 250, 'vectorizer': TfidfVectorizer(min_df=10,  max_features=None, strip_accents='unicode', analyzer='word',ngram_range=(1, 3), use_idf=True,smooth_idf=True,sublinear_tf=True,stop_words = stop_words,token_pattern=r'\w{1,}')},
                                                                           doTFIDF=None,#['search_term','product_info'],
                                                                           n_components=None,#[75,200],
                                                                           cleanData=True,
                                                                           merge_product_infos = False,
                                                                           spellchecker = True,
                                                                           useAttributes = None,
                                                                           word2vecFeates = False,
                                                                           removeCorr = False,
                                                                           computeAddFeates = True,
                                                                           computeAddFeates_new=False,
                                                                           query_correction=False)

    model = XgboostRegressor(n_estimators=800,learning_rate=0.01,max_depth=10, NA=0,subsample=.5,colsample_bytree=0.75,min_child_weight=5,n_jobs=4,objective='reg:linear',eval_metric='rmse',booster='gbtree',silent=1,eval_size=0.0)
    xmodel = XModel("xgb6_f8r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,Xval=Xval,yval=yval,cv_labels=cv_labels,bag_mode=False)
    ensemble.append(xmodel)



    #XRF2
    attribute_list = [u'MFG Brand Name',u'Material']
    Xtest, Xtrain, ytrain, idx, sample_weight, Xval, yval = prepareDataset(quickload=False,#'./data/store_db3.h5',
                                                                           seed=42,
                                                                           nsamples=-1, holdout='holdout_chrisslz.csv',
                                                                           keepFeatures=None,
                                                                           dropFeatures=[u'product_title',u'product_description',u'product_info']+ attribute_list,
                                                                           labelEncode=['search_term'],
                                                                           stemmData=None,
                                                                           createCommonWords=True,
                                                                           computeSim = {'columns': ['product_info','search_term'],'doSVD': 200, 'vectorizer': TfidfVectorizer(min_df=10,  max_features=None, strip_accents='unicode', analyzer='word',ngram_range=(1, 3), use_idf=True,smooth_idf=True,sublinear_tf=True,stop_words = stop_words,token_pattern=r'\w{1,}')},
                                                                           doTFIDF=['search_term','product_info'],
                                                                           n_components=[75,200],
                                                                           cleanData=True,
                                                                           merge_product_infos = [u'product_title',u'product_description'] +attribute_list,
                                                                           spellchecker = True,
                                                                           useAttributes = attribute_list,
                                                                           word2vecFeates = True,
                                                                           removeCorr = True,
                                                                           computeAddFeates = False,
                                                                           computeAddFeates_new=True,
                                                                           query_correction=False)
    model =  ExtraTreesRegressor(n_estimators=250,max_depth=None,min_samples_leaf=5,n_jobs=2, max_features=3*Xtrain.shape[1]/3)
    xmodel = XModel("xrf2_f8r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,Xval=Xval,yval=yval,cv_labels=cv_labels,bag_mode=False)
    ensemble.append(xmodel)


    #XSVR2 RMSE =
    Xtest, Xtrain, ytrain, idx, sample_weight, Xval, yval = prepareDataset(quickload='./data/store_db1.h5',
                                                                           seed=42,
                                                                           nsamples=-1, holdout='holdout_chrisslz.csv',
                                                                           keepFeatures=None,
                                                                           dropFeatures=['product_title','product_description','product_info'],
                                                                           labelEncode=['search_term'],
                                                                           stemmData=None,
                                                                           createCommonWords=True,
                                                                           useAttributes=None,
                                                                           computeSim={'columns': ['product_info','search_term'],'doSVD': 250, 'vectorizer': TfidfVectorizer(min_df=10,  max_features=None, strip_accents='unicode', analyzer='word',ngram_range=(1, 3), use_idf=True,smooth_idf=True,sublinear_tf=True,stop_words = stop_words,token_pattern=r'\w{1,}')},
                                                                           doTFIDF=['search_term','product_title'],
                                                                           n_components=[50,50],
                                                                           cleanData='load',
                                                                           merge_product_infos = True,
                                                                           computeAddFeates=True)

    model = KernelRidge(kernel='linear')
    model = Pipeline([('scaler', StandardScaler()), ('model',model)])
    xmodel = XModel("svr1_f8r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,Xval=Xval,yval=yval,cv_labels=cv_labels,bag_mode=False)
    ensemble.append(xmodel)


    #LR1 RMSE=4735
    Xtest, Xtrain, ytrain, idx, sample_weight, Xval, yval = prepareDataset(quickload='./data/store_db1b.h5')
    model = Pipeline([('scaler', StandardScaler()), ('nn', RidgeCV())])
    xmodel = XModel("lr1_f8r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,Xval=Xval,yval=yval,cv_labels=cv_labels,bag_mode=False)
    ensemble.append(xmodel)

    #RF1
    Xtest, Xtrain, ytrain, idx, sample_weight, Xval, yval = prepareDataset(quickload='./data/store_db1b.h5')
    model = RandomForestRegressor(n_estimators=250,max_depth=None,min_samples_leaf=5,n_jobs=2, max_features=Xtrain.shape[1]/3,oob_score=False)
    xmodel = XModel("rf1_f8r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,Xval=Xval,yval=yval,cv_labels=cv_labels,bag_mode=False)
    ensemble.append(xmodel)


    #KNN1
    Xtest, Xtrain, ytrain, idx, sample_weight, Xval, yval = prepareDataset(quickload='./data/store_db1b.h5')
    model = KNeighborsRegressor(n_neighbors=30)
    xmodel = XModel("knn1_f8r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,Xval=Xval,yval=yval,cv_labels=cv_labels,bag_mode=False)
    ensemble.append(xmodel)
    """



    for m in ensemble:
        m.summary()
    return (ensemble)


def finalizeModel(m, use_proba=True):
    """
    Make predictions and save them
    """
    print "Make predictions and save them..."
    m.summary()

    # put data to data.frame and save
    # OOB DATA
    m.oob_preds = pd.DataFrame(np.asarray(m.oob_preds), columns=['oob'])

    # validation
    if hasattr(m, 'val_preds') and m.val_preds is not None:
        m.val_preds = pd.DataFrame(np.asarray(m.val_preds), columns=['val'])

    # TESTSET prediction
    m.preds = pd.DataFrame(np.asarray(m.preds), columns=['prediction'])

    # save final model
    allpred = pd.concat([m.preds, m.oob_preds])
    # submission data is first, train data is last!
    filename = "./data/" + m.name + ".csv"
    print "Saving oob + predictions as csv to:", filename
    allpred.to_csv(filename, index=False)

    # XModel.saveModel(m,"/home/loschen/Desktop/datamining-kaggle/higgs/data/"+m.name+".pkl")
    XModel.saveCoreData(m, "./data/" + m.name + ".pkl")
    return (m)


def saveTrainData(ensemble):
    """
    parallel oob creation
    """
    for m in ensemble:
        print "Saving data for model:", m.name
        XModel.saveDataSet(m)


def loadDataSet(ensemble):
    """
    parallel oob creation
    """
    basedir = "./data/"
    for i, model in enumerate(ensemble):

        xmodel = XModel.loadModel(basedir + model)

        Xtrain, Xtest = XModel.loadDataSet(xmodel)
        print "model: %-20s %20r %20r %20r" % (xmodel.name, Xtrain.shape, Xtest.shape, type(xmodel.classifier))


def createOOBdata(ensemble, repeats=1, n_folds=10, n_jobs=1, score_func='log_loss', verbose=False, calibrate=False,
                  use_proba=True):
    """
    parallel oob creation
    """
    global funcdict

    for m in ensemble:
        print m.name

    for m in ensemble:
        bag_mode = m.bag_mode
        print "\nComputing oob predictions for:", m.name
        print m.classifier.get_params
        if m.class_names is not None:
            n_classes = len(m.class_names)
        else:
            n_classes = 1
        print "n_classes", n_classes

        oob_preds = np.zeros((m.ytrain.shape[0], n_classes, repeats),dtype=np.float32)
        preds = np.zeros((m.Xtest.shape[0], n_classes, repeats))
        val_preds = None
        if m.Xval is not None:
            val_preds = np.zeros((m.yval.shape[0], n_classes, repeats))

        oobscore = np.zeros(repeats)
        maescore = np.zeros(repeats)

        # outer loop
        for j in xrange(repeats):
            if m.cv_labels is not None:
                print "Using cv-labels..."
                cv = LabelKFold(m.cv_labels, n_folds=n_folds)
                #cv = LabelShuffleSplit(m.cv_labels,n_iter=2,test_size=0.2)

            else:
                print "Normal random split for cv..."
                cv = StratifiedKFold(m.ytrain, n_folds=n_folds, shuffle=True, random_state=None)

            scores = np.zeros(len(cv))

            # parallel stuff
            parallel = Parallel(n_jobs=n_jobs, verbose=True,
                                pre_dispatch='2*n_jobs')

            # parallel run, returns a list of oob predictions
            results = parallel(
                delayed(fit_and_score)(clone(m.classifier), m.Xtrain.copy(), m.ytrain, train, test,
                                       sample_weight=m.sample_weight, use_proba=use_proba, returnModel=bag_mode) for train, test in cv)

            for i, (train, test) in enumerate(cv):
                oob_pred, cv_model = results[i]
                if use_proba:
                    oob_pred = oob_pred[:,1]
                oob_pred = oob_pred.reshape(oob_pred.shape[0], n_classes)
                oob_preds[test, :, j] = oob_pred

                scores[i] = funcdict[score_func](m.ytrain[test], oob_preds[test, :, j])

                if bag_mode:
                    print "Using cv models for test set(bag_mode)..."
                    if use_proba:
                        p = cv_model.predict_proba(m.Xtest)
                        p = p.reshape(p.shape[0], n_classes)
                        preds[:, :, j] = p
                    else:
                        p = cv_model.predict(m.Xtest)
                        p = p.reshape(p.shape[0], n_classes)
                        preds[:, :, j] = p

                        if m.Xval is not None:
                            raise Exception("Currently not supported in Bag mode!")
                            # p = cv_model.predict(m.Xval)
                            # p = p.reshape(p.shape[0], n_classes)
                            # val_preds[:, :, j] = p
                            # print "Fold %d - score:%0.3f " % (i,scores[i])
                            # scores_mae[i]=funcdict['mae'](ly[test],oob_preds[test,j])

            oobscore[j] = funcdict[score_func](m.ytrain, oob_preds[:, :, j])
            # maescore[j]=funcdict['mae'](ly,oob_preds[:,j])

            print "Iteration:", j,
            print " <score>: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std()),
            print " score,oob: %0.3f" % (oobscore[j])
        # print " ## <mae>: %0.3f (+/- %0.3f)" % (scores_mae.mean(), scores_mae.std()),
        # print " score3,oob: %0.3f" %(maescore[j])

        # simple averaging of blending
        m.oob_preds = np.mean(oob_preds, axis=2)

        score_oob = funcdict[score_func](m.ytrain, m.oob_preds)
        print "Summary: <score,oob>: %6.3f +- %6.3f   score,oob-total: %0.3f (after %d repeats)\n" % (
            oobscore.mean(), oobscore.std(), score_oob, repeats)

        orig_classifier = clone(m.classifier)
        m.classifier = clone(orig_classifier)
        if not bag_mode:
            # Train full model on total train data
            print "Training on full train set..."
            Xtrain_ = m.Xtrain
            ly_ = m.ytrain
            if m.sample_weight is not None:
                print "... with sample weights"
                sample_weight_ = m.sample_weight

                m.classifier.fit(Xtrain_, ly_, sample_weight_)
            else:
                m.classifier.fit(Xtrain_, ly_)

            if m.Xval is not None:
                print "Prediction for val set...",
                if use_proba:
                    m.val_preds = m.classifier.predict_proba(m.Xval)[:,1]
                else:
                    m.val_preds = m.classifier.predict(m.Xval)
                # check
                score = funcdict[score_func](m.yval, m.val_preds)
                print " score,validation: %0.4f" % (score)

            else:
                print "Predicting on test set..."
                if use_proba:
                    m.preds = m.classifier.predict_proba(m.Xtest)[:,1]
                else:
                    m.preds = m.classifier.predict(m.Xtest)

            if m.Xval is not None:
                print "Re-training on train & val set..."
                Xtrain_ = pd.concat([m.Xtrain, m.Xval])
                ly_ = np.hstack((m.ytrain.ravel(), m.yval.ravel()))
                if m.sample_weight is not None:
                    raise Exception("Not supported for now...")

                #here we need to clone and retrain!
                m.classifier = clone(orig_classifier)
                m.classifier.fit(Xtrain_, ly_)

                print "Predicting on test set..."
                if use_proba:
                    print m.Xtest.describe()
                    m.preds = m.classifier.predict_proba(m.Xtest)[:,1]
                else:
                    m.preds = m.classifier.predict(m.Xtest)


        else:
            print "bag_mode: averaging all cv classifier results"
            # print preds[:10]
            m.preds = np.mean(preds, axis=2)
            if m.Xval is not None:
                raise Exception("Currently not supported for Holdout....")

        m = finalizeModel(m, use_proba=use_proba)
        del oob_preds
        del preds
        if m.Xval is not None:
            del val_preds
    return ensemble


def fit_and_score(xmodel, X, y, train, valid, sample_weight=None, scale_wt=None, use_proba=False, returnModel=True):
    """
    Score function for parallel oob creation
    """

    if isinstance(X, pd.DataFrame):
        X = X.values

    if isinstance(y, pd.DataFrame):
        y = y.values

    Xtrain = X[train]
    Xvalid = X[valid]

    ytrain = y[train]

    if sample_weight is not None:
        print "Using sample weight...", sample_weight[train]
        xmodel.fit(Xtrain, ytrain, sample_weight=sample_weight[train])
    else:

        xmodel.fit(Xtrain, ytrain)

    if use_proba:
        # saving out-of-bag predictions
        local_pred = xmodel.predict_proba(Xvalid)
    # prediction for test set
    # classification/regression
    else:
        local_pred = xmodel.predict(Xvalid)
    if returnModel:
        return local_pred, xmodel
    else:
        return local_pred, None


def trainEnsemble(ensemble, mode='linear', score_func='log_loss', useCols=None, addMetaFeatures=False, use_proba=True,
                  dropCorrelated=False, skipCV=False, subfile=""):
    """
    Train the ensemble
    """

    #TODO keep models in memory

    basedir = "./data/"

    for i, model in enumerate(ensemble):
        print ''.join(['-'] * 60)
        print "Loading model:", i, " name:", model
        xmodel = XModel.loadModel(basedir + model)
        class_names = xmodel.class_names
        if class_names is None:
            class_names = ['Class']
        print "OOB data:", xmodel.oob_preds.shape
        if hasattr(xmodel, 'Xval') and xmodel.Xval is not None:
            print "Holdout data:", xmodel.val_preds.shape
        print "pred data:", xmodel.preds.shape
        print "y train:", xmodel.ytrain.shape

        if i > 0:
            xmodel.oob_preds.columns = [model + "_" + n for n in class_names]
            Xtrain = pd.concat([Xtrain, xmodel.oob_preds], axis=1)
            Xtest = pd.concat([Xtest, xmodel.preds], axis=1)
            if hasattr(xmodel, 'Xval') and xmodel.Xval is not None:
                Xval = pd.concat([Xval, xmodel.val_preds], axis=1)

        else:
            Xtrain = xmodel.oob_preds
            Xtest = xmodel.preds
            y = xmodel.ytrain
            colnames = [model + "_" + n for n in class_names]
            Xtrain.columns = colnames
            Xval = None
            yval = None
            if hasattr(xmodel, 'Xval') and xmodel.Xval is not None:
                Xval = xmodel.val_preds
                yval = xmodel.yval
                print Xval.shape

    Xtest.columns = Xtrain.columns
    if hasattr(xmodel, 'Xval') and xmodel.Xval is not None:
        Xval.columns = Xtrain.columns

    print Xtrain.columns
    print Xtrain.shape

    # print "spearman-correlation:\n",Xtrain.corr(method='spearman')
    print "pearson-correlation :\n", Xtrain.corr(method='pearson')

    # print Xtrain.describe()
    print Xtest.shape
    # print Xtest.describe()

    if mode is 'classical':
        results = classicalBlend(ensemble, Xtrain, Xtest, y, valpreds=Xval, yval=yval, score_func=score_func,
                                 use_proba=use_proba, skipCV=skipCV,
                                 subfile=subfile, cv_labels=xmodel.cv_labels, dropCorrelated=dropCorrelated)
    elif mode is 'mean':
        results = linearBlend(ensemble, Xtrain, Xtest, y, Xval=Xval, yval=yval, score_func=score_func, takeMean=True,
                              subfile=subfile,
                              dropCorrelated=dropCorrelated)
    elif mode is 'voting':
        results = voting_multiclass(ensemble, Xtrain, Xtest, y, score_func=score_func, n_classes=1, subfile=subfile,
                                    dropCorrelated=dropCorrelated)
    elif mode is 'oob':
        return (Xtest, Xtrain, y, None, xmodel.cv_labels, None)

    else:
        results = linearBlend(ensemble, Xtrain, Xtest, y, Xval=Xval, yval=yval, score_func=score_func, takeMean=False,
                              subfile=subfile,
                              dropCorrelated=dropCorrelated)
    return (results)


def voting_multiclass(ensemble, Xtrain, Xtest, y, n_classes=9, use_proba=False, score_func='log_loss', plotting=True,
                      subfile=None):
    """
    Voting for multi classifiction result
    """
    if use_proba:
        print "Majority voting for predictions using proba"
        voter = np.reshape(Xtrain.values, (Xtrain.shape[0], -1, n_classes)).swapaxes(0, 1)

        for model in voter:
            max_idx = model.argmax(axis=1)
            for row, idx in zip(model, max_idx):
                row[:] = 0.0
                row[idx] = 1.0

        voter = voter.mean(axis=0)
        print voter
        print voter.shape
    else:
        print "Majority voting for predictions"
        # assuming all classes are predicted
        if Xtrain.shape[1] % 2 == 0:
            print "Warning: Even number of voters..."

        classes = np.unique(Xtrain.values)

        votes_train = np.zeros((Xtrain.shape[0], classes.shape[0]))
        votes_test = np.zeros((Xtest.shape[0], classes.shape[0]))

        for i, c in enumerate(classes):
            votes_train[:, i] = np.sum(Xtrain.values == c, axis=1)
            votes_test[:, i] = np.sum(Xtest.values == c, axis=1)

        votes_train = np.argmax(votes_train, axis=1)
        votes_test = np.argmax(votes_test, axis=1)

        encoder = preprocessing.LabelEncoder()
        encoder.fit(y)
        ypred = encoder.inverse_transform(votes_train)
        preds = encoder.inverse_transform(votes_test)

        score = funcdict[score_func](y, ypred)
        print score_func + ": %0.3f" % (score)

    if subfile is not None:
        analyze_predictions(ypred, preds)
        makePredictions(None, Xtest=preds, idx=idx,filename=subfile)

        if plotting:
            plt.hist(ypred, bins=50, alpha=0.3, label='oob')
            plt.hist(preds, bins=50, alpha=0.3, label='pred')
            plt.legend()
            plt.show()

    else:
        return score


def analyze_predictions(ypred, preds):
    # ypred = ypred.astype(int)
    plt.hist(ypred, bins=50, alpha=0.3, label='oob')
    plt.hist(preds, bins=50, alpha=0.3, label='pred')
    plt.legend()
    plt.show()


def preprocess(oobpreds, testset, verbose=False):
    # print "Clipping data  data..."
    # lowerb = 0.41
    # upperb = 6.91
    # oobpreds = oobpreds.clip(lower=lowerb,upper=upperb,axis=0)
    # testset = pd.DataFrame(np.clip(testset.values,lowerb, upperb))#all labels are the same!

    # overfittet models
    noise_columns = []  # ['bagxgb5_br1_Class','nn7_br25_Class']
    print "Adding random noise:", noise_columns
    for col in noise_columns:
        if col in oobpreds.columns:
            oobpreds[col] = oobpreds[col].map(lambda x: x + np.random.normal(loc=0.0, scale=.05))

    if verbose:
        oobpreds.describe()
        showCorrelations(oobpreds)

    return oobpreds, testset


def classicalBlend(ensemble, oobpreds, testset, ly, valpreds=None, yval=None, use_proba=True, score_func='log_loss',
                   subfile=None, cv=5,
                   skipCV=False, **kwargs):
    """
    Blending using sklearn classifier
    """
    showAVGCorrelations(oobpreds, testset)

    if kwargs['dropCorrelated']:
        # showCorrelations(oobpreds)
        oobpreds, testset, valpreds = removeCorrelations(oobpreds, testset,valpreds, 0.995)
        print oobpreds.shape


    blender=Ridge(alpha=1.0)#0.4546/0.4552
    #blender = LinearRegression()
    #blender = Pipeline([('pca', PCA(n_components=19,whiten=False)), ('model', LinearRegression())])
    # blender = Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=1.0))])
    #blender = ConstrainedLinearRegressor(lowerbound=0, upperbound=.2, n_classes=1, alpha=None, corr_penalty=None,normalize=False, loss='rmse', greater_is_better=False)  # 0.216467

    #blender=ExtraTreesRegressor(n_estimators=250,max_depth=None,min_samples_leaf=5,n_jobs=4,criterion='mse', max_features=4*oobpreds.shape[1]/5,oob_score=False)#0.215702
    #blender = Pipeline([('ohc', OneHotEncoder(sparse=False)), ('model',ExtraTreesClassifier(n_estimators=300,max_depth=None,min_samples_leaf=7,n_jobs=4,criterion='gini', max_features=3,oob_score=False))])
    # blender=RandomForestRegressor(n_estimators=100,max_depth=None,min_samples_leaf=10,n_jobs=1,criterion='entropy', max_features=5,oob_score=False)
    #blender = XgboostRegressor(n_estimators=200,learning_rate=0.03,max_depth=2,subsample=.5,n_jobs=4,min_child_weight=1,objective='reg:linear',eval_metric='rmse',booster='gbtree',silent=1)#0.4547 / 0.4560
    #print oobpreds.shape
    model = KerasNN(dims=oobpreds.shape[1],nb_classes=1,nb_epoch=30,learning_rate=0.001,validation_split=0.0,batch_size=256,verbose=1,activation='sigmoid', layers=[32]*2, dropout=[0.0]*2,loss='mse')
    blender = Pipeline([('scaler', StandardScaler()), ('model',model)])#0.206
    blender = BaggingRegressor(base_estimator=blender, n_estimators=20, n_jobs=1, verbose=2, random_state=None,max_samples=1.0, max_features=1.0, bootstrap=False)# + 0.454

    if not skipCV:
        # blender = CalibratedClassifierCV(baseblender, method='sigmoid', cv=3)
        #cv = StratifiedKFold(ly, n_folds=8,shuffle=True)
        pd.Series.from_csv('./data/labels_for_cv.csv')
        cv = LabelKFold(cv_labels, n_folds=8)
        #ForwardDateCV(m.Xtrain.Month,m.Xtrain.Year,n_iter=8,useAll=False,verbose=True)
        #score_func = make_scorer(funcdict[score_func], greater_is_better = False)
        #parameters = {'n_estimators':[200,400],'max_depth':[3],'learning_rate':[0.03],'subsample':[0.5],'colsample_bytree':[0.5],'min_child_weight':[5]}#XGB
        parameters = {'n_estimators':[500],'max_features':[8,10,12],'min_samples_leaf':[5],'criterion':['mse']}#RF
        # parameters = {'max_features':[0.9,0.95,1.0],'max_samples':[0.9,0.95,1.0],'bootstrap':[False,True]}#XGB
        # parameters = {'model__hidden1_num_units': [128],'model__dropout1_p':[0.0],'model__hidden2_num_units': [128],'model__dropout2_p':[0.0],'model__max_epochs':[75],'model__objective_alpha':[0.002]}
        #parameters = {'model__max_epochs':[5,10,15]}
        blender=makeGridSearch(blender,oobpreds,ly,n_jobs=1,refit=False,cv=cv,scoring=score_func,parameters=parameters,random_iter=-1)
        #buildXvalModel(blender,oobpreds,ly,sample_weight=None,class_names=None,refit=True,cv=cv)
        blend_scores = np.zeros(len(cv))
        n_classes = 1
        blend_oob = np.zeros((oobpreds.shape[0], n_classes))
        print blender
        for i, (train, test) in enumerate(cv):
            clf = clone(blender)
            Xtrain = oobpreds.iloc[train]
            Xtest = oobpreds.iloc[test]
            clf.fit(Xtrain.values, ly[train])
            if use_proba:
                t = clf.predict_proba(Xtest)[:,1]
                blend_oob[test] = t.reshape(blend_oob[test].shape)
            else:
                blend_oob[test] = clf.predict(Xtest).reshape(blend_oob[test].shape)
            blend_scores[i] = funcdict[score_func](ly[test], blend_oob[test])
            print "Fold: %3d <%s>: %0.6f ~mean: %6.4f std: %6.4f" % (
                i, score_func, blend_scores[i], blend_scores[:i + 1].mean(), blend_scores[:i + 1].std())

        print " <" + score_func + ">: %0.6f (+/- %0.4f)" % (blend_scores.mean(), blend_scores.std()),
        oob_auc = funcdict[score_func](ly, blend_oob)
        # showMisclass(ly,blend_oob,oobpreds,index=kwargs['cv_labels'])
        print " " + score_func + ": %0.6f" % (oob_auc)

        if subfile is not None:
            print "Make model fit on oob data..."
            blender.fit(oobpreds, ly)
            if valpreds is not None:
                print "Evaluate full model on validation data...",
                if use_proba:
                    y_val_pred = blender.predict_proba(valpreds)[:,1]
                else:
                    y_val_pred = blender.predict(valpreds)
                score = funcdict[score_func](yval, y_val_pred)
                print " " + score_func + ": %0.6f" % (score)

                print "Make model fit on oob & validation data..."
                oobpreds = pd.concat([oobpreds, valpreds], axis=0)
                ly = np.hstack((ly.ravel(), yval.ravel()))
                blender.fit(oobpreds, ly)
                # raw_input()

        if hasattr(blender, 'coef_'):
            print "%-3s %-24s %10s %10s" % ("nr", "model", score_func, "coef")
            for i, model in enumerate(oobpreds.columns):
                coldata = np.asarray(oobpreds.iloc[:, i])
                score = funcdict[score_func](ly, coldata)
                print "%-3d %-24s %10.4f%10.4f" % (i + 1, model.replace("_Class", ""), score, blender.coef_.flatten()[i])
            print "sum coef: %4.4f" % (np.sum(blender.coef_))

        if subfile is not None:
            info_dist(ly, "orig")
            info_dist(blender.predict(oobpreds), "fit")

    if subfile is not None:
        print "Make final ensemble prediction..."
        # blend results
        if use_proba:
            preds = blender.predict_proba(testset)[:,1]
        else:
            preds = blender.predict(testset)
            preds = preds.flatten()

        # print preds
        info_dist(preds, "preds")
        makePredictions(None, preds, idx=idx, filename=subfile)
        analyze_predictions(blend_oob, preds)

    return (blend_scores.mean())


def multiclass_mult(Xtrain, params, n_classes):
    """
    Multiplication rule for multiclass models
    """
    ypred = np.zeros((len(params), Xtrain.shape[0], n_classes))
    for i, p in enumerate(params):
        idx_start = n_classes * i
        idx_end = n_classes * (i + 1)
        ypred[i] = Xtrain.iloc[:, idx_start:idx_end] * p
    ypred = np.mean(ypred, axis=0)
    return ypred


def blend_mult(Xtrain, params, n_classes=None):
    if n_classes < 2:
        return np.dot(Xtrain, params)
    else:
        return multiclass_mult(Xtrain, params, n_classes)


def linearBlend(ensemble, Xtrain, Xtest, y, Xval=None, yval=None, score_func='log_loss', greater_is_better=False,
                use_proba=False,
                normalize=False, removeZeroModels=-1, takeMean=False, alpha=None, subfile=None, plotting=False,
                **kwargs):
    """
    Blending for multiclass systems
    """

    def fopt(params):
        # nxm  * m*1 ->n*1
        if np.isnan(np.sum(params)):
            print "We have NaN here!!"
            score = 0.0
        else:
            ypred = blend_mult(Xtrain, params, n_classes)
            # if not use_proba: ypred = np.round(ypred).astype(int)
            score = funcdict[score_func](y, ypred)
            # regularization
            if alpha is not None:
                penalty = alpha * np.sum(np.square(params))
                #print "orig score:%8.3f" % (score),
                score = score - penalty
                print " - Regularization - alpha: %8.3f penalty: %8.3f regularized score: %8.3f" % (
                    alpha, penalty, score)
            if greater_is_better: score = -1 * score
        return score

    y = np.asarray(y)
    n_models = len(ensemble)
    n_classes = Xtrain.shape[1] / len(ensemble)

    lowerbound = 0.0
    upperbound = 0.5
    #constr = None
    constr = [lambda x, z=i: x[z] - lowerbound for i in range(n_models)]
    #constr2 = [lambda x, z=i: upperbound - x[z] for i in range(n_models)]
    #constr = constr + constr2

    #cons = ({'type': 'ineq', 'fun': [lambda x, z=i: x[z] - lowerbound for i in range(n_models)]},
     #       {'type': 'ineq', 'fun': [lambda x, z=i: upperbound - x[z] for i in range(n_models)]})

    x0 = np.ones((n_models, 1)) / float(n_models)

    if not takeMean:
        xopt = fmin_cobyla(fopt, x0, constr, rhoend=1e-5, maxfun=2000)
        # xopt = minimize(fopt, x0,method='Nelder-Mead')
        # xopt = minimize(fopt, x0,method='COBYLA',constraints=cons)
        print xopt
    # xopt = xopt.x
    else:
        xopt = x0

    # normalize coefficient
    if normalize:
        xopt = xopt / np.sum(xopt)
        print "Normalized coefficients:", xopt

    if np.isnan(np.sum(xopt)):
        print "We have NaN here!!"

    ypred = blend_mult(Xtrain, xopt, n_classes)
    # ymean= blend_mult(Xtrain,x0,n_classes).flatten()
    ymean = np.mean(Xtrain.values, axis=1)
    # ymean=np.median(Xtrain.values,axis=1)

    if takeMean:
        print "Taking the mean/median..."
        ypred = ymean

    # print ymean[:10]
    # if not use_proba:
    #  ymean = np.round(ymean+1E-2).astype(int)
    #  ypred = np.round(ypred+1E-6).astype(int)

    print "ypred:", ypred.sum()
    print "ypred:", ypred
    print "ymean:", ymean.sum()
    print "ymean:", ymean

    score = funcdict[score_func](y, ymean)
    print "->score,mean: %4.4f" % (score)
    oob_score = funcdict[score_func](y, ypred)
    print "->score,opt: %4.4f" % (oob_score)
    if Xval is not None:
        print "Evaluating on validation set..."
        yval_mean = np.mean(Xval.values, axis=1)
        pred_score = funcdict[score_func](yval, yval_mean)
        print "->score,mean: %4.4f" % (pred_score)
        yval_pred = blend_mult(Xval, xopt, n_classes)
        pred_score = funcdict[score_func](yval, yval_pred)
        print "->score,opt: %4.4f" % (pred_score)

    zero_models = []
    print "%4s %-48s %6s %6s" % ("nr", "model", "score", "coeff")
    for i, model in enumerate(ensemble):
        idx_start = n_classes * i
        idx_end = n_classes * (i + 1)
        coldata = np.asarray(Xtrain.iloc[:, idx_start:idx_end])
        score = funcdict[score_func](y, coldata)
        print "%4d %-48s %6.3f %6.3f" % (i + 1, model, score, xopt[i]),
        if xopt[i] < removeZeroModels:
            zero_models.append(model)
        if Xval is not None:
            coldata_val = np.asarray(Xval.iloc[:, idx_start:idx_end])
            score = funcdict[score_func](yval, coldata_val)
            print "(val: %6.3f)" % (score)
        else:
            print ""

    print "##sum coefficients: %4.4f" % (np.sum(xopt))

    if removeZeroModels > 0.0:
        print "Dropping ", len(zero_models), " columns:", zero_models
        Xtrain = Xtrain.drop(zero_models, axis=1)
        Xtest = Xtest.drop(zero_models, axis=1)
        return (Xtrain, Xtest)

    # prediction flatten makes a n-dim row vector from a nx1 column vector...
    if takeMean:
        print "Taking the mean/median for predictions..."
        preds = np.mean(Xtest.values, axis=1)
    else:
        preds = blend_mult(Xtest, xopt, n_classes).flatten()
    # if not use_proba: preds = np.round(preds).astype(int)

    if subfile is not None:
        info_dist(y, "orig")
        info_dist(ypred, "fit")
        info_dist(preds, "pred")
        plt.hist(y,bins=50)
        plt.hist(preds,bins=50)
        plt.show()

        makePredictions(None, Xtest=preds, idx=idx, filename=subfile)
    else:
        if Xval is not None:
            #print "Returning the validation score...!"
            return oob_score
        else:
            return oob_score


def info_dist(y, info):
    print info + "-  max: %4.2f mean: %4.2f median: %4.2f min: %4.2f" % (np.amax(y), y.mean(), np.median(y), np.amin(y))


def selectModels(ensemble, startensemble=[], niter=10, mode='linear', useCols=None):
    """
    Random mode for best model selection
    """
    randBinList = lambda n: [randint(0, 1) for b in range(1, n + 1)]
    auc_list = [0.0]
    ens_list = []
    cols_list = []
    for i in range(niter):
        print "iteration %5d/%5d, current max_score: %6.3f" % (i + 1, niter, max(auc_list))
        actlist = randBinList(len(ensemble))
        actensemble = [x for x in itertools.compress(ensemble, actlist)]
        actensemble = startensemble + actensemble
        print actensemble
        # print actensemble
        score = trainEnsemble(actensemble, mode=mode, useCols=useCols, addMetaFeatures=False, dropCorrelated=False)
        auc_list.append(score)
        ens_list.append(actensemble)
    # cols_list.append(actCols)
    max_score = 0.0
    topens = None
    topcols = None
    for ens, score in zip(ens_list, auc_list):
        print "SCORE: %4.4f" % (score),
        print ens
        if score > max_score:
            maxauc = score
            topens = ens
            # topcols=col
    print "\nTOP ensemble:", topens
    print "TOP score: %4.4f" % (max_score)


def selectModelsGreedy(ensemble, startensemble=[], niter=2, mode='mean', useCols=None, dropCorrelated=False,
                       greater_is_better=False,score_func='rmse', replacement=False):
    """
    Select best models in a greedy forward selection
    """
    topensemble = startensemble
    score_list = []
    ens_list = []
    if greater_is_better:
        bestscore = 0.0
    else:
        bestscore = 1E15
    for i in range(niter):
        if greater_is_better:
            maxscore = 0.0
        else:
            maxscore = 1E15
        topidx = -1
        for j in range(len(ensemble)):
            if not replacement:
                if ensemble[j] not in topensemble:
                    actensemble = topensemble + [ensemble[j]]
                else:
                    continue
            else:
                actensemble = topensemble + [ensemble[j]]

            # score=trainEnsemble(actensemble,mode=mode,useCols=useCols,addMetaFeatures=False,dropCorrelated=dropCorrelated)
            # score=trainEnsemble(actensemble,mode=mode,useCols=None,use_proba=False)
            # score = trainEnsemble(actensemble,mode=mode,score_func='quadratic_weighted_kappa',use_proba=False,subfile=None)
            score = trainEnsemble(actensemble, mode=mode, score_func=score_func, use_proba=False, useCols=None,
                                  subfile=None, dropCorrelated=dropCorrelated)
            print "##(Current top score: %4.4f | overall best score: %4.4f) current score: %4.4f  - " % (
                maxscore, bestscore, score)
            if greater_is_better:
                if score > maxscore:
                    maxscore = score
                    topidx = j
            else:
                if score < maxscore:
                    maxscore = score
                    topidx = j

        # pick best set
        # if not maxscore+>bestscore:
        #    print "Not gain in score anymore, leaving..."
        #    break
        topensemble.append(ensemble[topidx])
        print "TOP score: %4.4f" % (maxscore),
        print " - actual ensemble:", topensemble
        score_list.append(maxscore)
        ens_list.append(list(topensemble))
        if greater_is_better:
            if maxscore > bestscore:
                bestscore = maxscore
        else:
            if maxscore < bestscore:
                bestscore = maxscore

    for ens, score in zip(ens_list, score_list):
        print "SCORE: %4.4f" % (score),
        print ens

    plt.plot(score_list)
    plt.show()
    return topensemble


def blendSubmissions(fileList, coefList):
    """
    Simple blend dataframes from fileList
    """
    pass


if __name__ == "__main__":
    np.random.seed(123)
    USE_PROBA = False
    OBJECTIVE = 'rmse'

    plt.interactive(False)
    global idx,cv_lables
    #store = pd.HDFStore('./data/store_db1.h5')
    #idx = store['test_id']
    idx = pd.read_csv('./data/test.csv', encoding="ISO-8859-1")['id']
    cv_labels = pd.Series.from_csv('./data/labels_for_cv.csv')

    """
    # 1nd LEVEL MODEL BUILDING
    """
    #ensemble = createModels()
    #ensemble = createOOBdata(ensemble, repeats=1, n_folds=8, n_jobs=1, use_proba=USE_PROBA,score_func=OBJECTIVE)  # oob data averaging leads to significant variance reduction

    # createDataSets()
    # saveTrainData(ensemble)

    """
    # 1nd LEVEL ENSEMBLING
    """

    all_models_old = ['xgb0_f8r1','xgb1_f8r1','xgb2_f8r1','xgb2b_f8r1','xgb2c_f8r1','xgb3_f8r1','xgb4_f8r1','xrf1_f8r1','rf1_f8r1','nn1_f8r1','nn1b_f8r1','nn2_f8r1','knn1_f8r1','svr1_f8r1','lr1_f8r1']
    all_models = ['bagxgb1_f8r1','xgb1b_f8r1','xgb0_f8r1','xrf3_f8r1','nn1_f8r1','nn2_f8r1','bagnn1_f8r1','xgb2b_f8r1','xgb3_f8r1','xgb4_f8r1','xgb5_f8r1','xgb6_f8r1','xrf2_f8r1','lr1_f8r1','knn1_f8r1','rf1_f8r1']
    best = ['xgb1b_f8r1', 'bagxgb1_f8r1', 'xgb1b_f8r1', 'xrf2_f8r1']
    models = all_models
    #trainEnsemble(models, mode='classical', score_func=OBJECTIVE, useCols=None, addMetaFeatures=False, use_proba=USE_PROBA,dropCorrelated=False, subfile='./submissions/sub150416a.csv')
    selectModelsGreedy(all_models,startensemble=['xgb1b_f8r1'],niter=10,mode='mean',greater_is_better=False,score_func=OBJECTIVE, replacement = True)

