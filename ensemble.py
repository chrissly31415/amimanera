#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""  
Ensemble helper tools

Chrissly31415
October,September 2014

"""

from FullModel import *
import itertools
from scipy.optimize import fmin,fmin_cobyla
from random import randint
import sys
from sklearn.externals.joblib import Parallel, delayed, logger
from sklearn.base import clone
from soil import *






def createModels():
    ensemble=[]
    ##PLS-MODELLS
    ##Ca RMSE=0.384 
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,onlySpectra=False,deleteSpectra=False,plotting=False,standardize=True,doPCA=None,findPeaks=None,makeDerivative=None,featureFilter=None,loadFeatures=None,deleteFeatures=None,removeVar=False,removeCor=None,useSavitzkyGolay=False,addNoiseColumns=None,addLandscapes=False)
    #model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=100,mode='percentile')), ('model', PLSRegression(n_components=20))])
    #xmodel = XModel("pls1_Ca",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['Ca']])
    #ensemble.append(xmodel)
    
    ##P RMSE=0.886
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,onlySpectra=False,deleteSpectra=False,plotting=False,standardize=True,doPCA=None,findPeaks=None,makeDerivative=None,featureFilter=None,loadFeatures=None,deleteFeatures=None,removeVar=False,removeCor=None,useSavitzkyGolay=False,addNoiseColumns=None,addLandscapes=False)
    #model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=98,mode='percentile')), ('model', PLSRegression(n_components=20))])
    #xmodel = XModel("pls1_P",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['P']])
    #ensemble.append(xmodel)
    
    ##pH RMSE=0.346
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,onlySpectra=False,deleteSpectra=False,plotting=False,standardize=True,doPCA=None,findPeaks=None,makeDerivative=None,featureFilter=None,loadFeatures=None,deleteFeatures=None,removeVar=False,removeCor=None,useSavitzkyGolay=False,addNoiseColumns=None,addLandscapes=False)
    #model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=98,mode='percentile')), ('model', PLSRegression(n_components=50))])
    #xmodel = XModel("pls1_pH",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['pH']])
    #ensemble.append(xmodel)
    
    ##SOC RMSE =0.348 
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,onlySpectra=False,deleteSpectra=False,plotting=False,standardize=True,doPCA=None,findPeaks=None,makeDerivative=None,featureFilter=None,loadFeatures=None,deleteFeatures=None,removeVar=False,removeCor=None,useSavitzkyGolay=False,addNoiseColumns=None,addLandscapes=False)
    #model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=100,mode='percentile')), ('model', PLSRegression(n_components=40))])
    #xmodel = XModel("pls1_SOC",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['SOC']])
    #ensemble.append(xmodel)
    
    ##Sand RMSE=0.356
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,onlySpectra=False,deleteSpectra=False,plotting=False,standardize=True,doPCA=None,findPeaks=None,makeDerivative=None,featureFilter=None,loadFeatures=None,deleteFeatures=None,removeVar=False,removeCor=None,useSavitzkyGolay=False,addNoiseColumns=None,addLandscapes=False)
    #model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=99,mode='percentile')), ('model', PLSRegression(n_components=40))])
    #xmodel = XModel("pls1_Sand",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['Sand']])
    #ensemble.append(xmodel)
    
    ##SVM_MODELLS
    ##Ca RMSE=0.287
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,onlySpectra=False,deleteSpectra=False,plotting=False,standardize=True,doPCA=None,findPeaks=None,makeDerivative=None,featureFilter=None,loadFeatures=None,deleteFeatures=None,removeVar=False,removeCor=None,useSavitzkyGolay=False,addNoiseColumns=None,addLandscapes=False)
    #model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=99,mode='percentile')), ('model', SVR(C=100.0, gamma=0.005, verbose = 0))])
    #xmodel = XModel("svm1_Ca",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['Ca']])
    #ensemble.append(xmodel)
    
    ##P RMSE=0.886
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,onlySpectra=False,deleteSpectra=False,plotting=False,standardize=True,doPCA=None,findPeaks=None,makeDerivative=None,featureFilter=None,loadFeatures=None,deleteFeatures=None,removeVar=False,removeCor=None,useSavitzkyGolay=False,addNoiseColumns=None,addLandscapes=False)
    #model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=100,mode='percentile')), ('model', SVR(C=100000.0, gamma=0.0005, verbose = 0))]) 
    #xmodel = XModel("svm1_P",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['P']])
    #ensemble.append(xmodel)
    
    ##pH RMSE=0.321
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,onlySpectra=False,deleteSpectra=False,plotting=False,standardize=True,doPCA=None,findPeaks=None,makeDerivative=None,featureFilter=None,loadFeatures=None,deleteFeatures=None,removeVar=False,removeCor=None,useSavitzkyGolay=False,addNoiseColumns=None,addLandscapes=False)
    #model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=95,mode='percentile')), ('model', SVR(C=10000.0, gamma=0.0005, verbose = 0))])
    #xmodel = XModel("svm1_pH",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['pH']])
    #ensemble.append(xmodel)
    
    ##SOC RMSE=0.278
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,onlySpectra=False,deleteSpectra=False,plotting=False,standardize=True,doPCA=None,findPeaks=None,makeDerivative=None,featureFilter=None,loadFeatures=None,deleteFeatures=None,removeVar=False,removeCor=None,useSavitzkyGolay=False,addNoiseColumns=None,addLandscapes=False)
    #model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=90,mode='percentile')), ('model', SVR(C=10000.0, gamma=0.0005, verbose = 0))])
    #xmodel = XModel("svm1_SOC",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['SOC']])
    #ensemble.append(xmodel)
    
    ##Sand RMSE=0.316
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,onlySpectra=False,deleteSpectra=False,plotting=False,standardize=True,doPCA=None,findPeaks=None,makeDerivative=None,featureFilter=None,loadFeatures=None,deleteFeatures=None,removeVar=False,removeCor=None,useSavitzkyGolay=False,addNoiseColumns=None,addLandscapes=False)
    #model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=99,mode='percentile')), ('model', SVR(C=10000.0, gamma=0.0005, verbose = 0))])
    #xmodel = XModel("svm1_Sand",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['Sand']])
    #ensemble.append(xmodel)
    
    #PCA RidgCV models
    #Ca RMSE=0.410
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,onlySpectra=False,deleteSpectra=False,plotting=False,standardize=True,doPCA=None,findPeaks=None,makeDerivative=None,featureFilter=None,loadFeatures=None,deleteFeatures=None,removeVar=False,removeCor=None,useSavitzkyGolay=False,addNoiseColumns=None,addLandscapes=False)
    #model = Pipeline([('pca', PCA(n_components=40)),('model', RidgeCV())])
    #xmodel = XModel("pcaridge1_Ca",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['Ca']])
    #ensemble.append(xmodel)
    
    ##P RMSE=0.862
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,onlySpectra=False,deleteSpectra=False,plotting=False,standardize=True,doPCA=None,findPeaks=None,makeDerivative=None,featureFilter=None,loadFeatures=None,deleteFeatures=None,removeVar=False,removeCor=None,useSavitzkyGolay=False,addNoiseColumns=None,addLandscapes=False)
    #model = Pipeline([('pca', PCA(n_components=200)),('model', RidgeCV())])
    #xmodel = XModel("pcaridge1_P",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['P']])
    #ensemble.append(xmodel)
    
    ##pH RMSE=0.362
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,onlySpectra=False,deleteSpectra=False,plotting=False,standardize=True,doPCA=None,findPeaks=None,makeDerivative=None,featureFilter=None,loadFeatures=None,deleteFeatures=None,removeVar=False,removeCor=None,useSavitzkyGolay=False,addNoiseColumns=None,addLandscapes=False)
    #model = Pipeline([('pca', PCA(n_components=200)),('model', RidgeCV())])
    #xmodel = XModel("pcaridge1_pH",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['pH']])
    #ensemble.append(xmodel)
    
    ##SOC RMSE=0.347
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,onlySpectra=False,deleteSpectra=False,plotting=False,standardize=True,doPCA=None,findPeaks=None,makeDerivative=None,featureFilter=None,loadFeatures=None,deleteFeatures=None,removeVar=False,removeCor=None,useSavitzkyGolay=False,addNoiseColumns=None,addLandscapes=False)
    #model = Pipeline([('pca', PCA(n_components=200)),('model', RidgeCV())])
    #xmodel = XModel("pcaridge1_SOC",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['SOC']])
    #ensemble.append(xmodel)
    
    ##Sand RMSE=0.347
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,onlySpectra=False,deleteSpectra=False,plotting=False,standardize=True,doPCA=None,findPeaks=None,makeDerivative=None,featureFilter=None,loadFeatures=None,deleteFeatures=None,removeVar=False,removeCor=None,useSavitzkyGolay=False,addNoiseColumns=None,addLandscapes=False)
    #model = Pipeline([('pca', PCA(n_components=200)),('model', RidgeCV())])
    #xmodel = XModel("pcaridge1_Sand",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['Sand']])
    #ensemble.append(xmodel)
    
    #leaps selected lin reg
    ##Ca RMSE=
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,featureFilter=getFeatures('Ca'))
    #model = LinearRegression()
    #xmodel = XModel("leaps1_Ca",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['Ca']])
    #ensemble.append(xmodel)
    
    ##P RMSE=
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,featureFilter=getFeatures('P'))
    #model = LinearRegression()
    #xmodel = XModel("leaps1_P",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['P']])
    #ensemble.append(xmodel)
    
    ##pH RMSE=
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,featureFilter=getFeatures('pH'))
    #model = LinearRegression()
    #xmodel = XModel("leaps1_pH",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['pH']])
    #ensemble.append(xmodel)
    
    ##SOC RMSE=
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,featureFilter=getFeatures('SOC'))
    #model = LinearRegression()
    #xmodel = XModel("leaps1_SOC",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['SOC']])
    #ensemble.append(xmodel)
    
    ##Sand RMSE=
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,featureFilter=getFeatures('Sand'))
    #model = LinearRegression()
    #xmodel = XModel("leaps1_Sand",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['Sand']])
    #ensemble.append(xmodel)
    
    #leaps selected RF
    #Ca RMSE=
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,featureFilter=getFeatures('Ca'))
    #model = RandomForestRegressor(n_estimators=500,max_depth=None,min_samples_leaf=1, max_features='auto')
    #xmodel = XModel("rf1_Ca",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['Ca']])
    #ensemble.append(xmodel)
    
    #P RMSE=
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,featureFilter=getFeatures('P'))
    #model = RandomForestRegressor(n_estimators=500,max_depth=None,min_samples_leaf=1, max_features='auto')
    #xmodel = XModel("rf1_P",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['P']])
    #ensemble.append(xmodel)
    
    #pH RMSE=
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,featureFilter=getFeatures('pH'))
    #model = RandomForestRegressor(n_estimators=500,max_depth=None,min_samples_leaf=1, max_features='auto')
    #xmodel = XModel("rf1_pH",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['pH']])
    #ensemble.append(xmodel)
    
    #SOC RMSE=
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,featureFilter=getFeatures('SOC'))
    #model = RandomForestRegressor(n_estimators=500,max_depth=None,min_samples_leaf=1, max_features='auto')
    #xmodel = XModel("rf1_SOC",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['SOC']])
    #ensemble.append(xmodel)
    
    #Sand RMSE=
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,featureFilter=getFeatures('Sand'))
    #model = RandomForestRegressor(n_estimators=500,max_depth=None,min_samples_leaf=1, max_features='auto')
    #xmodel = XModel("rf1_Sand",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['Sand']])
    #ensemble.append(xmodel)
    
    #PLS models 
    ##Ca RMSE=
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,onlySpectra=False,deleteSpectra=False,plotting=False,standardize=True,doPCA=None,findPeaks=None,makeDerivative=None,featureFilter=None,loadFeatures=None,deleteFeatures=None,removeVar=False,removeCor=None,useSavitzkyGolay=False,addNoiseColumns=None,addLandscapes=False)
    #model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=98,mode='percentile')), ('model', PLSRegression(n_components=20))])
    #xmodel = XModel("pls2_Ca",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['Ca']])
    #ensemble.append(xmodel)
    
    ##P RMSE=
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,onlySpectra=False,deleteSpectra=False,plotting=False,standardize=True,doPCA=None,findPeaks=None,makeDerivative=None,featureFilter=None,loadFeatures=None,deleteFeatures=None,removeVar=False,removeCor=None,useSavitzkyGolay=False,addNoiseColumns=None,addLandscapes=False)
    #model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=80,mode='percentile')), ('model', PLSRegression(n_components=5))])
    #xmodel = XModel("pls2_P",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['P']])
    #ensemble.append(xmodel)
    
    ##pH RMSE=
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,onlySpectra=False,deleteSpectra=False,plotting=False,standardize=True,doPCA=None,findPeaks=None,makeDerivative=None,featureFilter=None,loadFeatures=None,deleteFeatures=None,removeVar=False,removeCor=None,useSavitzkyGolay=False,addNoiseColumns=None,addLandscapes=False)
    #model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=100,mode='percentile')), ('model', PLSRegression(n_components=50))])
    #xmodel = XModel("pls2_pH",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['pH']])
    #ensemble.append(xmodel)
    
    ##SOC RMSE = 
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,onlySpectra=False,deleteSpectra=False,plotting=False,standardize=True,doPCA=None,findPeaks=None,makeDerivative=None,featureFilter=None,loadFeatures=None,deleteFeatures=None,removeVar=False,removeCor=None,useSavitzkyGolay=False,addNoiseColumns=None,addLandscapes=False)
    #model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=90,mode='percentile')), ('model', PLSRegression(n_components=25))])
    #xmodel = XModel("pls2_SOC",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['SOC']])
    #ensemble.append(xmodel)
    
    ##Sand RMSE=
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,onlySpectra=False,deleteSpectra=False,plotting=False,standardize=True,doPCA=None,findPeaks=None,makeDerivative=None,featureFilter=None,loadFeatures=None,deleteFeatures=None,removeVar=False,removeCor=None,useSavitzkyGolay=False,addNoiseColumns=None,addLandscapes=False)
    #model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=98,mode='percentile')), ('model', PLSRegression(n_components=20))])
    #xmodel = XModel("pls2_Sand",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['Sand']])
    #ensemble.append(xmodel)
    
    #ridge model (with 2nd derivate)
    #Ca RMSE=
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,standardize=True,makeDerivative='2nd',deleteFeatures=getFeatures('co2'),compressIR=150)
    #model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=98,mode='percentile')), ('model', Ridge(alpha=1))])
    #xmodel = XModel("ridge2_Ca",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['Ca']])
    #ensemble.append(xmodel)
    
    ##P RMSE= 
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,standardize=True,makeDerivative='2nd',deleteFeatures=getFeatures('co2'),compressIR=150)
    #model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=80,mode='percentile')), ('model', Ridge(alpha=100))])
    #xmodel = XModel("ridge2_P",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['P']])
    #ensemble.append(xmodel)
    
    ##pH RMSE= 
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,standardize=True,makeDerivative='2nd',deleteFeatures=getFeatures('co2'),compressIR=150)
    #model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=50,mode='percentile')), ('model', Ridge(alpha=10))])
    #xmodel = XModel("ridge2_pH",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['pH']])
    #ensemble.append(xmodel)
    
    ##SOC RMSE = 
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,standardize=True,makeDerivative='2nd',deleteFeatures=getFeatures('co2'),compressIR=150)
    #model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=80,mode='percentile')), ('model', Ridge(alpha=1))])
    #xmodel = XModel("ridge2_SOC",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['SOC']])
    #ensemble.append(xmodel)
    
    ##Sand RMSE=
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,standardize=True,makeDerivative='2nd',deleteFeatures=getFeatures('co2'),compressIR=150)
    #model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=50,mode='percentile')), ('model', Ridge(alpha=1))])
    #xmodel = XModel("ridge2_Sand",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['Sand']])
    #ensemble.append(xmodel)
    
    #leaps selected lin reg
    #Ca RMSE=
    (X,Xtest,ymat) = prepareDatasets(nsamples=-1,standardize=True,featureFilter=getFeatures('Ca_greedy'),compressIR=150,deleteFeatures=getFeatures('co2'))
    model = LinearRegression()
    xmodel = XModel("leaps2_Ca",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['Ca']])
    ensemble.append(xmodel)
    
    #P RMSE=
    (X,Xtest,ymat) = prepareDatasets(nsamples=-1,standardize=True,featureFilter=getFeatures('P_greedy'),compressIR=150,deleteFeatures=getFeatures('co2'))
    model = LinearRegression()
    xmodel = XModel("leaps2_P",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['P']])
    ensemble.append(xmodel)
    
    #pH RMSE=
    (X,Xtest,ymat) = prepareDatasets(nsamples=-1,standardize=True,featureFilter=getFeatures('pH_greedy'),compressIR=150,deleteFeatures=getFeatures('co2'))
    model = LinearRegression()
    xmodel = XModel("leaps2_pH",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['pH']])
    ensemble.append(xmodel)
    
    #SOC RMSE=
    (X,Xtest,ymat) = prepareDatasets(nsamples=-1,standardize=True,featureFilter=getFeatures('SOC_greedy'),compressIR=150,deleteFeatures=getFeatures('co2'))
    model = LinearRegression()
    xmodel = XModel("leaps2_SOC",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['SOC']])
    ensemble.append(xmodel)
    
    #Sand RMSE=
    (X,Xtest,ymat) = prepareDatasets(nsamples=-1,standardize=True,featureFilter=getFeatures('Sand_greedy'),compressIR=150,deleteFeatures=getFeatures('co2'))
    model = LinearRegression()
    xmodel = XModel("leaps2_Sand",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['Sand']])
    ensemble.append(xmodel)
    
    #ridge model (with 1st derivate)
    #Ca RMSE0.339 (+/- 0.303)
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,onlySpectra=False,deleteSpectra=False,plotting=False,standardize=True,doPCA=None,findPeaks=None,makeDerivative='1st',featureFilter=None,loadFeatures=None,deleteFeatures=None,removeVar=0.1,removeCor=None,useSavitzkyGolay=False,addNoiseColumns=None,addLandscapes=False)
    #model = Pipeline([('pca', PCA(n_components=100)), ('model', Ridge(alpha=0.1))])
    #xmodel = XModel("pcridge2_Ca",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['Ca']])
    #ensemble.append(xmodel)
    
    ##P RMSE0.743 (+/- 0.645)
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,onlySpectra=False,deleteSpectra=False,plotting=False,standardize=True,doPCA=None,findPeaks=None,makeDerivative='1st',featureFilter=None,loadFeatures=None,deleteFeatures=None,removeVar=0.1,removeCor=None,useSavitzkyGolay=False,addNoiseColumns=None,addLandscapes=False)
    #model = Pipeline([('pca', PCA(n_components=100)), ('model', Ridge(alpha=100))])
    #xmodel = XModel("pcridge2_P",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['P']])
    #ensemble.append(xmodel)
    
    ##pH RMSE0.474 (+/- 0.182)
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,onlySpectra=False,deleteSpectra=False,plotting=False,standardize=True,doPCA=None,findPeaks=None,makeDerivative='1st',featureFilter=None,loadFeatures=None,deleteFeatures=None,removeVar=0.1,removeCor=None,useSavitzkyGolay=False,addNoiseColumns=None,addLandscapes=False)
    #model = Pipeline([('pca', PCA(n_components=200)), ('model', Ridge(alpha=100))])
    #xmodel = XModel("pcridge2_pH",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['pH']])
    #ensemble.append(xmodel)
    
    ##SOC RMSE 0.466 (+/- 0.308)
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,onlySpectra=False,deleteSpectra=False,plotting=False,standardize=True,doPCA=None,findPeaks=None,makeDerivative='1st',featureFilter=None,loadFeatures=None,deleteFeatures=None,removeVar=0.1,removeCor=None,useSavitzkyGolay=False,addNoiseColumns=None,addLandscapes=False)
    #model = Pipeline([('pca', PCA(n_components=150)), ('model', Ridge(alpha=0.1))])
    #xmodel = XModel("pcridge2_SOC",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['SOC']])
    #ensemble.append(xmodel)
    
    ##Sand RMSE0.456 (+/- 0.260)
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,onlySpectra=False,deleteSpectra=False,plotting=False,standardize=True,doPCA=None,findPeaks=None,makeDerivative='1st',featureFilter=None,loadFeatures=None,deleteFeatures=None,removeVar=0.1,removeCor=None,useSavitzkyGolay=False,addNoiseColumns=None,addLandscapes=False)
    #model = Pipeline([('pca', PCA(n_components=150)), ('model', Ridge(alpha=10))])
    #xmodel = XModel("pcridge2_Sand",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['Sand']])
    #ensemble.append(xmodel)
    
    #gbm models for non linear effects
    #Ca RMSE=<score>= 0.414 (+/- 0.526)
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,standardize=True,deleteFeatures=getFeatures('co2'),removeVar=0.1,compressIR=75)
    #model = GradientBoostingRegressor(loss='huber',n_estimators=300, learning_rate=0.1, max_depth=2,subsample=1.0)
    #xmodel = XModel("gbm2_Ca",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['Ca']])
    #ensemble.append(xmodel)
    
    ##P RMSE <score>= 0.682 (+/- 0.741)
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,standardize=True,deleteFeatures=getFeatures('co2'),removeVar=0.1,compressIR=75)
    #model = GradientBoostingRegressor(loss='huber',n_estimators=50, learning_rate=0.1, max_depth=2,subsample=1.0)
    #xmodel = XModel("gbm2_P",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['P']])
    #ensemble.append(xmodel)
    
    ##pH RMSE=0.674 (+/- 0.295)
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,standardize=True,deleteFeatures=getFeatures('co2'),removeVar=0.1,compressIR=75)
    #model = GradientBoostingRegressor(loss='huber',n_estimators=500, learning_rate=0.1, max_depth=2,subsample=1.0)
    #xmodel = XModel("gbm2_pH",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['pH']])
    #ensemble.append(xmodel)
    
    ##SOC RMSE =0.427 (+/- 0.399)
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,standardize=True,deleteFeatures=getFeatures('co2'),removeVar=0.1,compressIR=75)
    #model = GradientBoostingRegressor(loss='huber',n_estimators=500, learning_rate=0.1, max_depth=2,subsample=1.0)
    #xmodel = XModel("gbm2_SOC",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['SOC']])
    #ensemble.append(xmodel)
    
    ##Sand RMSE)= 0.585 (+/- 0.292)
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,standardize=True,deleteFeatures=getFeatures('co2'),removeVar=0.1,compressIR=75)
    #model = GradientBoostingRegressor(loss='huber',n_estimators=50, learning_rate=0.1, max_depth=2,subsample=1.0)
    #xmodel = XModel("gbm2_Sand",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['Sand']])
    #ensemble.append(xmodel)
    
    ##ridge model using compressed fature + savitzky
    ##Ca RMSE= 0.287 (+/- 0.290)
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,standardize=True,deleteFeatures=getFeatures('co2')+['DEPTH','BSAN','BSAS','BSAV','CTI','EVI','LSTD','REF1','REF3','REF7','RELI'],removeVar=0.1,useSavitzkyGolay=True,compressIR=150)
    #model =  Pipeline([('filter', GenericUnivariateSelect(f_regression, param=90,mode='percentile')), ('model', Ridge(alpha=0.1))])
    #xmodel = XModel("compridge2_Ca",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['Ca']])
    #ensemble.append(xmodel)
    
    ##P RMSE =  0.700 (+/- 0.678)
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,standardize=True,deleteFeatures=getFeatures('co2')+['BSAS','BSAV','CTI','ELEV','EVI','LSTN','REF1','REF3','DEPTH'],removeVar=0.1,useSavitzkyGolay=True,compressIR=150)
    #model =  Pipeline([('filter', GenericUnivariateSelect(f_regression, param=80,mode='percentile')), ('model', Ridge(alpha=10))])
    #xmodel = XModel("compridge2_P",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['P']])
    #ensemble.append(xmodel)
    
    ##pH RMSE = 0.457 (+/- 0.162)
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,standardize=True,deleteFeatures=getFeatures('co2')+['BSAN','BSAV','CTI','REF1','REF2','RELI','DEPTH'],removeVar=0.1,useSavitzkyGolay=True,compressIR=150)
    #model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=98,mode='percentile')), ('model', Ridge(alpha=0.1))])
    #xmodel = XModel("compridge2_pH",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['pH']])
    #ensemble.append(xmodel)
    
    ##SOC RMSE =0.443 (+/- 0.397)
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,standardize=True,deleteFeatures=getFeatures('co2')+['BSAS','CTI','EVI','LSTN','REF7','TMAP','DEPTH'],removeVar=0.1,useSavitzkyGolay=True,compressIR=150)
    #model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=90,mode='percentile')), ('model', Ridge(alpha=0.001))])
    #xmodel = XModel("compridge2_SOC",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['SOC']])
    #ensemble.append(xmodel)
    
    ##Sand RMSE = 0.449 (+/- 0.258)
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,standardize=True,deleteFeatures=getFeatures('co2')+['BSAN','BSAS','REF2','REF3','RELI','TMFI','DEPTH'],removeVar=0.1,useSavitzkyGolay=True,compressIR=150)
    #model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=100,mode='percentile')), ('model', Ridge(alpha=0.1))])
    #xmodel = XModel("compridge2_Sand",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['Sand']])
    #ensemble.append(xmodel)
    
    #SVM_MODELLS part 3
    #Ca RMSE=0.226 (+/- 0.283)
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,onlySpectra=False,deleteFeatures=getFeatures('co2')+getFeatures('non-spectra'),removeVar=False,useSavitzkyGolay=True,compressIR=200)
    #model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=99,mode='percentile')), ('model', SVR(C=100000.0, gamma=0.001, verbose = 0))])
    #xmodel = XModel("svm3_Ca",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['Ca']])
    #ensemble.append(xmodel)
    
    ##P RMSE=0.657 (+/- 0.731)
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,onlySpectra=False,deleteFeatures=getFeatures('co2')+getFeatures('non-spectra'),removeVar=False,useSavitzkyGolay=True,compressIR=200)
    #model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=99,mode='percentile')), ('model', SVR(C=100000.0, gamma=0.0001, verbose = 0))]) 
    #xmodel = XModel("svm3_P",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['P']])
    #ensemble.append(xmodel)
    
    ##pH RMSE=0.468 (+/- 0.165)
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,onlySpectra=False,deleteFeatures=getFeatures('co2')+getFeatures('non-spectra'),removeVar=False,useSavitzkyGolay=True,compressIR=200)
    #model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=99,mode='percentile')), ('model', SVR(C=100.0, gamma=0.0021544346900318821, verbose = 0))])
    #xmodel = XModel("svm3_pH",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['pH']])
    #ensemble.append(xmodel)
    
    ##SOC RMSE=0.372 (+/- 0.388)
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,onlySpectra=False,deleteFeatures=getFeatures('co2')+getFeatures('non-spectra'),removeVar=False,useSavitzkyGolay=True,compressIR=200)
    #model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=99,mode='percentile')), ('model', SVR(C=10000.0, gamma=0.0021544346900318821, verbose = 0))])
    #xmodel = XModel("svm3_SOC",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['SOC']])
    #ensemble.append(xmodel)
    
    ##Sand RMSE=0.422 (+/- 0.247)
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,onlySpectra=False,deleteFeatures=getFeatures('co2')+getFeatures('non-spectra'),removeVar=False,useSavitzkyGolay=True,compressIR=200)
    #model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=99,mode='percentile')), ('model', SVR(C=100000.0, gamma=0.0001, verbose = 0))])
    #xmodel = XModel("svm3_Sand",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['Sand']])
    #ensemble.append(xmodel)
    
    
    
    #SVM_MODELLS
    ##Ca RMSE=
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,onlySpectra=False,deleteSpectra=False,plotting=False,standardize=True,doPCA=None,findPeaks=None,makeDerivative=None,featureFilter=None,loadFeatures=None,deleteFeatures=None,removeVar=False,removeCor=None,useSavitzkyGolay=False,addNoiseColumns=None,addLandscapes=False)
    #model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=99,mode='percentile')), ('model', SVR(C=10000.0, gamma=0.005, verbose = 0))])
    #xmodel = XModel("svm2_Ca",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['Ca']])
    #ensemble.append(xmodel)
    
    ##P RMSE=
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,onlySpectra=False,deleteSpectra=False,plotting=False,standardize=True,doPCA=None,findPeaks=None,makeDerivative=None,featureFilter=None,loadFeatures=None,deleteFeatures=None,removeVar=False,removeCor=None,useSavitzkyGolay=False,addNoiseColumns=None,addLandscapes=False)
    #model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=99,mode='percentile')), ('model', SVR(C=1000.0, gamma=0.0005, verbose = 0))]) 
    #xmodel = XModel("svm2_P",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['P']])
    #ensemble.append(xmodel)
    
    ##pH RMSE=
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,onlySpectra=False,deleteSpectra=False,plotting=False,standardize=True,doPCA=None,findPeaks=None,makeDerivative=None,featureFilter=None,loadFeatures=None,deleteFeatures=None,removeVar=False,removeCor=None,useSavitzkyGolay=False,addNoiseColumns=None,addLandscapes=False)
    #model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=99,mode='percentile')), ('model', SVR(C=10000.0, gamma=0.0005, verbose = 0))])
    #xmodel = XModel("svm2_pH",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['pH']])
    #ensemble.append(xmodel)
    
    ##SOC RMSE=
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,onlySpectra=False,deleteSpectra=False,plotting=False,standardize=True,doPCA=None,findPeaks=None,makeDerivative=None,featureFilter=None,loadFeatures=None,deleteFeatures=None,removeVar=False,removeCor=None,useSavitzkyGolay=False,addNoiseColumns=None,addLandscapes=False)
    #model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=99,mode='percentile')), ('model', SVR(C=10000.0, gamma=0.0005, verbose = 0))])
    #xmodel = XModel("svm2_SOC",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['SOC']])
    #ensemble.append(xmodel)
    
    ##Sand RMSE=
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,onlySpectra=False,deleteSpectra=False,plotting=False,standardize=True,doPCA=None,findPeaks=None,makeDerivative=None,featureFilter=None,loadFeatures=None,deleteFeatures=None,removeVar=False,removeCor=None,useSavitzkyGolay=False,addNoiseColumns=None,addLandscapes=False)
    #model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=99,mode='percentile')), ('model', SVR(C=100.0, gamma=0.0005, verbose = 0))])
    #xmodel = XModel("svm2_Sand",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['Sand']])
    #ensemble.append(xmodel)
    
    #elastic net models
    #Ca RMSE=0.301 (+/- 0.287) 
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,onlySpectra=False,deleteSpectra=False,plotting=False,standardize=True,doPCA=None,findPeaks=None,makeDerivative=None,featureFilter=None,loadFeatures=None,deleteFeatures=None,removeVar=False,removeCor=None,useSavitzkyGolay=False,addNoiseColumns=None,addLandscapes=False)
    #model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=99,mode='percentile')),('model', ElasticNet(alpha=.01,l1_ratio=0.001,max_iter=1000)) ])
    #xmodel = XModel("elnet2_Ca",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['Ca']])
    #ensemble.append(xmodel)
    
    ##P RMSE=0.754 (+/- 0.643)
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,onlySpectra=False,deleteSpectra=False,plotting=False,standardize=True,doPCA=None,findPeaks=None,makeDerivative=None,featureFilter=None,loadFeatures=None,deleteFeatures=None,removeVar=False,removeCor=None,useSavitzkyGolay=False,addNoiseColumns=None,addLandscapes=False)
    #model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=99,mode='percentile')),('model', ElasticNet(alpha=.01,l1_ratio=0.001,max_iter=1000)) ])
    #xmodel = XModel("elnet2_P",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['P']])
    #ensemble.append(xmodel)
    
    ##pH RMSE=0.453 (+/- 0.166)
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,onlySpectra=False,deleteSpectra=False,plotting=False,standardize=True,doPCA=None,findPeaks=None,makeDerivative=None,featureFilter=None,loadFeatures=None,deleteFeatures=None,removeVar=False,removeCor=None,useSavitzkyGolay=False,addNoiseColumns=None,addLandscapes=False)
    #model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=99,mode='percentile')),('model', ElasticNet(alpha=.01,l1_ratio=0.001,max_iter=1000))])
    #xmodel = XModel("elnet2_pH",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['pH']])
    #ensemble.append(xmodel)
    
    ##SOC RMSE =0.461 (+/- 0.362) 
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,onlySpectra=False,deleteSpectra=False,plotting=False,standardize=True,doPCA=None,findPeaks=None,makeDerivative=None,featureFilter=None,loadFeatures=None,deleteFeatures=None,removeVar=False,removeCor=None,useSavitzkyGolay=False,addNoiseColumns=None,addLandscapes=False)
    #model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=99,mode='percentile')), ('model', ElasticNet(alpha=.001,l1_ratio=0.001,max_iter=1000))])
    #xmodel = XModel("elnet2_SOC",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['SOC']])
    #ensemble.append(xmodel)
    
    ##Sand RMSE=0.445 (+/- 0.229) 
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,onlySpectra=False,deleteSpectra=False,plotting=False,standardize=True,doPCA=None,findPeaks=None,makeDerivative=None,featureFilter=None,loadFeatures=None,deleteFeatures=None,removeVar=False,removeCor=None,useSavitzkyGolay=False,addNoiseColumns=None,addLandscapes=False)
    #model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=99,mode='percentile')), ('model', ElasticNet(alpha=.001,l1_ratio=0.001,max_iter=1000))])
    #xmodel = XModel("elnet2_Sand",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['Sand']])
    #ensemble.append(xmodel)
    
    #pcr regression
    #Ca RMSE=0.344 (+/- 0.180) 9folds 0.306 (+/- 0.282) 37 folds
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,onlySpectra=False,deleteSpectra=False,plotting=False,standardize=True,doPCA=None,findPeaks=None,makeDerivative=None,featureFilter=None,loadFeatures=None,deleteFeatures=None,removeVar=False,removeCor=None,useSavitzkyGolay=False,addNoiseColumns=None,addLandscapes=False)
    #model = Pipeline([('pca', PCA(n_components=40)),('filter', GenericUnivariateSelect(f_regression, param=98,mode='percentile')), ('model', LinearRegression())])
    #xmodel = XModel("pcreg2_Ca",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['Ca']])
    #ensemble.append(xmodel)
    
    ##P RMSE=0.884 (+/- 0.426) 0.708 (+/- 0.681)
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,onlySpectra=False,deleteSpectra=False,plotting=False,standardize=True,doPCA=None,findPeaks=None,makeDerivative=None,featureFilter=None,loadFeatures=None,deleteFeatures=None,removeVar=False,removeCor=None,useSavitzkyGolay=False,addNoiseColumns=None,addLandscapes=False)
    #model =  Pipeline([('pca', PCA(n_components=5)),('filter', GenericUnivariateSelect(f_regression, param=100,mode='percentile')), ('model', LinearRegression())])
    #xmodel = XModel("pcreg2_P",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['P']])
    #ensemble.append(xmodel)
    
    ##pH RMSE=<score>= 0.474 (+/- 0.097) 0.471 (+/- 0.163)
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,onlySpectra=False,deleteSpectra=False,plotting=False,standardize=True,doPCA=None,findPeaks=None,makeDerivative=None,featureFilter=None,loadFeatures=None,deleteFeatures=None,removeVar=False,removeCor=None,useSavitzkyGolay=False,addNoiseColumns=None,addLandscapes=False)
    #model = Pipeline([('pca', PCA(n_components=30)),('filter', GenericUnivariateSelect(f_regression, param=98,mode='percentile')), ('model', LinearRegression())])
    #xmodel = XModel("pcreg2_pH",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['pH']])
    #ensemble.append(xmodel)
    
    ##SOC RMSE = <score>= 0.563 (+/- 0.231) 0.483 (+/- 0.332)
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,onlySpectra=False,deleteSpectra=False,plotting=False,standardize=True,doPCA=None,findPeaks=None,makeDerivative=None,featureFilter=None,loadFeatures=None,deleteFeatures=None,removeVar=False,removeCor=None,useSavitzkyGolay=False,addNoiseColumns=None,addLandscapes=False)
    #model = Pipeline([('pca', PCA(n_components=50)),('filter', GenericUnivariateSelect(f_regression, param=98,mode='percentile')), ('model', LinearRegression())])
    #xmodel = XModel("pcreg2_SOC",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['SOC']])
    #ensemble.append(xmodel)
    
    ##Sand RMSE=0.480 (+/- 0.108) 0.425 (+/- 0.239
    #(X,Xtest,ymat) = prepareDatasets(nsamples=-1,onlySpectra=False,deleteSpectra=False,plotting=False,standardize=True,doPCA=None,findPeaks=None,makeDerivative=None,featureFilter=None,loadFeatures=None,deleteFeatures=None,removeVar=False,removeCor=None,useSavitzkyGolay=False,addNoiseColumns=None,addLandscapes=False)
    #model = Pipeline([('pca', PCA(n_components=40)),('filter', GenericUnivariateSelect(f_regression, param=98,mode='percentile')), ('model', LinearRegression())])
    #xmodel = XModel("pcreg2_Sand",classifier=model,Xtrain=X,Xtest=Xtest,ytrain=ymat[['Sand']])
    #ensemble.append(xmodel)
    
    #some info
    for m in ensemble:
	m.summary()
    return(ensemble)

    
def finalizeModel(m,binarizeProbas=False):
	"""
	Make predictions and save them
	"""
	print "Make predictions and save them..."
	if binarizeProbas:
	    #oob class predictions,binarize data
	    vfunc = np.vectorize(binarizeProbs)
	    #oob from crossvalidation
	    yoob = vfunc(np.asarray(m.oob_preds),m.cutoff)
	    #probas final prediction
	    m.preds = m.classifier.predict_proba(m.Xtest)[:,1]	  	    
	    #classes for final test set
	    ypred = vfunc(np.asarray(m.preds),m.cutoff)

	else:
	    #if 0-1 outcome, only classes or regression
	    #oob from crossvalidation
	    yoob = m.oob_preds
	    #final prediction
	    m.preds = m.classifier.predict(m.Xtest)	    
	    ypred = m.preds
	    
	m.summary()	
	
	#put data to data.frame and save
	#OOB DATA
	m.oob_preds=pd.DataFrame(np.asarray(m.oob_preds),columns=m.ytrain.columns)
		
	#TESTSET prediction	
	m.preds=pd.DataFrame(np.asarray(m.preds),columns=m.ytrain.columns)
	#save final model
	#TESTSETscores
	#OOBDATA
	allpred = pd.concat([m.preds, m.oob_preds])
	#print allpred
	#submission data is first, train data is last!
	filename="/home/loschen/Desktop/datamining-kaggle/african_soil/data/"+m.name+".csv"
	print "Saving oob + predictions as csv to:",filename
	allpred.to_csv(filename,index=False)
	
	#XModel.saveModel(m,"/home/loschen/Desktop/datamining-kaggle/higgs/data/"+m.name+".pkl")
	XModel.saveCoreData(m,"/home/loschen/Desktop/datamining-kaggle/african_soil/data/"+m.name+".pkl")
	return(m)
    

def createOOBdata_parallel(ensemble,repeats=1,nfolds=4,n_jobs=1,score_func='rmse',verbose=False,calibrate=False,binarize=False):
    """
    parallel oob creation
    """
    global funcdict

    for m in ensemble:
	print "Computing oob predictions for:",m.name
	print m.classifier.get_params
	oob_preds=np.zeros((m.Xtrain.shape[0],repeats))
	ly=m.ytrain.values.flatten()
	oobscore=np.zeros(repeats)
	maescore=np.zeros(repeats)
	
	#outer loop
	for j in xrange(repeats):
	    #cv = KFold(ly.shape[0], n_folds=nfolds,shuffle=True,random_state=j)
	    #cv = cross_validation.LeavePLabelOut(pd.read_csv('/home/loschen/Desktop/datamining-kaggle/african_soil/landscapes_quick3.csv',index_col=0)['LANDSCAPE'],1)
	    cv = cross_validation.LeavePLabelOut(pd.read_csv('/home/loschen/Desktop/datamining-kaggle/african_soil/landscapes_int.csv',index_col=0)['LANDSCAPE'],1)
	    #cv = StratifiedKFold(ly, n_folds=nfolds,shuffle=True,random_state=None)
	    #cv = StratifiedShuffleSplit(ly, n_iter=nfolds, test_size=0.25,random_state=j)
	    
	    scores=np.zeros(len(cv))
	    scores_mae=np.zeros(len(cv))
	    
	    #parallel stuff
	    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
			    pre_dispatch='2*n_jobs')
	    
	    #parallel run
	    oob_pred = parallel(delayed(fit_and_score)(clone(m.classifier), m.Xtrain, ly, train, test)
			  for train, test in cv)
	    
	    oob_pred=np.array(oob_pred)[:, 0]
	    for i,(train,test) in enumerate(cv):
		oob_preds[test,j] = oob_pred[i]
		scores[i]=funcdict[score_func](ly[test],oob_preds[test,j])
		scores_mae[i]=funcdict['mae'](ly[test],oob_preds[test,j])

	    oobscore[j]=funcdict[score_func](ly,oob_preds[:,j])
	    maescore[j]=funcdict['mae'](ly,oob_preds[:,j])
	    
	    print "Iteration:",j,
	    print " <score>: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std()),
	    print " score,oob: %0.3f" %(oobscore[j]),
	    print " ## <mae>: %0.3f (+/- %0.3f)" % (scores_mae.mean(), scores_mae.std()),
	    print " mae,oob: %0.3f" %(maescore[j])
	    
	#simple averaging of blending
	oob_avg=np.mean(oob_preds,axis=1)
	#probabilities
	m.oob_preds=oob_avg	

	if binarize:
	    #if 0-1 outcome, only classes
	    vfunc = np.vectorize(binarizeProbs)
	    m.oob_preds=vfunc(oob_avg,0.5)
	   
	score_oob = funcdict[score_func](ly,m.oob_preds)
	mae_oob = funcdict['mae'](ly,m.oob_preds)
	print "Summary: <score,oob>: %6.3f +- %6.3f   score,oob-total: %0.3f (after %d repeats) ## mae,oob-total: %0.3f" %(oobscore.mean(),oobscore.std(),score_oob,repeats,mae_oob)
	
	#Train model with test sets
	print "Train full modell...",	
	if m.sample_weight is not None:
	    print "... with sample weights"
	    m.classifier.fit(m.Xtrain,ly,m.sample_weight)
	else:
	    m.classifier.fit(m.Xtrain,ly)
	
	m=finalizeModel(m)
	
    return(ensemble)
    
def fit_and_score(xmodel,X,y,train,test,sample_weight=None,scale_wt=None,use_proba=False,cutoff=0.5):
    """
    Score function for parallel oob creation
    """
    Xtrain = X.iloc[train]
    Xvalid = X.iloc[test]
    ytrain = y[train].flatten()
    
    if sample_weight is not None: 
	wtrain = sample_weight[train]
	xmodel.fit(Xtrain,ytrain,sample_weight=wtrain)
    else:
	wtrain_fit=None
	xmodel.fit(Xtrain,ytrain)
    
    if use_proba:
	#saving out-of-bag predictions
	local_pred = xmodel.predict_proba(Xvalid)[:,1]	    
    #classification/regression   
    else:
	local_pred = xmodel.predict(Xvalid)
    
    #score=funcdict['rmse'](y[test],local_pred)
    #rndint = randint(0,10000000)
    #print "score %6.3f - %10d "%(score,rndint)
    #outpd = pd.DataFrame({'truth':y[test].flatten(),'pred':local_pred.flatten()})
    #outpd.index = Xvalid.index
    #outpd = pd.concat([Xvalid[['TMAP']], outpd],axis=1)
    #outpd.to_csv("pred"+str(rndint)+".csv")
    #raw_input()
    
    
    return [local_pred.flatten()]


def trainEnsemble(ensemble,mode='classical',useCols=None,addMetaFeatures=False,use_proba=True,dropCorrelated=False,subfile=""):
    """
    Train the ensemble
    """
    basedir="/home/loschen/Desktop/datamining-kaggle/african_soil/data/"
    xensemble=[]
    for i,model in enumerate(ensemble):
	print "Loading model:",i," name:",model
	xmodel = XModel.loadModel(basedir+model)
	print "OOB data:",xmodel.oob_preds.shape
	print "y data:",xmodel.preds.shape
	if i>0:
	    Xtrain = pd.concat([Xtrain,xmodel.oob_preds], axis=1)
	    Xtest = pd.concat([Xtest,xmodel.preds], axis=1)
	    
	else:
	    #print type(xmodel.oob_preds)
	    #print xmodel.oob_preds
	    #Xall = pd.read_csv("/home/loschen/Desktop/datamining-kaggle/higgs/data/"+model+".csv", sep=",")[col]
	    Xtrain = xmodel.oob_preds
	    Xtest = xmodel.preds
	xensemble.append(xmodel)

    y = xensemble[0].ytrain
    Xtrain.columns=ensemble
    #print Xtrain.head(10)
    #print Xtrain.describe()
    
    Xtest.columns=ensemble
    #print Xtest.head(10)
    #print Xtest.describe()
    
    #scale data
    X_all=pd.concat([Xtest,Xtrain])   
    if dropCorrelated: X_all=removeCorrelations(X_all,0.995)
    #X_all = (X_all - X_all.min()) / (X_all.max() - X_all.min())
    
    Xtrain=X_all[Xtest.shape[0]:]
    Xtest=X_all[:Xtest.shape[0]]
    
    if mode is 'classical':
	results=classicalBlend(ensemble,Xtrain,Xtest,y,subfile=subfile)
    elif mode is 'mean':
	results=linearBlend(ensemble,Xtrain,Xtest,y,takeMean=True,subfile=subfile)
    else:
	results=linearBlend(ensemble,Xtrain,Xtest,y,takeMean=False,subfile=subfile)
    return(results)
    
    
def voting(ensemble,Xtrain,Xtest,y,test_indices,subfile):
    """
    Voting for simple classifiction result
    """
    
    vfunc = np.vectorize(binarizeProbs)
    
    print "Majority voting for predictions"
    weights=np.asarray(pd.read_csv('../datamining-kaggle/higgs/training.csv', sep=",", na_values=['?'], index_col=0)['Weight'])

    
    for i,col in enumerate(Xtrain.columns):
	cutoff = computeCutoff(Xtrain[col].values,False)
	#label=col+"_"+str(i)
	Xtrain[col]=vfunc(Xtrain[col].values,cutoff)
	#oob_avg=np.mean(Xtmp,axis=1)
	score=ams_score(y,Xtrain[col].values,sample_weight=weights,use_proba=False,cutoff=cutoff,verbose=False)
	print "%4d AMS,oob data: %0.4f colum: %20s" % (i,score,col)
	cutoff = computeCutoff(Xtest[col].values,False)
	Xtest[col]=vfunc(Xtest[col].values,cutoff)
	#plt.hist(np.asarray(Xtrain[col]),bins=50,label=col+'_oob')
	#plt.hist(np.asarray(Xtest[col]),bins=50,label=col+'pred',alpha=0.2)
	#plt.legend()
	#plt.show()
	#del Xtrain[col]
    
    
    oob_avg=np.asarray(np.mean(Xtrain,axis=1))
    score=ams_score(y,oob_avg,sample_weight=weights,use_proba=False,cutoff=0.5,verbose=True)
    print " AMS,oob all: %0.4f" % (score)
    
    preds=np.asarray(np.mean(Xtest,axis=1))
    
    plt.hist(np.asarray(oob_avg),bins=30,label='oob')
    plt.hist(preds,bins=50,label='pred',alpha=0.2)
    plt.legend()
    plt.show()
    
    if len(subfile)>1:
	Xtest.index=test_indices
	makePredictions(preds,Xtest,subfile,useProba=True,cutoff=0.5)
	checksubmission(subfile)
    

def classicalBlend(ensemble,oobpreds,testset,ly,test_indices=None,subfile="subXXX.csv"):
    weights=np.asarray(pd.read_csv('../datamining-kaggle/higgs/training.csv', sep=",", na_values=['?'], index_col=0)['Weight'])
    #blending
    folds=16
    cutoff_all='compute'
    scale_wt='auto'
    
    print "Blending, using general cutoff %6s, "%(str(cutoff_all)),
    if scale_wt is not None:
	print "scale_weights %6s:"%(str(scale_wt))
  
    #blender=LogisticRegression(penalty='l2', tol=0.0001, C=1.0)
    #blender = Pipeline([('filter', SelectPercentile(f_regression, percentile=25)), ('model', LogisticRegression(penalty='l2', tol=0.0001, C=0.1))])
    blender=SGDClassifier(alpha=0.1, n_iter=50,penalty='l2',loss='log',n_jobs=folds)
    #blender=AdaBoostClassifier(learning_rate=0.01,n_estimators=50)
    #blender=RandomForestClassifier(n_estimators=50,n_jobs=4, max_features='auto',oob_score=False)
    #blender=ExtraTreesClassifier(n_estimators=500,max_depth=None,min_samples_leaf=5,n_jobs=4,criterion='entropy', max_features='auto',oob_score=False)
    #blender=RandomForestClassifier(n_estimators=500,max_depth=None,min_samples_leaf=10,n_jobs=1,criterion='entropy', max_features=5,oob_score=False)
    #blender=ExtraTreesRegressor(n_estimators=500,max_depth=None)
    #cv = KFold(oobpreds.shape[0], n_folds=folds,random_state=123)
    cv = StratifiedShuffleSplit(ly, n_iter=folds, test_size=0.5)
    blend_scores=np.zeros(folds)
    ams_scores=np.zeros(folds)
    blend_oob=np.zeros((oobpreds.shape[0]))
    for i, (train, test) in enumerate(cv):
	Xtrain = oobpreds.iloc[train]
	Xtest = oobpreds.iloc[test]
	wfit=None
	if scale_wt is not None:
	  wfit = modTrainWeights(weights[train],ly[train],scale_wt)
	  blender.fit(Xtrain, ly[train],sample_weight=wfit)
	else:
	  blender.fit(Xtrain, ly[train])
	
	
	if hasattr(blender,'predict_proba'):
	    blend_oob[test] = blender.predict_proba(Xtest)[:,1]
	    use_proba=True
	else:
	    print "Warning: Using predict, no proba!"
	    blend_oob[test] = blender.predict(Xtest)
	    use_proba=False
	blend_scores[i]=roc_auc_score(ly[test],blend_oob[test])
	ams_scores[i]=ams_score(ly[test],blend_oob[test],sample_weight=weights[test],use_proba=use_proba,cutoff=cutoff_all,verbose=False)
    
    print " <AUC>: %0.4f (+/- %0.4f)" % (blend_scores.mean(), blend_scores.std()),
    oob_auc=roc_auc_score(ly,blend_oob)
    print " AUC oob: %0.4f" %(oob_auc)
    
    print " <AMS>: %0.4f (+/- %0.4f)" % (ams_scores.mean(), ams_scores.std()),
    oob_ams=ams_score(ly,blend_oob,sample_weight=weights,use_proba=use_proba,cutoff=cutoff_all,verbose=False)
    print " AMS oob: %0.4f" %(oob_ams)
    
    if hasattr(blender,'coef_'):
      print "%-16s %5s %5s %5s" %("model","AMS","auc","coef")
      for i,model in enumerate(oobpreds.columns):
	coldata=np.asarray(oobpreds.iloc[:,i])
	ams=ams_score(ly,coldata,sample_weight=weights,use_proba=use_proba,cutoff=cutoff_all,verbose=False)
	auc=roc_auc_score(ly, coldata)
	print "%-16s %5.3f %5.3f %5.3f" %(model,ams,auc,blender.coef_[0][i])
      print "sum coef: %4.4f"%(np.sum(blender.coef_))
    #plt.plot(range(len(ensemble)),scores,'ro')
    
    if len(subfile)>1:
	#Prediction
	print "Make final ensemble prediction..."
	#make prediction for each classifiers   
	if scale_wt is not None:
	    wfit = modTrainWeights(weights,ly,scale_wt)
	    preds=blender.fit(oobpreds,ly,sample_weight=wfit)
	else:
	    preds=blender.fit(oobpreds,ly)
	#blend results
	preds=blender.predict_proba(testset)[:,1]
	#print preds
	plt.hist(blend_oob,bins=50,label='oob')
	plt.hist(preds,bins=50,alpha=0.3,label='pred')
	plt.legend()
	plt.show()
	
	#preds=pd.DataFrame(preds,columns=["label"],index=test_indices)
	testset.index=test_indices
	makePredictions(preds,testset,subfile,useProba=True,cutoff=cutoff_all)
	checksubmission(subfile)
    #preds.to_csv('/home/loschen/Desktop/datamining-kaggle/higgs/submissions/subXXXa.csv')
    #print preds
    #return(oob_ams)
    return(ams_scores.mean())


def linearBlend(ensemble,Xtrain,Xtest,y,weights=None,score_func='rmse',test_indices=None,normalize=True,takeMean=False,removeZeroModels=0.0,alpha=None,subfile="",plotting=False):

    def fopt(params):
	# nxm  * m*1 ->n*1
	if np.isnan(np.sum(params)):
	    print "We have NaN here!!"
	    score=0.0
	else:
	    ypred=np.dot(Xtrain,params).flatten()
	    score=funcdict[score_func](ypred,y)
	    #score=funcdict['mae'](ypred,y.values)
	    
	    #regularization
	    if alpha is not None:
	      penalty=alpha*np.sum(np.square(params))
	      print "orig score:%8.3f"%(score),
	      score=score-penalty
	      print " - Regularization - alpha: %8.3f penalty: %8.3f regularized score: %8.3f"%(alpha,penalty,score) 
	return score

    y = np.asarray(y)
    lowerbound=0.0
    upperbound=0.3
    constr=[lambda x,z=i: x[z]-lowerbound for i in range(len(Xtrain.columns))]
    constr2=[lambda x,z=i: upperbound-x[z] for i in range(len(Xtrain.columns))]
    constr=constr+constr2
    
    n_models=len(Xtrain.columns)
    x0 = np.ones((n_models, 1)) / n_models
    #x0= np.random.random_sample((n_models,1))
    
    xopt = fmin_cobyla(fopt, x0,constr,rhoend=1e-7,maxfun=5000)
    
    if takeMean:
	print "Taking the mean..."
	xopt=x0
    
    #normalize coefficient
    if normalize: xopt=xopt/np.sum(xopt)
    
    if np.isnan(np.sum(xopt)):
	    print "We have NaN here!!"
    
    ypred=np.dot(Xtrain,xopt).flatten()
    #ypred = ypred/np.max(ypred)#normalize?
    
    ymean= np.dot(Xtrain,x0).flatten()
    #ymean = ymean/np.max(ymean)#normalize?
    
    oob_score=funcdict[score_func](y,ypred)
    print "->score,opt: %4.4f" %(oob_score)
    score=funcdict[score_func](y,ymean)
    print "->score,mean: %4.4f" %(score)
    
    
    zero_models=[]
    print "%4s %-48s %6s %6s" %("nr","model","score","coeff")
    for i,model in enumerate(Xtrain.columns):
	coldata=np.asarray(Xtrain.iloc[:,i])
	score = funcdict[score_func](y,coldata)	
	print "%4d %-48s %6.3f %6.3f" %(i+1,model,score,xopt[i])
	if xopt[i]<removeZeroModels:
	    zero_models.append(model)
    if not normalize: print "##sum coefficients: %4.4f"%(np.sum(xopt))
    
    if removeZeroModels>0.0:
	print "Dropping ",len(zero_models)," columns:",zero_models
	Xtrain=Xtrain.drop(zero_models,axis=1)
	Xtest=Xtest.drop(zero_models,axis=1)
	return (Xtrain,Xtest)
    
    #prediction flatten makes a n-dim row vector from a nx1 column vector...
    preds=np.dot(Xtest,xopt).flatten()

    if plotting:
	plt.hist(ypred,bins=50,alpha=0.3,label='oob')
	plt.hist(preds,bins=50,alpha=0.3,label='pred')
	plt.legend()
	plt.show()
    
    #return dataframes with blending results
    Xtrain['blend']=ypred
    Xtest['blend']=preds
    Xtrain['ytrain']=y

    return(Xtrain,Xtest)
  
def selectModels(ensemble,startensemble=[],niter=10,mode='amsMaximize',useCols=None): 
    """
    Random mode for best model selection
    """
    randBinList = lambda n: [randint(0,1) for b in range(1,n+1)]
    auc_list=[0.0]
    ens_list=[]
    cols_list=[]
    for i in range(niter):
	print "iteration %5d/%5d, current max_score: %6.3f"%(i+1,niter,max(auc_list))
	actlist=randBinList(len(ensemble))
	actensemble=[x for x in itertools.compress(ensemble,actlist)]
	actensemble=startensemble+actensemble
	print actensemble
	#print actensemble
	auc=trainEnsemble(actensemble,mode=mode,useCols=useCols,addMetaFeatures=False,dropCorrelated=False)
	auc_list.append(auc)
	ens_list.append(actensemble)
	#cols_list.append(actCols)
    maxauc=0.0
    topens=None
    topcols=None
    for ens,auc in zip(ens_list,auc_list):	
	print "SCORE: %4.4f" %(auc),
	print ens
	if auc>maxauc:
	  maxauc=auc
	  topens=ens
	  #topcols=col
    print "\nTOP ensemble:",topens
    print "TOP score: %4.4f" %(maxauc)

def selectModelsGreedy(ensemble,startensemble=[],niter=2,mode='classical',useCols=None,dropCorrelated=False):    
    """
    Select best models in a greedy forward selection
    """
    topensemble=startensemble
    score_list=[]
    ens_list=[]
    bestscore=0.0
    for i in range(niter):
	maxscore=0.0
	topidx=-1
	for j in range(len(ensemble)):
	    if ensemble[j] not in topensemble:
		actensemble=topensemble+[ensemble[j]]
	    else:
		continue
	    
	    auc=trainEnsemble(actensemble,mode=mode,useCols=useCols,addMetaFeatures=False,dropCorrelated=dropCorrelated)
	    print "##(Current top score: %4.4f | overall best score: %4.4f) actual score: %4.4f  - " %(maxscore,bestscore,auc),
	    print actensemble
	    if auc>maxscore:
		maxscore=auc
		topidx=j
	#pick best set
	if not maxscore+0.05>bestscore:
	    print "Not gain in score anymore, leaving..."
	    break
	topensemble.append(ensemble[topidx])
	print "TOP score: %4.4trainEnsemblef" %(maxscore),
	print " - actual ensemble:",topensemble
	score_list.append(maxscore)
	ens_list.append(list(topensemble))
	if maxscore>bestscore:
	    bestscore=maxscore
    
    for ens,score in zip(ens_list,score_list):	
	print "SCORE: %4.4f" %(score),
	print ens
	
    plt.plot(score_list)
    plt.show()
    return topensemble
    
def trainAllEnsembles(models,mode='linearBlend',plotting=False,subfile=""):
    """
    Train ensembles for all targets 
    """
    targets=["Ca","P","pH","SOC","Sand"]
    train_list=[]
    test_list=[]
    for target in targets:
	#generate submodels
	submodels=[model +"_"+target for model in models]
	Xtrain,Xtest = trainEnsemble(submodels,mode=mode,useCols=None,addMetaFeatures=False,use_proba=True,dropCorrelated=False,subfile="")
	
	print Xtrain.describe()
	print Xtest.describe()
	
	train_list.append(Xtrain)
	test_list.append(Xtest)
    
    #Final statistics#
    print "%10s "%("target"),
    for col in Xtest.columns:
	print "%14s "%(col),
    print ""
    
    score_blend=np.zeros(len(targets))
    score_models=np.zeros((len(targets),len(Xtest.columns)))
    for i,(target,Xtrain,Xtest) in enumerate(zip(targets,train_list,test_list)):
	score_blend[i]=funcdict['rmse'](Xtrain.ytrain,Xtrain.blend)
	print "%10s"%(target),
	for j,col in enumerate(Xtest.columns):
	    score_models[i,j]=funcdict['rmse'](Xtrain.ytrain,Xtrain[col])
	    print " %14.3f"%(score_models[i,j]),
	
	print ""
	
	if plotting:
	    plt.hist(Xtrain.blend,bins=50,alpha=0.3,label='train')
	    plt.hist(Xtest.blend,bins=50,alpha=0.3,label='test')
	    plt.legend()
	    plt.show()
	
    
    print "%10s"%('avg.'),
    for i in xrange(score_models.shape[1]):
	print " %14.3f"%(score_models[:,i].mean()),
    
    
    print "\navg,blend: %6.3f (+- %6.3f)"%(score_blend.mean(),score_blend.std())
    #print "avg,blend: %6.3f (+- %6.3f)"%(score_blend[].mean(),score_blend.std())
    
    if subfile is not "":
	sample = pd.read_csv('/home/loschen/Desktop/datamining-kaggle/african_soil/sample_submission.csv')
	for target,Xtest in zip(targets,test_list):
	    sample[target] = Xtest.blend
	print "Saving submission to ",subfile
	sample.to_csv(subfile, index = False)
    
    
if __name__=="__main__":
    np.random.seed(123)
    #ensemble=createModels()
    #ensemble=createOOBdata_parallel(ensemble,repeats=1,nfolds=8,n_jobs=8) #oob data averaging leads to significant variance reduction
    
    #models=['pls1','svm1','pcaridge1','leaps1','rf1']
    models_LS=['leaps2','svm3','svm2','pcridge2','pls2','elnet2','pcreg2','gbm2','compridge2','ridge2']
    #models_LS=['leaps2']
    #models_LS=['svm2','elnet2','pls2','ridge2','ridge2']
    #models=['pcaridge1']
    #useCols=['A']
    useCols=None
    trainAllEnsembles(models_LS,plotting=True,mode='linearBlend',subfile='/home/loschen/Desktop/datamining-kaggle/african_soil/submissions/sub1610b.csv')
    #trainEnsemble(model,mode='classical',useCols=useCols,addMetaFeatures=False,use_proba=True,dropCorrelated=False,subfile='/home/loschen/Desktop/datamining-kaggle/higgs/submissions/sub1509b.csv')
    