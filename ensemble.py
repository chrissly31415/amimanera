#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""  Chrissly31415
August,September 2013

"""

from higgs import *
from FullModel import *
import itertools
from scipy.optimize import fmin,fmin_cobyla
from random import randint
import sys
from sklearn.externals.joblib import Parallel, delayed, logger
from sklearn.base import clone

train_indices=[]
test_indices=[]

def createModels():
    global train_indices,test_indices
    ensemble=[]
    #GBMTEST AMS~ 3.675 +/-0.014 PL ~3.671
    #X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=None,dropCorrelated=False,scale_data=False,clusterFeature=False)
    #model = GradientBoostingClassifier(loss='deviance',n_estimators=800, learning_rate=0.02, max_depth=6,subsample=1.0,max_features=8,min_samples_leaf=100,verbose=False)
    #xmodel = XModel("gbm_test1",model,X,Xtest,w,cutoff='compute',scale_wt=200)
    #ensemble.append(xmodel)
    
    #GBMTEST 3.650 +/- 0.003	PL ~3.62
    #X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=None,dropCorrelated=False,scale_data=False,clusterFeature=False)
    #model = GradientBoostingClassifier(loss='deviance',n_estimators=800, learning_rate=0.02, max_depth=6,subsample=1.0,max_features=8,min_samples_leaf=100,verbose=False)
    #xmodel = XModel("gbm_test2",model,X,Xtest,w,cutoff='compute',scale_wt=200)
    #ensemble.append(xmodel)
    
    #GBMTEST 3.676 +- 0.003 PL ~3.636
    #X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=None,dropCorrelated=False,scale_data=False,clusterFeature=False)
    #model = GradientBoostingClassifier(loss='deviance',n_estimators=800, learning_rate=0.02, max_depth=6,subsample=1.0,max_features=8,min_samples_leaf=100,verbose=False)
    #xmodel = XModel("gbm_test3",model,X,Xtest,w,cutoff='compute',scale_wt=200)
    #ensemble.append(xmodel)
    
    #GBMTEST AMS~ 3.671 0.014
    #X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=None,dropCorrelated=False,scale_data=False,clusterFeature=False)
    #model = GradientBoostingClassifier(loss='deviance',n_estimators=800, learning_rate=0.02, max_depth=6,subsample=1.0,max_features=8,min_samples_leaf=100,verbose=False)
    #xmodel = XModel("gbm_test4",model,X,Xtest,w,cutoff='compute',scale_wt=200)
    #ensemble.append(xmodel)
    
    #GBMTEST AMS~ 3.648 0.019
    #X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=None,dropCorrelated=False,scale_data=False,clusterFeature=False)
    #model = GradientBoostingClassifier(loss='deviance',n_estimators=800, learning_rate=0.02, max_depth=6,subsample=1.0,max_features=8,min_samples_leaf=50,verbose=False)
    #xmodel = XModel("gbm_test5",model,X,Xtest,w,cutoff='compute',scale_wt=200)
    #ensemble.append(xmodel)
    
    #GBMTEST AMS~ 3.673   0.020
    #X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=None,dropCorrelated=False,scale_data=False,clusterFeature=False)
    #model = GradientBoostingClassifier(loss='deviance',n_estimators=800, learning_rate=0.02, max_depth=7,subsample=.5,max_features=8,min_samples_leaf=50,verbose=False)
    #xmodel = XModel("gbm_test6",model,X,Xtest,w,cutoff=0.85,scale_wt=200)
    #ensemble.append(xmodel)

    #GBMTEST 3.676 0.018 PL ~3.65471
    #X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=None,dropCorrelated=False,scale_data=False,clusterFeature=False)
    #model = GradientBoostingClassifier(loss='deviance',n_estimators=800, learning_rate=0.02, max_depth=7,subsample=.5,max_features=6,min_samples_leaf=150,verbose=False)
    #xmodel = XModel("gbm_test7",model,X,Xtest,w,cutoff=0.85,scale_wt=200)
    #ensemble.append(xmodel)
  
    #GBMTEST 3.671 0.012   PL ~3.67298
    #X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=None,dropCorrelated=False,scale_data=False,clusterFeature=False)
    #model = GradientBoostingClassifier(loss='deviance',n_estimators=1200, learning_rate=0.02, max_depth=7,subsample=.5,max_features=8,min_samples_leaf=100,verbose=False)
    #xmodel = XModel("gbm_test8",model,X,Xtest,w,cutoff=0.85,scale_wt=200)
    #ensemble.append(xmodel)
    
    #GBMTEST 
    #X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=None,dropCorrelated=False,scale_data=False,clusterFeature=False)
    #model = GradientBoostingClassifier(loss='deviance',n_estimators=800, learning_rate=0.02, max_depth=7,subsample=.5,max_features=8,min_samples_leaf=150,verbose=False)
    #xmodel = XModel("gbm_test9",model,X,Xtest,w,cutoff=0.85,scale_wt=200)
    #ensemble.append(xmodel)
    
    #GBMTEST 
    #X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=None,dropCorrelated=False,scale_data=False,clusterFeature=False)
    #model = GradientBoostingClassifier(loss='deviance',n_estimators=1200, learning_rate=0.02, max_depth=7,subsample=.5,max_features=6,min_samples_leaf=150,verbose=False)
    #xmodel = XModel("gbm_test10",model,X,Xtest,w,cutoff=0.85,scale_wt=200)
    #ensemble.append(xmodel)
    
    #GBMTEST 
    #X,y,Xtest,w=prepareDatasets(nsamples=200000,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=None,dropCorrelated=False,scale_data=False,clusterFeature=False)
    #model = GradientBoostingClassifier(loss='deviance',n_estimators=1200, learning_rate=0.02, max_depth=6,subsample=.5,max_features=8,min_samples_leaf=150,verbose=False)
    #xmodel = XModel("gbm_subsetb",model,X,Xtest,w,cutoff='compute',scale_wt=200)
    #ensemble.append(xmodel)
    
    #GBMTEST 
    #X,y,Xtest,w=prepareDatasets(nsamples=200000,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=None,dropCorrelated=False,scale_data=False,clusterFeature=False)
    #model = GradientBoostingClassifier(loss='deviance',n_estimators=1200, learning_rate=0.02, max_depth=6,subsample=.5,max_features=8,min_samples_leaf=150,verbose=False)
    #xmodel = XModel("gbm_subsetc",model,X,Xtest,w,cutoff='compute',scale_wt=200)
    #ensemble.append(xmodel)
    
    #GBMTEST 
    #X,y,Xtest,w=prepareDatasets(nsamples=200000,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=None,dropCorrelated=False,scale_data=False,clusterFeature=False)
    #model = GradientBoostingClassifier(loss='deviance',n_estimators=1200, learning_rate=0.02, max_depth=6,subsample=.5,max_features=8,min_samples_leaf=150,verbose=False)
    #xmodel = XModel("gbm_subsetd",model,X,Xtest,w,cutoff='compute',scale_wt=200)
    #ensemble.append(xmodel)
    
    
    
    #GBMTEST 3.678 +/- 0.011 ~3.63433
    #X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=None,dropCorrelated=False,scale_data=False,clusterFeature=False)
    #model = GradientBoostingClassifier(loss='deviance',n_estimators=1600, learning_rate=0.01, max_depth=6,subsample=1.0,max_features=8,min_samples_leaf=100,verbose=False)
    #xmodel = XModel("gbm_test2",model,X,Xtest,w,cutoff='compute',scale_wt=200)
    #ensemble.append(xmodel)
    
    #GBMTEST 3.669 +/- 0.008 ~ 3.62735
    #X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=None,dropCorrelated=False,scale_data=False,clusterFeature=False)
    #model = GradientBoostingClassifier(loss='deviance',n_estimators=2000, learning_rate=0.01, max_depth=6,subsample=1.0,max_features=8,min_samples_leaf=100,verbose=False)
    #xmodel = XModel("gbm_test3",model,X,Xtest,w,cutoff='compute',scale_wt=200)
    #ensemble.append(xmodel)
    
    #GBMTEST 3.682 +/- 0.014 overfitted??? PL 3.66648
    #X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=None,dropCorrelated=False,scale_data=False,clusterFeature=False)
    #model = GradientBoostingClassifier(loss='deviance',n_estimators=2000, learning_rate=0.01, max_depth=7,subsample=1.0,max_features=8,min_samples_leaf=100,verbose=False)
    #xmodel = XModel("gbm_test4",model,X,Xtest,w,cutoff='compute',scale_wt=200)
    #ensemble.append(xmodel)
    
    #GBM1 AMS~3.66 OK overfits the PL with 3.61
    #X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=None,dropCorrelated=False,scale_data=False,clusterFeature=False)
    #model = GradientBoostingClassifier(loss='deviance',n_estimators=200, learning_rate=0.08, max_depth=7,subsample=1.0,max_features=10,min_samples_leaf=20,verbose=False) #opt weight =500 AMS=3.548
    #xmodel = XModel("gbm1",model,X,Xtest,w,cutoff=0.85,scale_wt=200)
    #ensemble.append(xmodel)
    
    #GBM3 
    #X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=None,dropCorrelated=False,scale_data=False,clusterFeature=False)
    #model = GradientBoostingClassifier(loss='deviance',n_estimators=200, learning_rate=0.1, max_depth=6,subsample=.5,max_features=8,min_samples_leaf=100,verbose=False)
    #xmodel = XModel("gbm3",model,X,Xtest,w,cutoff=0.85,scale_wt=200)
    #ensemble.append(xmodel)
    
    #GBM4
    #X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=None,dropCorrelated=False,scale_data=False,clusterFeature=False)
    #model = GradientBoostingClassifier(loss='deviance',n_estimators=200, learning_rate=0.1, max_depth=6,subsample=.5,max_features=8,min_samples_leaf=100,verbose=False)
    #xmodel = XModel("gbm4",model,X,Xtest,w,cutoff=0.85,scale_wt=200)
    #ensemble.append(xmodel)
    
    #X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=None,dropCorrelated=False,scale_data=False,clusterFeature=False)
    #model = GradientBoostingClassifier(loss='deviance',n_estimators=200, learning_rate=0.1, max_depth=6,subsample=.5,max_features=8,min_samples_leaf=100,verbose=False)
    #xmodel = XModel("gbm5",model,X,Xtest,w,cutoff=0.85,scale_wt=200)
    #ensemble.append(xmodel)
    
    #X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=None,dropCorrelated=False,scale_data=False,clusterFeature=False)
    #model = GradientBoostingClassifier(loss='deviance',n_estimators=200, learning_rate=0.1, max_depth=6,subsample=.5,max_features=8,min_samples_leaf=100,verbose=False)
    #xmodel = XModel("gbm6",model,X,Xtest,w,cutoff=0.85,scale_wt=200)
    #ensemble.append(xmodel)
    
    #X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=None,dropCorrelated=False,scale_data=False,clusterFeature=False)
    #model = GradientBoostingClassifier(loss='deviance',n_estimators=200, learning_rate=0.1, max_depth=6,subsample=.5,max_features=8,min_samples_leaf=100,verbose=False)
    #xmodel = XModel("gbm7",model,X,Xtest,w,cutoff=0.85,scale_wt=200)
    #ensemble.append(xmodel)
    
    #X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=None,dropCorrelated=False,scale_data=False,clusterFeature=False)
    #model = GradientBoostingClassifier(loss='deviance',n_estimators=200, learning_rate=0.1, max_depth=6,subsample=.5,max_features=8,min_samples_leaf=100,verbose=False)
    #xmodel = XModel("gbm8",model,X,Xtest,w,cutoff=0.85,scale_wt=200)
    #ensemble.append(xmodel)
    
    #X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=None,dropCorrelated=False,scale_data=False,clusterFeature=False)
    #model = GradientBoostingClassifier(loss='deviance',n_estimators=200, learning_rate=0.1, max_depth=6,subsample=.5,max_features=8,min_samples_leaf=100,verbose=False)
    #xmodel = XModel("gbm9",model,X,Xtest,w,cutoff=0.85,scale_wt=200)
    #ensemble.append(xmodel)
    
    #X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=None,dropCorrelated=False,scale_data=False,clusterFeature=False)
    #model = GradientBoostingClassifier(loss='deviance',n_estimators=200, learning_rate=0.1, max_depth=6,subsample=.5,max_features=8,min_samples_leaf=100,verbose=False)
    #xmodel = XModel("gbm10",model,X,Xtest,w,cutoff=0.85,scale_wt=200)
    #ensemble.append(xmodel)
    
    
    #XRF1 AMS~3.4 OK
    #X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=None,dropCorrelated=False,scale_data=False,clusterFeature=False)
    #model = ExtraTreesClassifier(n_estimators=250,max_depth=None,min_samples_leaf=5,n_jobs=4,criterion='entropy', max_features=10,oob_score=False)##scale_wt 600 cutoff 0.85
    #xmodel = XModel("xrf1",model,X,Xtest,w,cutoff=0.85,scale_wt=600)
    #ensemble.append(xmodel)
    
    #RF1 AMS~3.5 OK
    #X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=None,dropCorrelated=False,scale_data=False,clusterFeature=False)
    #model =  RandomForestClassifier(n_estimators=300,max_depth=None,min_samples_leaf=10,n_jobs=4,criterion='entropy', max_features=5,oob_score=False)
    #xmodel = XModel("rf1",model,X,Xtest,w,cutoff=0.85,scale_wt=600)
    #ensemble.append(xmodel)
    
    #RF2 AMS~3.5 no weights for fit!!! OK
    #X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=None,dropCorrelated=False,scale_data=False,clusterFeature=False)
    #model =  RandomForestClassifier(n_estimators=250,max_depth=None,min_samples_leaf=5,n_jobs=4,criterion='entropy', max_features=5,oob_score=False)
    #xmodel = XModel("rf2",model,X,Xtest,w,cutoff=0.75,scale_wt=None)
    #ensemble.append(xmodel)
    
    #RF3 AMS~3.5 no weights for fit!!! OK
    #X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=None,dropCorrelated=False,scale_data=False,clusterFeature=False,createMassEstimate=True)
    #model =  RandomForestClassifier(n_estimators=300,max_depth=None,min_samples_leaf=5,n_jobs=4,criterion='entropy', max_features=5,oob_score=False)
    #xmodel = XModel("rf3",model,X,Xtest,w,cutoff='compute',scale_wt='auto')
    #ensemble.append(xmodel)
    
    #gbm_newfeat1 3.608+/-0.04
    #X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=None,dropCorrelated=False,scale_data=False,clusterFeature=False,createMassEstimate=True)
    #model = GradientBoostingClassifier(loss='deviance',n_estimators=200, learning_rate=0.08, max_depth=6,subsample=1.0,max_features=10,min_samples_leaf=20,verbose=False) #opt weight =500 AMS=3.548
    #xmodel = XModel("gbm_newfeat1",model,X,Xtest,w,cutoff=0.93,scale_wt='auto')
    #ensemble.append(xmodel)
    
    #gbm_newfeat2 3.642+/-0.025
    #X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=None,dropCorrelated=False,scale_data=False,clusterFeature=False,createMassEstimate=True)
    #model = GradientBoostingClassifier(loss='deviance',n_estimators=500, learning_rate=0.05, max_depth=6,subsample=1.0,max_features=6,min_samples_leaf=100,verbose=False)
    #xmodel = XModel("gbm_newfeat2",model,X,Xtest,w,cutoff=0.94,scale_wt='auto')
    #ensemble.append(xmodel)
    
    #gbm_newfeat3 3.660 +/-0.048
    #X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=None,dropCorrelated=False,scale_data=False,clusterFeature=False,createMassEstimate=True)
    #model = GradientBoostingClassifier(loss='deviance',n_estimators=800, learning_rate=0.03, max_depth=6,subsample=1.0,max_features=6,min_samples_leaf=100,verbose=False)
    #xmodel = XModel("gbm_newfeat3",model,X,Xtest,w,cutoff=0.94,scale_wt='auto')
    #ensemble.append(xmodel)
    
    #gbm_newfeat4 <AMS,cv>:  3.637 +-  0.022 (cutoff=0.94) <AMS,cv>:  3.656 +-  0.017 cutoff="compute"->0.94
    X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=None,dropCorrelated=False,scale_data=False,clusterFeature=False,createMassEstimate=True)
    model = GradientBoostingClassifier(loss='deviance',n_estimators=1000, learning_rate=0.03, max_depth=6,subsample=1.0,max_features=6,min_samples_leaf=100,verbose=False)
    xmodel = XModel("gbm_newfeat4",model,X,Xtest,w,cutoff="compute",scale_wt='auto')
    ensemble.append(xmodel)
    
    #gbm_newfeat5 <AMS,cv>:  3.643 +-  0.020
    X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=None,dropCorrelated=False,scale_data=False,clusterFeature=False,createMassEstimate=True)
    model = GradientBoostingClassifier(loss='deviance',n_estimators=1000, learning_rate=0.03, max_depth=6,subsample=1.0,max_features=10,min_samples_leaf=100,verbose=False)
    xmodel = XModel("gbm_newfeat5",model,X,Xtest,w,cutoff="compute",scale_wt='auto')
    ensemble.append(xmodel)
    
    #GBM2 loss exponential AMS ~3.5 ->predict_proba... OK
    #X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=None,dropCorrelated=False,scale_data=False,clusterFeature=False)
    #model = GradientBoostingClassifier(loss='exponential',n_estimators=200, learning_rate=0.2, max_depth=6,subsample=1.0,min_samples_leaf=5,verbose=False)
    #xmodel = XModel("gbm2",model,X,Xtest,w,cutoff=None,scale_wt=35)
    #ensemble.append(xmodel)
 
    #XGBOOST1 AMS ~3.58 (single model PUB.LD)
    #X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=None,dropCorrelated=False,scale_data=False,clusterFeature=False)
    #model = XgboostClassifier(n_estimators=120,learning_rate=0.1,max_depth=6,n_jobs=4,NA=-999.9)
    #xmodel = XModel("xgboost1",model,X,Xtest,w,cutoff=0.7,scale_wt=1)
    #ensemble.append(xmodel)
    
    #XGBOOST2 somewhat finetuned ~3.520
    #X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=None,dropCorrelated=False,scale_data=False,clusterFeature=False)
    #model = XgboostClassifier(n_estimators=200,learning_rate=0.08,max_depth=7,n_jobs=4,NA=-999.9)
    #xmodel = XModel("xgboost2",model,X,Xtest,w,cutoff=0.7,scale_wt=1)
    #ensemble.append(xmodel)
    
    #KNN1
    #X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=True,plotting=False,stats=False,transform=False,createNAFeats=None,dropCorrelated=False,scale_data=False,clusterFeature=False)
    #model = KNeighborsClassifier(n_neighbors=5,weights='distance',algorithm='ball_tree')#AMS~2.245
    #xmodel = XModel("KNN1",model,X,Xtest,w,cutoff=0.7,scale_wt=None)
    #ensemble.append(xmodel)
    
    #KNN2
    #X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=True,plotting=False,stats=False,transform=False,createNAFeats=None,dropCorrelated=True,scale_data=False,clusterFeature=False,dropFeatures=None,polyFeatures=None,createMassEstimate=True,imputeMassModel=None)
    #model = KNeighborsClassifier(n_neighbors=15)#AMS~2.4
    #xmodel = XModel("KNN2",model,X,Xtest,w,cutoff=0.7,scale_wt=None)
    #ensemble.append(xmodel)
    
    #NB1 with massmodel
    #X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=True,plotting=False,stats=False,transform=True,createNAFeats=None,dropCorrelated=True,scale_data=False,clusterFeature=False,dropFeatures=None,createMassEstimate=False,imputeMassModel='load')#,imputeMassModel=RandomForestRegressor(n_estimators=250,max_depth=None,min_samples_leaf=5,n_jobs=4,criterion='mse', max_features=5,oob_score=False))
    #model = GaussianNB()#AMS~1.8
    #xmodel = XModel("NB1",model,X,Xtest,w,cutoff=0.15,scale_wt=None)
    #ensemble.append(xmodel)
        
        
    #ADAboost AMS ~2.9
    #X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=None,dropCorrelated=False,scale_data=False,clusterFeature=False,dropFeatures=None,createMassEstimate=False,imputeMassModel=None)
    #model = Pipeline([('filter', SelectPercentile(f_classif, percentile=80)), ('model', AdaBoostClassifier(n_estimators=200,learning_rate=0.1))])
    #model = AdaBoostClassifier(n_estimators=200,learning_rate=0.1)
    #xmodel = XModel("ADAboost1",model,X,Xtest,w,cutoff=0.515,scale_wt=200)
    #ensemble.append(xmodel)
    
    #GBM bagging gbm_bag1 <AMS>: 3.7073 PL 3.67 (10 iterations), gbm_bag2 <AMS>: 3.729 (20 iterations) ????
    #X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=None,dropCorrelated=False,scale_data=False,clusterFeature=False)
    #model = GradientBoostingClassifier(loss='deviance',n_estimators=300, learning_rate=0.06, max_depth=7,subsample=1.0,max_features=10,min_samples_leaf=20,verbose=False) #opt weight =500 AMS=3.548
    #xmodel = XModel("gbm_bag3",model,X,Xtest,w,cutoff=0.85,scale_wt=200)
    #ensemble.append(xmodel)
    
    #GBM BAGGING severely overfits now new try PL 3.65
    #X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=None,dropCorrelated=False,scale_data=False,clusterFeature=False)
    #basemodel = GradientBoostingClassifier(loss='deviance',n_estimators=1200, learning_rate=0.02, max_depth=6,subsample=0.5,max_features=8,min_samples_leaf=150,verbose=False)
    #model = BaggingClassifier(base_estimator=basemodel,n_estimators=10,n_jobs=8,verbose=False)
    #xmodel = XModel("gbm_bag1",model,X,Xtest,w,cutoff='compute',scale_wt=200)
    #ensemble.append(xmodel)
    
    #GBM BAGGING new try
    #X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=None,dropCorrelated=False,scale_data=False,clusterFeature=False)
    #basemodel = GradientBoostingClassifier(loss='deviance',n_estimators=300, learning_rate=0.08, max_depth=7,subsample=1.0,max_features=10,min_samples_leaf=20,verbose=False) 
    #model = BaggingClassifier(base_estimator=basemodel,n_estimators=40,n_jobs=8,max_features=1.0,verbose=1)
    #xmodel = XModel("gbm_realbag1",model,X,Xtest,w,cutoff=0.85,scale_wt=200)
    #ensemble.append(xmodel)
    
    #X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=None,dropCorrelated=False,scale_data=False,clusterFeature=False)
    #basemodel = GradientBoostingClassifier(loss='deviance',n_estimators=300, learning_rate=0.08, max_depth=7,subsample=1.0,max_features=10,min_samples_leaf=20,verbose=False) 
    #model = BaggingClassifier(base_estimator=basemodel,n_estimators=80,n_jobs=8,max_features=1.0,verbose=1)
    #xmodel = XModel("gbm_realbag2",model,X,Xtest,w,cutoff=0.85,scale_wt=200)
    #ensemble.append(xmodel)
    
    #X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=None,dropCorrelated=False,scale_data=False,clusterFeature=False)
    #basemodel = GradientBoostingClassifier(loss='deviance',n_estimators=300, learning_rate=0.08, max_depth=7,subsample=1.0,max_features=10,min_samples_leaf=20,verbose=False) 
    #model = BaggingClassifier(base_estimator=basemodel,n_estimators=40,n_jobs=8,max_features=1.0,max_samples=0.5,bootstrap=False,verbose=1)
    #xmodel = XModel("gbm_realbag3",model,X,Xtest,w,cutoff=0.85,scale_wt=200)
    #ensemble.append(xmodel)
    
    #TODO use cutoff from models, seem to be more stable than top15%    
    #TODO Bumping
    #TODO compute rapidity http://en.wikipedia.org/wiki/Pseudorapidity
    #TODO compute collinear mass or momenta 
    #TODO compute polar angle
    #TODO impute mass
    #TODO compute cutoff by fix percentage of signals -> OK
    #TODO add 1 to columns...?? SHOULD be done automatically ->NO
    #TODO SEPARATE bagging and ensemble building with oob xvalidation data ->OK
    #TODO USE cutoff optimizer within AMS
    #TODO in train ensemble load labels->voting # voting is mist
    #TODO RF2MOD modfiy weights only slighty from unity
    #TODO modify sqrt(negwts) or pow(posweihgts,n*0.5) for training with n as parameter
    
    #some info
    for m in ensemble:
	m.summary()
    return(ensemble,y)

    
def finalizeModel(m,use_proba=True):
	"""
	Make predictions and save them
	"""
	vfunc = np.vectorize(binarizeProbs)
	
	print "Make predictions and save them..."
	if use_proba:
	    #oob class predictions
	    yoob = vfunc(np.asarray(m.oob_preds),m.cutoff)
	    #probas for final test set
	    m.preds = m.classifier.predict_proba(m.Xtest)[:,1]	  	    
	    #classes for final test set
	    ypred = vfunc(np.asarray(m.preds),m.cutoff)

	else:
	    #if 0-1 outcome, only classes
	    yoob = m.oob_preds
	    m.preds = m.classifier.predict(m.Xtest)	    
	    ypred = m.preds
	    
	m.summary()	
	
	
	#put data to data.frame and save
	#OOB DATA
	m.oob_preds=pd.DataFrame(np.asarray(m.oob_preds),columns=["proba"])
	tmp=pd.DataFrame(yoob,columns=["label"])
	m.oob_preds=pd.concat([tmp, m.oob_preds],axis=1)
		
	#TESTSET prediction	
	m.preds=pd.DataFrame(np.asarray(m.preds),columns=["proba"])
	tmp=pd.DataFrame(ypred,columns=["label"])	
	m.preds=pd.concat([tmp, m.preds],axis=1)
	#save final model

	#TESTSETscores
	#OOBDATA
	allpred = pd.concat([m.preds, m.oob_preds])
	#print allpred
	#submission data is first, train data is last!
	filename="/home/loschen/Desktop/datamining-kaggle/higgs/data/"+m.name+".csv"
	print "Saving oob + predictions as csv to:",filename
	allpred.to_csv(filename,index=False)
	
	#XModel.saveModel(m,"/home/loschen/Desktop/datamining-kaggle/higgs/data/"+m.name+".pkl")
	XModel.saveCoreData(m,"/home/loschen/Desktop/datamining-kaggle/higgs/data/"+m.name+".pkl")
	return(m)
    

def createOOBdata_parallel(ensemble,ly,repeats=5,nfolds=4,n_jobs=1,verbose=False,calibrate=False):
    """
    parallel oob creation
    """

    for m in ensemble:
	use_proba = m.cutoff is not None
	
	print "Computing oob predictions for:",m.name
	print m.classifier.get_params
	oob_preds=np.zeros((m.Xtrain.shape[0],repeats))
	ams_oobscore=np.zeros(repeats)
	
	#outer loop
	for j in xrange(repeats):
	    cv = KFold(ly.shape[0], n_folds=nfolds,shuffle=True,random_state=j)
	    #cv = StratifiedKFold(ly, n_folds=nfolds,shuffle=True,random_state=None)
	    #cv = StratifiedShuffleSplit(ly, n_iter=nfolds, test_size=0.25,random_state=j)
	    print cv
	    
	    scores=np.zeros(nfolds)
	    ams_scores=np.zeros(nfolds)
	    
	    #parallel stuff
	    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
			    pre_dispatch='2*n_jobs')
	    
	    #parallel run
	    oob_pred = parallel(delayed(fit_and_score)(clone(m.classifier), m.Xtrain, ly, train, test,m.sample_weight,m.scale_wt,use_proba,m.cutoff)
			  for train, test in cv)
	    #collect oob_pred
	    oob_pred=np.array(oob_pred)[:, 0]
	    
	    for i,(train,test) in enumerate(cv):
		oob_preds[test,j] = oob_pred[i,:]
		ams_scores[i]=ams_score(ly[test],oob_preds[test,j],sample_weight=m.sample_weight[test],use_proba=use_proba,cutoff=m.cutoff)

	    auc_oobscore=roc_auc_score(ly,oob_preds[:,j])
	    ams_oobscore[j]=ams_score(ly,oob_preds[:,j],sample_weight=m.sample_weight,use_proba=use_proba,cutoff=m.cutoff)
	    
	    print "Iteration:",j,
	    
	    print " <AMS>: %0.3f (+/- %0.3f)" % (ams_scores.mean(), ams_scores.std()),
	    print " AMS,oob: %0.3f" %(ams_oobscore[j]),	    
	    print " --- AUC,oob: %0.3f" %(auc_oobscore)
	    
	    
	#simple averaging of blending
	oob_avg=np.mean(oob_preds,axis=1)
	if use_proba:
	    #probabilities
	    m.oob_preds=oob_avg		    
	    #plot deviations
	    if calibrate:
		ir = IsotonicRegression(y_min=0.0,y_max=1.0)
		ir.fit(oob_avg,ly)
		new = ir.predict(oob_avg)

		plt.hist(oob_avg,bins=50,label='orig')
		plt.hist(new,bins=50,label='calibrated',alpha=0.3)
		plt.legend()
		plt.show()
		#print dev
		#print oob_avg
		plt.plot(oob_avg,new,'r+')
		calibrated_ams=ams_score(ly,new,sample_weight=m.sample_weight,use_proba=use_proba,cutoff='compute')
		print "AMS,calibrated: %6.3f"%(calibrated_ams)
		plt.show()

	else:
	    #if 0-1 outcome, only classes
	    vfunc = np.vectorize(binarizeProbs)
	    m.oob_preds=vfunc(oob_avg,0.5)
	print "Summary: <AMS,cv>: %6.3f +- %6.3f   <AMS,oob>: %0.3f (after %d repeats) --- <AUC,oob>: %0.3f" %(ams_oobscore.mean(),ams_oobscore.std(),ams_score(ly,m.oob_preds,sample_weight=m.sample_weight,use_proba=use_proba,cutoff=m.cutoff,verbose=False),repeats,roc_auc_score(ly,m.oob_preds))
	
	#Train model with test sets
	print "Train full modell...",	
	if m.scale_wt is not None:
	    print "... with sample weights"
	    w_fit=modTrainWeights(m.sample_weight,ly,m.scale_wt)
	    m.classifier.fit(m.Xtrain,ly,w_fit)
	else:
	    print "... no sample weights"
	    m.classifier.fit(m.Xtrain,ly)
	
	m=finalizeModel(m,use_proba)
	
    return(ensemble)
    
def fit_and_score(xmodel,X,y,train,test,sample_weight=None,scale_wt=None,use_proba=True,cutoff=0.5):
    """
    Score function for parallel oob creation
    """
    Xtrain = X.iloc[train]
    Xvalid = X.iloc[test]
    ytrain = y[train]
    wtrain = sample_weight[train]
    
    if scale_wt is not None:
	wtrain_fit=modTrainWeights(wtrain,ytrain,scale_wt)
	xmodel.fit(Xtrain,ytrain,sample_weight=wtrain_fit)
	
    else:
	wtrain_fit=None
	xmodel.fit(Xtrain,ytrain)
    
    if use_proba:
	#saving out-of-bag predictions
	oob_pred = xmodel.predict_proba(Xvalid)[:,1]
	#if probabilities are available we can do the auc
	#scores[i]=roc_auc_score(ly[valid],oob_preds[valid,j])		    
    #classification    
    else:
	oob_pred = xmodel.predict(Xvalid)
    
    score=ams_score(y[test],oob_pred,sample_weight=sample_weight[test],use_proba=use_proba,cutoff=cutoff,verbose=False)
    return [oob_pred]


   
def trainEnsemble(ensemble,mode='classical',useCols=None,addMetaFeatures=False,use_proba=True,dropCorrelated=True,subfile=""):
    """
    Prepare and train the ensemble
    """
    test_indices = pd.read_csv('../datamining-kaggle/higgs/test.csv', sep=",", na_values=['?'], index_col=0).index
    y = pd.read_csv('../datamining-kaggle/higgs/training.csv', sep=",", na_values=['?'])
    y = y['Label'].str.replace(r's','1').str.replace(r'b','0')
    y = np.asarray(y.astype(float))
    
    col=['proba']
    if not use_proba:
	col=['label']
    
    
    for i,model in enumerate(ensemble):
	print "Loading model:",i," name:",model
	if i>0:
	    X = pd.read_csv("/home/loschen/Desktop/datamining-kaggle/higgs/data/"+model+".csv", sep=",")[col]
	    X.columns=[model]
	    Xall=pd.concat([Xall,X], axis=1)
	else:
	    Xall = pd.read_csv("/home/loschen/Desktop/datamining-kaggle/higgs/data/"+model+".csv", sep=",")[col]
	    Xall.columns=[model]
    
    Xtrain=Xall[len(test_indices):]
    Xtest=Xall[:len(test_indices)]
    
    #if we add metafeature we should not use aucMinimize...
    if addMetaFeatures:
	multiply=False
	(Xs,y,Xs_test,w)=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=None,dropCorrelated=False,scale_data=False,clusterFeature=False)

	Xs=Xs.loc[:,useCols]
	Xs.index=Xtrain.index
	Xs_test=Xs_test.loc[:,useCols]
	Xs_test.index=Xtest.index
		
	if multiply:
	    pass
	    #n=len(Xtrain.columns)
	    #for i,col in enumerate(useCols):
		#newcolnames=[]
		#for name in Xtrain.columns[0:n]:
		    #tmp=name+"X"+col
		    #newcolnames.append(tmp)
		#newcolnames= Xtrain.columns.append(pd.Index(newcolnames))
		##print type(Xs.ix[:,i])
		##Xtrain=pd.concat([Xtrain,Xs], axis=1)	    
		#Xtrain=pd.concat([Xtrain,Xtrain.ix[:,0:n].mul(Xs.ix[:,i],axis=0)], axis=1)
		#Xtrain.columns=newcolnames
		#Xtest=pd.concat([Xtest,Xtest.ix[:,0:n].mul(Xs_test.ix[:,i],axis=0)], axis=1)
		#Xtest.columns=newcolnames
		
	else:
	    Xtrain=pd.concat([Xtrain,Xs], axis=1)
	    Xtest=pd.concat([Xtest,Xs_test], axis=1)
	    
	    print "Dim train",Xtrain.shape
	    print "Dim test",Xtest.shape
    
    #scale data
    X_all=pd.concat([Xtest,Xtrain])   
    if dropCorrelated: X_all=removeCorrelations(X_all,0.995)
    #X_all = (X_all - X_all.min()) / (X_all.max() - X_all.min())
    Xtrain=X_all[len(test_indices):]
    Xtest=X_all[:len(test_indices)]
    #print Xtrain
    #print "New shape",Xtrain.shape
    #print Xtrain
    #print Xtest
    #Xtrain,Xtest = aucMinimize(ensemble,Xtrain,Xtest,y,test_indices,takeMean=False,removeZeroModels=0.0001)
    #Xtrain,Xtest = aucMinimize(ensemble,Xtrain,Xtest,y,test_indices,takeMean=False,removeZeroModels=0.0001)
    #auc=aucMinimize(ensemble,Xtrain,Xtest,y,test_indices,takeMean=False)
    if mode is 'classical':
	score=classicalBlend(ensemble,Xtrain,Xtest,y,test_indices,subfile=subfile)
    elif mode is 'voting':
	score=voting(ensemble,Xtrain,Xtest,y,test_indices,subfile=subfile)
    elif mode is 'mean':
	score=amsMaximize(ensemble,Xtrain,Xtest,y,test_indices,takeMean=True,subfile=subfile)
    else:
	score=amsMaximize(ensemble,Xtrain,Xtest,y,test_indices,takeMean=False,subfile=subfile)
    return(score)


def voting(ensemble,Xtrain,Xtest,y,test_indices,subfile):
    """
    Voting for simple classifiction result
    """
    
    vfunc = np.vectorize(binarizeProbs)
    
    print "Majority voting for predictions"
    weights=np.asarray(pd.read_csv('../datamining-kaggle/higgs/training.csv', sep=",", na_values=['?'], index_col=0)['Weight'])

    #for col in Xtrain.columns:
	#Xtmp = Xtrain.drop([col], axis=1)
	#oob_avg=np.mean(Xtmp,axis=1)
	#score=ams_score(y,oob_avg,sample_weight=weights,use_proba=False,cutoff='compute',verbose=False)
	#print " AMS,oob data: %0.4f omitted: %20s" % (score,col)
	##plt.hist(oob_avg,bins=50,label='oob')
	##plt.legend()
	##plt.show()
    
    for i,col in enumerate(Xtrain.columns):
	cutoff = computeCutoff(Xtrain[col].values,False)
	Xtrain[col]=vfunc(Xtrain[col].values,cutoff)
	Xtest[col]=vfunc(Xtest[col].values,cutoff)
	#oob_avg=np.mean(Xtmp,axis=1)
	score=ams_score(y,Xtrain[col].values,sample_weight=weights,use_proba=False,cutoff=cutoff,verbose=False)
	print "%4d AMS,oob data: %0.4f colum: %20s" % (i,score,col)
	#plt.hist(oob_avg,bins=50,label='oob')
	#plt.legend()
	#plt.show()
    
    
    oob_avg=np.mean(Xtrain,axis=1)
    score=ams_score(y,oob_avg,sample_weight=weights,use_proba=False,cutoff=0.5,verbose=False)
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
    


def classicalBlend(ensemble,oobpreds,testset,ly,test_indices,subfile="subXXX.csv"):
    weights=np.asarray(pd.read_csv('../datamining-kaggle/higgs/training.csv', sep=",", na_values=['?'], index_col=0)['Weight'])
    #blending
    folds=4
    cutoff_all='compute'
    scale_wt='auto'
    
    print "Blending, using general cutoff %6s, "%(str(cutoff_all)),
    if scale_wt is not None:
	print "scale_weights %6s:"%(str(scale_wt))
  
    #blender=LogisticRegression(penalty='l2', tol=0.0001, C=1.0)
    #blender = Pipeline([('filter', SelectPercentile(f_regression, percentile=25)), ('model', LogisticRegression(penalty='l2', tol=0.0001, C=0.1))])
    blender=SGDClassifier(alpha=0.1, n_iter=50,penalty='l2',loss='log')
    #blender=AdaBoostClassifier(learning_rate=0.01,n_estimators=50)
    #blender=RandomForestClassifier(n_estimators=50,n_jobs=4, max_features='auto',oob_score=False)
    #blender=ExtraTreesClassifier(n_estimators=500,max_depth=None,min_samples_leaf=5,n_jobs=4,criterion='entropy', max_features='auto',oob_score=False)
    #blender=RandomForestClassifier(n_estimators=500,max_depth=None,min_samples_leaf=10,n_jobs=1,criterion='entropy', max_features=5,oob_score=False)
    #blender=ExtraTreesRegressor(n_estimators=500,max_depth=None)
    cv = KFold(oobpreds.shape[0], n_folds=folds,random_state=123)
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
	ams_scores[i]=ams_score(ly[test],blend_oob[test],sample_weight=weights[test],use_proba=use_proba,cutoff=cutoff_all,verbose=True)
    
    
	
    print " <AUC>: %0.4f (+/- %0.4f)" % (blend_scores.mean(), blend_scores.std()),
    oob_auc=roc_auc_score(ly,blend_oob)
    print " AUC oob after blending: %0.4f" %(oob_auc)
    
    print " <AMS>: %0.4f (+/- %0.4f)" % (ams_scores.mean(), ams_scores.std()),
    oob_ams=ams_score(ly,blend_oob,sample_weight=weights,use_proba=use_proba,cutoff=cutoff_all,verbose=False)
    print " AMS oob after blending: %0.4f" %(oob_ams)
    
    if hasattr(blender,'coef_'):
      for i,model in enumerate(oobpreds.columns):
	coldata=np.asarray(oobpreds.iloc[:,i])
	ams=ams_score(ly,coldata,sample_weight=weights,use_proba=use_proba,cutoff=cutoff_all,verbose=False)
	auc=roc_auc_score(ly, coldata)
	print "%-16s ams: %5.3f auc: %4.3f coef: %4.3f" %(model,ams,auc,blender.coef_[0][i])
      print "Sum: %4.4f"%(np.sum(blender.coef_))
    #plt.plot(range(len(ensemble)),scores,'ro')
    
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
    print preds
    
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
    return(oob_ams)


def amsMaximize(ensemble,Xtrain,testset,y,test_indices,takeMean=False,removeZeroModels=0.0,subfile=""):
    weights=np.asarray(pd.read_csv('../datamining-kaggle/higgs/training.csv', sep=",", na_values=['?'], index_col=0)['Weight'])
    use_proba=True
    cutoff_all="compute"

    def fopt(params):
	# nxm  * m*1 ->n*1
	if np.isnan(np.sum(params)):
	    print "We have NaN here!!"
	    auc=0.0
	else:
	    ypred=np.dot(Xtrain,params).flatten()
	    #Force between 0 and 1
	    #ypred=sigmoid(ypred)
	    ypred = ypred/np.max(ypred)
	    #auc=roc_auc_score(y, np.dot(Xtrain,params))
	    #plt.hist(ypred)
	    #plt.show()
	    auc=ams_score(y,ypred,sample_weight=weights,use_proba=use_proba,cutoff=cutoff_all,verbose=True)
	    
	#print "score: %6.3f"%(auc)
	#print "params:",params
	return -auc
	
    #constr=[lambda x,z=i: x[z] for i in range(len(Xtrain.columns))]
    lowerbound=0.0
    upperbound=0.2
    constr=[lambda x,z=i: x[z]-lowerbound for i in range(len(Xtrain.columns))]
    constr2=[lambda x,z=i: upperbound-x[z] for i in range(len(Xtrain.columns))]
    constr=constr+constr2
    
    n_models=len(Xtrain.columns)
    x0 = np.ones((n_models, 1)) / n_models
    #x0= np.random.random_sample((n_models,1))
    
    xopt = fmin_cobyla(fopt, x0,constr,rhoend=1e-8)
    
    if takeMean:
	xopt=x0
    
    #normalize coefficient
    xopt=xopt/np.sum(xopt)
    if np.isnan(np.sum(xopt)):
	    print "We have NaN here!!"
    
    ypred=np.dot(Xtrain,xopt).flatten()
    ypred = ypred/np.max(ypred)#normalize?
    
    ymean= np.dot(Xtrain,x0).flatten()
    ymean = ymean/np.max(ymean)#normalize?
    
    oob_auc=roc_auc_score(y, ypred)
    print "->AUC,opt: %4.4f" %(oob_auc)
    auc=roc_auc_score(y,ymean)
    print "->AUC,mean: %4.4f" %(auc)
    
   
    oob_ams=ams_score(y,ypred,sample_weight=weights,use_proba=use_proba,cutoff=cutoff_all,verbose=False)
    print "->AMS,opt: %4.4f" %(oob_ams)
    ams=ams_score(y,ymean,sample_weight=weights,use_proba=use_proba,cutoff=cutoff_all,verbose=False)
    print "->AMS,mean: %4.4f" %(ams)
    
    
    zero_models=[]
    for i,model in enumerate(Xtrain.columns):
	coldata=np.asarray(Xtrain.iloc[:,i])
	auc = roc_auc_score(y, coldata)
	ams = ams_score(y,coldata,sample_weight=weights,use_proba=use_proba,cutoff=cutoff_all,verbose=False)	
	print "%4d %-48s ams: %4.3f  auc: %4.3f  coef: %4.4f cutoff: %10s" %(i+1,model,ams,auc,xopt[i],str(cutoff_all))
	if xopt[i]<removeZeroModels:
	    zero_models.append(model)
    #print "Sum: %4.4f"%(np.sum(xopt))
    
    if removeZeroModels>0.0:
	print "Dropping ",len(zero_models)," columns:",zero_models
	Xtrain=Xtrain.drop(zero_models,axis=1)
	testset=testset.drop(zero_models,axis=1)
	return (Xtrain,testset)
    
    #prediction flatten makes a n-dim vector from a nx1 vector...
    
    preds=np.dot(testset,xopt).flatten()
    preds=preds/np.max(preds)
    #preds=sigmoid(pred)
    
    #plt.hist(ypred,bins=50,label='oob')
    #plt.hist(preds,bins=50,alpha=0.3,label='pred')
    #plt.legend()
    #plt.show()
    
    print preds.shape
    
    #preds=pd.DataFrame(preds,columns=["label"],index=test_indices)
    if len(subfile)>1:
	testset.index=test_indices
	makePredictions(preds,testset,subfile,useProba=True,cutoff=cutoff_all)
	checksubmission(subfile)
    #preds.to_csv('/home/loschen/Desktop/datamining-kaggle/higgs/submissions/subXXXa.csv')

    return(oob_ams)
  
def selectModels(ensemble,niter=20,mode='amsMaximize',useCols=None): 
   
    randBinList = lambda n: [randint(0,1) for b in range(1,n+1)]
    auc_list=[0.0]
    ens_list=[]
    cols_list=[]
    for i in range(niter):
	print "iteration %5d/%5d, current max_score: %6.3f"%(i+1,niter,max(auc_list))
	actlist=randBinList(len(ensemble))
	actensemble=[x for x in itertools.compress(ensemble,actlist)]
	
	#actlist=randBinList(len(useCols))
	#actCols=[x for x in itertools.compress(useCols,actlist)]
	
	#print actensemble
	auc=trainEnsemble(actensemble,mode=mode,useCols=useCols,addMetaFeatures=False)
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
    print "TOP ensemble:",topens,
    #print "TOP cols",topcols
    print "TOP score: %4.4f" %(maxauc)

def selectModelsGreedy(ensemble,mode='amsMaximize',useCols=None):    
    auc_list=[0.0]
    ens_list=[]
    cols_list=[]
    for i in range(len(ensemble)):
	print "iteration %5d/%5d, current max_score: %6.3f"%(i+1,niter,max(auc_list))
	actlist=randBinList(len(ensemble))
	actensemble=[x for x in itertools.compress(ensemble,actlist)]
	
	#actlist=randBinList(len(useCols))
	#actCols=[x for x in itertools.compress(useCols,actlist)]
	
	#print actensemble
	auc=trainEnsemble(actensemble,mode=mode,useCols=useCols,addMetaFeatures=False)
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
    print "TOP ensemble:",topens,
    #print "TOP cols",topcols
    print "TOP score: %4.4f" %(maxauc)    
    
if __name__=="__main__":
    np.random.seed(123)
    ensemble,y=createModels()
    ensemble=createOOBdata_parallel(ensemble,y,repeats=1,nfolds=2,n_jobs=2) #only 1 repeat otherwise oob data averaging leads to significant variance reduction!
    #normal models
    gbm_models=["gbm1","gbm2","gbm3","gbm4","gbm5","gbm6","gbm7","gbm8","gbm9","gbm10"]#all the same but different seeds, oob is overfitted???
    del gbm_models[1]

    #approved models
    nongbmModels=["KNN1","KNN2","NB1","rf1","rf2","xrf1","ADAboost1","rf3"]
    gbm_models2=["gbm_test1","gbm_test2","gbm_test3","gbm_test4","gbm_test5","gbm_test6","gbm_test7","gbm_test8","gbm_test9","gbm_test10"]
    gbm_bag=["gbm_realbag1","gbm_realbag2","gbm_realbag3"]
    xgboost=["xgboost1","xgboost2"]
    new_feat=["gbm_newfeat1","gbm_newfeat2","gbm_newfeat3"]
    models=["gbm_newfeat3"]
    #useCols=['DER_mass_MMC']
    useCols=None
    #trainEnsemble(models,mode='mean',useCols=useCols,addMetaFeatures=False,use_proba=True,dropCorrelated=False,subfile='/home/loschen/Desktop/datamining-kaggle/higgs/submissions/sub0109b.csv')
    #selectModels(models)
    