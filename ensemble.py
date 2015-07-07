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

from FullModel import *
import itertools
from scipy.optimize import fmin,fmin_cobyla
from random import randint
import sys
from sklearn.externals.joblib import Parallel, delayed, logger
from sklearn.base import clone
from crowd import *

from sklearn import preprocessing


def createModels():
    ensemble=[]
   
    #KNN1  0.577/0.569 (SKF 5fold) PL: ~0.577
    #garbage=["<.*?>", "http", "www","img","border","style","px","margin","left", "right","font","solid","This translation tool is for your convenience only.*?Note: The accuracy and accessibility of the resulting translation is not guaranteed"]
    #stop_words = text.ENGLISH_STOP_WORDS.union(garbage)
    #(Xtrain, ytrain, Xtest,idx,_)  = prepareDataset(seed=42,nsamples=-1,doSeparateTFID=['product_title','query'],doSVDseparate=200,stop_words=stop_words,standardize=True,useOnlyTrain=False)
    #model = KNeighborsClassifier(n_neighbors=5)
    #xmodel = XModel("knn1_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,class_names=None)
    #ensemble.append(xmodel)
    
    #KNN2 0.591 (SKF 5fold) PL: ?
    #garbage=["<.*?>", "http", "www","img","border","style","px","margin","left", "right","font","solid","This translation tool is for your convenience only.*?Note: The accuracy and accessibility of the resulting translation is not guaranteed"]
    #stop_words  = text.ENGLISH_STOP_WORDS.union(garbage).union(corpus.stopwords.words('english'))
    #(Xtrain, ytrain, Xtest,idx,_)  = prepareDataset(seed=42,nsamples=-1,doSeparateTFID=['product_title','query'],doSVDseparate=200,stop_words=stop_words,standardize=True,computeFeatures=True,useOnlyTrain=True)
    #model = KNeighborsClassifier(n_neighbors=5)
    #xmodel = XModel("knn2_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,class_names=None)
    #ensemble.append(xmodel)
    
    #SVM1
    #garbage=["<.*?>", "http", "www","img","border","style","px","margin","left", "right","font","solid","This translation tool is for your convenience only.*?Note: The accuracy and accessibility of the resulting translation is not guaranteed"]
    #garbage2=['http','www','img','border','0','1','2','3','4','5','6','7','8','9','a','the']
    #stop_words = text.ENGLISH_STOP_WORDS.union(garbage).union(garbage2)
    #(Xtrain, ytrain, Xtest,idx,_)  = prepareDataset(seed=42,nsamples=-1,doSeparateTFID=['product_title','query'],doSVDseparate=200,stop_words=stop_words,standardize=True)
    #model = SVC(C=32,gamma=0.001)
    #xmodel = XModel("svm1_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,class_names=None)
    #ensemble.append(xmodel)

    #SVM2 ~ abhishek benchmark
    #stop_words = text.ENGLISH_STOP_WORDS
    #(Xtrain, ytrain, Xtest,idx,_)  = prepareDataset(seed=42,nsamples=-1,doTFID=True,concat=True,doSVD=400,stop_words=stop_words,standardize=True)
    #model = SVC(C=10,gamma=0.001)
    #xmodel = XModel("svm2_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,class_names=None)
    #ensemble.append(xmodel)
    
    #SMV3 like other benchmark #possibly overfitted? should we include desription as well?
    garbage=["<.*?>", "http", "www","img","border","style","px","margin","left", "right","font","solid","This translation tool is for your convenience only.*?Note: The accuracy and accessibility of the resulting translation is not guaranteed"]
    garbage2=['http','www','img','border','0','1','2','3','4','5','6','7','8','9','a','the']
    stop_words = text.ENGLISH_STOP_WORDS.union(garbage).union(garbage2)
    sw = []
    for stw in stop_words:
      sw.append("q"+stw)
      sw.append("z"+stw)
    stop_words = stop_words.union(sw)   
    (Xtrain, ytrain, Xtest,idx,_)  = prepareDataset(seed=42,nsamples=-1,doBenchMark=True,doTFID=True,doSVD=261,stop_words=stop_words)
    model = Pipeline([('scl', StandardScaler(copy=True, with_mean=True, with_std=True)), ('svm', SVC(C=10.0, kernel='rbf', degree=3, gamma=0.001, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None))])  
    xmodel = XModel("svm3_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,class_names=None)
    ensemble.append(xmodel)
    
    #SVM4 added features #0.657
    #(Xtrain, ytrain, Xtest,idx)  = prepareDataset(seed=42,nsamples=-1,doSeparateTFID=['product_title','query'],doSVDseparate=200,computeFeatures=True,standardize=True)
    #model = SVC(C=10,gamma='auto')
    #xmodel = XModel("svm4_r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,class_names=None)
    #ensemble.append(xmodel)
    
    #SVM5 added features like svm4 but optimzed svm parameters #0.653
    #(Xtrain, ytrain, Xtest,idx)  = prepareDataset(seed=42,nsamples=-1,doSeparateTFID=['product_title','query'],doSVDseparate=200,computeFeatures=True,standardize=True)
    #model = SVC(C=16,gamma=0.001)
    #xmodel = XModel("svm5_r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,class_names=None)
    #ensemble.append(xmodel)
    
    #RF1 + added features 0.630 (SSS 16fold) 0.633 (SKF 5fold)
    #(Xtrain, ytrain, Xtest,idx)  = prepareDataset(seed=42,nsamples=-1,doSeparateTFID=['product_title','query'],doSVDseparate=200,computeFeatures=True,computeSim=True, standardize=False,vectorizer=None,stop_words=None)
    #model =  RandomForestClassifier(n_estimators=250,max_depth=None,min_samples_leaf=1,n_jobs=1,criterion='gini', max_features=100)
    ##xmodel = XModel("rf1_r5",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,class_names=None) #0.625
    #xmodel = XModel("rf1_r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,class_names=None) #0.581 
    #ensemble.append(xmodel)
    
    #XRF1 + added features 0.649 (SSS 16fold) 0.652 (SKF 5fold) 0.631 (PL)
    #garbage=["<.*?>", "http", "www","img","border","style","px","margin","left", "right","font","solid","This translation tool is for your convenience only.*?Note: The accuracy and accessibility of the resulting translation is not guaranteed"]
    #garbage2=['http','www','img','border','0','1','2','3','4','5','6','7','8','9','a','the']
    #stop_words = text.ENGLISH_STOP_WORDS.union(garbage).union(garbage2)
    #(Xtrain, ytrain, Xtest,idx,_)  = prepareDataset(seed=42,nsamples=-1,doSeparateTFID=['product_title','query'],doSVDseparate=200,computeFeatures=True,computeSim=True, computeKaggleDistance=True, standardize=False,vectorizer=None,stop_words=stop_words)
    #model = ExtraTreesClassifier(n_estimators=500,max_depth=None,min_samples_leaf=3,n_jobs=1,criterion='gini', max_features=150)
    #xmodel = XModel("xrf1_r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,class_names=None)
    #ensemble.append(xmodel)
    
    #XRF2 wie XRF1 but useOnlytrain   0.648 (SKF 5fold) 0.63092 (PL)
    #garbage=["<.*?>", "http", "www","img","border","style","px","margin","left", "right","font","solid","This translation tool is for your convenience only.*?Note: The accuracy and accessibility of the resulting translation is not guaranteed"]
    #garbage2=['http','www','img','border','0','1','2','3','4','5','6','7','8','9','a','the']
    #stop_words = text.ENGLISH_STOP_WORDS.union(garbage).union(garbage2)
    #(Xtrain, ytrain, Xtest,idx,_)  = prepareDataset(seed=42,nsamples=-1,doSeparateTFID=['product_title','query'],doSVDseparate=200,computeFeatures=True,computeSim=True, computeKaggleDistance=True,useOnlyTrain=True,standardize=False,vectorizer=None,stop_words=stop_words)
    #model = ExtraTreesClassifier(n_estimators=500,max_depth=None,min_samples_leaf=3,n_jobs=1,criterion='gini', max_features=150)
    #xmodel = XModel("xrf2_r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,class_names=None)
    #ensemble.append(xmodel)
    
    #XRF3 wie XRF1 but useOnlytrain  SVD 20 0.651 (SKF 5fold)   0.62759 (PL)
    #stop_words = corpus.stopwords.words('english')#are thery really good?
    #computeKaggleTopics=["notebook","computer","movie","clothes","media","shoe","kitchen","car","bike","toy","phone","food","sport"]
    #(Xtrain, ytrain, Xtest,idx,_)  = prepareDataset(seed=42,nsamples=-1,doSeparateTFID=['product_title','query'],doSVDseparate=20,cleanse=False,computeFeatures=True,computeSim=True, computeKaggleDistance=True,computeKaggleTopics=computeKaggleTopics,useOnlyTrain=True,standardize=False,vectorizer=None,stop_words=stop_words)
    #model = ExtraTreesClassifier(n_estimators=500,max_depth=None,min_samples_leaf=3,n_jobs=1,criterion='gini', max_features='auto')
    #xmodel = XModel("xrf3_r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,class_names=None)
    #ensemble.append(xmodel)
    
    #XRF4 wie XRF1 but useOnlytrain  SVD 20  0.660 (SKF 5fold)  0.64464 (PL)
    #garbage=["<.*?>", "http", "www","img","border","style","px","margin","left", "right","font","solid","This translation tool is for your convenience only.*?Note: The accuracy and accessibility of the resulting translation is not guaranteed"]
    #stop_words = text.ENGLISH_STOP_WORDS.union(garbage).union(corpus.stopwords.words('english'))
    #computeKaggleTopics=["notebook","computer","movie","clothes","media","shoe","kitchen","car","bike","toy","phone","food","sport"]
    #(Xtrain, ytrain, Xtest,idx,_)  = prepareDataset(seed=42,nsamples=-1,doSeparateTFID=['product_title','query'],doSVDseparate=20,cleanse=True,computeFeatures=True,computeSim=True, computeKaggleDistance=True,computeKaggleTopics=computeKaggleTopics,useOnlyTrain=False,standardize=False,vectorizer=None,stop_words=stop_words)
    #model = ExtraTreesClassifier(n_estimators=500,max_depth=None,min_samples_leaf=3,n_jobs=1,criterion='gini', max_features=40)
    #xmodel = XModel("xrf4_br3",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,class_names=None)
    #ensemble.append(xmodel)
    
    #XRF5 somewhat different
    #garbage=["<.*?>", "http", "www","img","border","style","px","margin","left", "right","font","solid","This translation tool is for your convenience only.*?Note: The accuracy and accessibility of the resulting translation is not guaranteed"]
    #stop_words = text.ENGLISH_STOP_WORDS.union(garbage).union(corpus.stopwords.words('english'))
    #(Xtrain, ytrain, Xtest,idx,_)  = prepareDataset(seed=42,nsamples=-1,doSeparateTFID=['product_title','query'],doSVDseparate=20,cleanse=True,computeFeatures=True,computeSim=None, computeKaggleDistance=None,computeKaggleTopics=None,useOnlyTrain=False,standardize=False,vectorizer=None,stop_words=stop_words)
    #model = ExtraTreesClassifier(n_estimators=500,max_depth=None,min_samples_leaf=3,n_jobs=1,criterion='gini', max_features=40)
    #xmodel = XModel("xrf5_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,class_names=None)
    #ensemble.append(xmodel)
    
    #XGB + added features 0.0.624 (SSS 16fold) 0.632 (SKF 5fold)
    #garbage=["<.*?>", "http", "www","img","border","style","px","margin","left", "right","font","solid","This translation tool is for your convenience only.*?Note: The accuracy and accessibility of the resulting translation is not guaranteed"]
    #garbage2=['http','www','img','border','0','1','2','3','4','5','6','7','8','9','a','the']
    #stop_words = text.ENGLISH_STOP_WORDS.union(garbage).union(garbage2)
    #(Xtrain, ytrain, Xtest,idx)  = prepareDataset(seed=42,nsamples=-1,doSeparateTFID=['product_title','query'],doSVDseparate=200,computeFeatures=True,computeSim=True, standardize=True,vectorizer=None,stop_words=stop_words)
    #model = XgboostClassifier(n_estimators=200,learning_rate=0.1,max_depth=4,subsample=.5,n_jobs=1,objective='multi:softmax',eval_metric='mlogloss',booster='gbtree',silent=1)
    #xmodel = XModel("xgb1_r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,class_names=None) #0.581 
    #ensemble.append(xmodel)
    
    #XGB + added features 0.639 (SSS fold)  (SKF 5fold)
    #stop_words = corpus.stopwords.words('english')
    #computeKaggleTopics=["notebook","computer","movie","clothes","media","shoe","kitchen","car","bike","toy","phone","food","sport"]
    #(Xtrain, ytrain, Xtest,idx,_)  = prepareDataset(seed=42,nsamples=-1,doSeparateTFID=['product_title','query'],cleanse='load',doSVDseparate=15,computeFeatures=True,computeSim=False, computeKaggleDistance=True, computeKaggleTopics=computeKaggleTopics,computeWord2Vec='load', standardize=True,vectorizer=None,stop_words=stop_words)
    #model = XgboostClassifier(n_estimators=500,learning_rate=0.1,max_depth=4,subsample=.5,n_jobs=4,objective='multi:softmax',eval_metric='error',booster='gbtree',silent=1)
    #xmodel = XModel("xgb2_r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,class_names=None) #0.581 
    #ensemble.append(xmodel)
    
    #SVM6 more features		0.663 (SKF 5fold) PL=0.646
    #garbage=["<.*?>", "http", "www","img","border","style","px","margin","left", "right","font","solid","This translation tool is for your convenience only.*?Note: The accuracy and accessibility of the resulting translation is not guaranteed"]
    #garbage2=['http','www','img','border','0','1','2','3','4','5','6','7','8','9','a','the']
    #stop_words = text.ENGLISH_STOP_WORDS.union(garbage).union(garbage2)
    #(Xtrain, ytrain, Xtest,idx,_)  = prepareDataset(seed=42,nsamples=-1,doSeparateTFID=['product_title','query'],doSVDseparate=200,computeFeatures=True,computeSim=True, computeKaggleDistance=True,standardize=True)
    #model = SVC(C=16,gamma=0.001)
    #xmodel = XModel("svm6_r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,class_names=None)
    #ensemble.append(xmodel)
    
    
    #SVM7 more features		0.663 (SKF 5fold) PL=0.636...
    #stop_words = corpus.stopwords.words('english')
    #(Xtrain, ytrain, Xtest,idx,_)  = prepareDataset(seed=42,nsamples=-1,doSeparateTFID=['query','product_title'],doSVDseparate=200,useOnlyTrain=False,computeFeatures=None,computeSim=True,computeWord2Vec='load',standardize=True)
    #model = SVC(C=10,gamma=0.001)
    #xmodel = XModel("svm7_r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,class_names=None)
    #ensemble.append(xmodel)
    
    #SVM7_r1 more features	 (SKF 5fold 3repeats) PL=
    #stop_words = corpus.stopwords.words('english')
    #(Xtrain, ytrain, Xtest,idx,_)  = prepareDataset(seed=42,nsamples=-1,doSeparateTFID=['query','product_title'],doSVDseparate=200,useOnlyTrain=False,computeFeatures=True,computeSim=True,computeWord2Vec=True,standardize=True)
    #model = SVC(C=10,gamma=0.001)
    #xmodel = XModel("svm7_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,class_names=None)
    #ensemble.append(xmodel)
  
    #SVM8_r1 somewhat different	 (SKF 5fold 3repeats) PL=
    #stop_words = corpus.stopwords.words('english')
    #(Xtrain, ytrain, Xtest,idx,_)  = prepareDataset(seed=42,nsamples=-1,doSeparateTFID=['query','product_title'],doSVDseparate=261,doSVDseparate_2nd=200,useOnlyTrain=False,computeFeatures=None,computeSim=True,computeWord2Vec=None,standardize=True)
    #model = SVC(C=10,gamma=0.001)
    #xmodel = XModel("svm8_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,class_names=None)
    #ensemble.append(xmodel)
  
    
    #LSVM1 more features	 0.601 (SKF 5fold)
    #stop_words = corpus.stopwords.words('english')
    #(Xtrain, ytrain, Xtest,idx,_)  = prepareDataset(seed=42,nsamples=-1,doSeparateTFID=['product_title','query'],doSVDseparate=200,computeFeatures=True,computeSim=True, computeKaggleDistance=True,standardize=True)
    #model = LinearSVC(C=0.1)
    #xmodel = XModel("lsvm1_r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,class_names=None)
    #ensemble.append(xmodel)
    
    #logreg1 + added features 0.616/0.618 (SKF 5fold) PL:0.586/?
    #garbage=["<.*?>", "http", "www","img","border","style","px","margin","left", "right","font","solid","This translation tool is for your convenience only.*?Note: The accuracy and accessibility of the resulting translation is not guaranteed"]
    #stop_words  = text.ENGLISH_STOP_WORDS.union(garbage).union(corpus.stopwords.words('english'))
    #(Xtrain, ytrain, Xtest,idx,_)  = prepareDataset(seed=42,nsamples=-1,doSeparateTFID=['product_title','query'],doSVDseparate=300,computeFeatures=True,computeSim=True, computeKaggleDistance=True, standardize=True,vectorizer=None,stop_words=stop_words)
    #model = LogisticRegression(penalty='l2', tol=0.0001, C=1.0)
    #xmodel = XModel("logreg1_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,class_names=None)
    #ensemble.append(xmodel)
    
    #logreg2 class_weight='auto'
    #garbage=["<.*?>", "http", "www","img","border","style","px","margin","left", "right","font","solid","This translation tool is for your convenience only.*?Note: The accuracy and accessibility of the resulting translation is not guaranteed"]
    #stop_words  = text.ENGLISH_STOP_WORDS.union(garbage).union(corpus.stopwords.words('english'))
    #(Xtrain, ytrain, Xtest,idx,_)  = prepareDataset(seed=42,nsamples=-1,doSeparateTFID=['product_title','query'],doSVDseparate=300,computeFeatures=True,computeSim=True, computeKaggleDistance=True, standardize=True,vectorizer=None,stop_words=stop_words)
    #model = LogisticRegression(penalty='l2', tol=0.0001, C=1.0,class_weight='auto')
    #xmodel = XModel("logreg2_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,class_names=None)
    #ensemble.append(xmodel)
    
    #logreg2 + added features 0.593 (SSS 16fold) 0.599 (SKF 5fold) PL: 0.55495
    #stop_words = corpus.stopwords.words('english')
    #(Xtrain, ytrain, Xtest,idx,_)  = prepareDataset(seed=42,nsamples=-1,cleanse=True,doSeparateTFID=['product_title','query'],doSVDseparate=200,computeFeatures=True,computeSim=True, computeKaggleDistance=True, standardize=True,vectorizer=None,stop_words=stop_words)
    #model = LogisticRegression(penalty='l2', tol=0.0001, C=1.0,solver='lbfgs')
    #xmodel = XModel("logreg2_r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,class_names=None)
    #ensemble.append(xmodel)
   
    """
    #NN with all features  (SSS 8fold) 0.673 (SKF 5fold) PL: 0.640 bagmode - PL 0.655!!
    garbage=["<.*?>", "http", "www","img","border","style","px","margin","left", "right","font","solid","This translation tool is for your convenience only.*?Note: The accuracy and accessibility of the resulting translation is not guaranteed"]
    stop_words  = text.ENGLISH_STOP_WORDS.union(garbage).union(corpus.stopwords.words('english'))
    computeKaggleTopics=["notebook","computer","movie","clothes","media","shoe","kitchen","car","bike","toy","phone","food","sport"]
    (Xtrain, ytrain, Xtest,idx,_)  = prepareDataset(seed=42,nsamples=-1,cleanse='load',doSeparateTFID=['product_title','query'],doSVDseparate=300,useOnlyTrain=False,computeSim=True,computeFeatures=True,computeWord2Vec=True,computeKaggleDistance=True,computeKaggleTopics=computeKaggleTopics, standardize=True,stop_words=stop_words)
    model = nnet_crowd = NeuralNet(layers=[('input', layers.InputLayer),
	  ('dropout0', layers.DropoutLayer),
	  ('hidden1', layers.DenseLayer),
	  ('dropout1', layers.DropoutLayer),
	  ('hidden2', layers.DenseLayer),
	  ('dropout2', layers.DropoutLayer), 
	  ('output', layers.DenseLayer)],
	  input_shape=(None, 633),
	  dropout0_p=0.25,
	  hidden1_num_units=200,

	  dropout1_p=0.25,
	  hidden2_num_units=200,

	  dropout2_p=0.5,

	  output_num_units=4,
	  output_nonlinearity=nonlinearities.softmax,

	  update=adagrad,
	  update_learning_rate=theano.shared(float32(0.02)),
	  #update_momentum=0.9, only used with nesterov_
	  eval_size=0.0,
	  verbose=1,
	  max_epochs=80,
	  
	  on_epoch_finished=[
	  AdjustVariable('update_learning_rate', start=0.02, stop=0.01),
	  #EarlyStopping(patience=20),
	  ],
	  )  
    model = BaggingClassifier(base_estimator=model,n_estimators=1,n_jobs=1,verbose=2,random_state=None,max_samples=1.0,max_features=1.0,bootstrap=False)
    xmodel = XModel("nnet1_br3",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,class_names=None)
    ensemble.append(xmodel)
    """
    
    """
    #NN with all features except standard features
    garbage=["<.*?>", "http", "www","img","border","style","px","margin","left", "right","font","solid","This translation tool is for your convenience only.*?Note: The accuracy and accessibility of the resulting translation is not guaranteed"]
    stop_words  = text.ENGLISH_STOP_WORDS.union(garbage).union(corpus.stopwords.words('english'))
    computeKaggleTopics=["notebook","computer","movie","clothes","media","shoe","kitchen","car","bike","toy","phone","food","sport"]
    (Xtrain, ytrain, Xtest,idx,_)  = prepareDataset(seed=42,nsamples=-1,cleanse='load',doSeparateTFID=['query','product_title'],doSVDseparate=261,useOnlyTrain=True,computeSim=True,computeFeatures=None,computeWord2Vec=True,computeKaggleDistance=True,computeKaggleTopics=computeKaggleTopics, standardize=True,stop_words=stop_words)
    model = nnet_crowd = NeuralNet(layers=[('input', layers.InputLayer),
	  ('dropout0', layers.DropoutLayer),
	  ('hidden1', layers.DenseLayer),
	  ('dropout1', layers.DropoutLayer),
	  ('hidden2', layers.DenseLayer),
	  ('dropout2', layers.DropoutLayer), 
	  ('output', layers.DenseLayer)],
	  input_shape=(None, 543),
	  dropout0_p=0.25,
	  hidden1_num_units=200,

	  dropout1_p=0.25,
	  hidden2_num_units=200,

	  dropout2_p=0.5,

	  output_num_units=4,
	  output_nonlinearity=nonlinearities.softmax,

	  update=adagrad,
	  update_learning_rate=theano.shared(float32(0.02)),
	  #update_momentum=0.9, only used with nesterov_
	  eval_size=0.0,
	  verbose=1,
	  max_epochs=80,
	  
	  on_epoch_finished=[
	  AdjustVariable('update_learning_rate', start=0.02, stop=0.01),
	  #EarlyStopping(patience=20),
	  ],
	  )  
    model = BaggingClassifier(base_estimator=model,n_estimators=1,n_jobs=1,verbose=2,random_state=None,max_samples=1.0,max_features=1.0,bootstrap=False)
    xmodel = XModel("nnet2_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,class_names=None)
    ensemble.append(xmodel)
    """
    
    """
    #NN with all features except standard features
    stop_words = text.ENGLISH_STOP_WORDS.union(corpus.stopwords.words('english'))
    (Xtrain, ytrain, Xtest,idx,_)  = prepareDataset(seed=42,nsamples=-1,cleanse='load',doSeparateTFID=['query','product_title'],doSVDseparate=261,doSVDseparate_2nd=300,  useOnlyTrain=False,computeSim=True,computeFeatures=True,computeWord2Vec=None,computeKaggleDistance=None, standardize=True,stop_words=stop_words)
    model = nnet_crowd = NeuralNet(layers=[('input', layers.InputLayer),
	  ('dropout0', layers.DropoutLayer),
	  ('hidden1', layers.DenseLayer),
	  ('dropout1', layers.DropoutLayer),
	  ('hidden2', layers.DenseLayer),
	  ('dropout2', layers.DropoutLayer), 
	  ('output', layers.DenseLayer)],
	  input_shape=(None, 575),
	  dropout0_p=0.25,
	  hidden1_num_units=250,

	  dropout1_p=0.5,
	  hidden2_num_units=250,

	  dropout2_p=0.25,

	  output_num_units=4,
	  output_nonlinearity=nonlinearities.softmax,

	  update=adagrad,
	  update_learning_rate=theano.shared(float32(0.02)),
	  #update_momentum=0.9, only used with nesterov_
	  eval_size=0.0,
	  verbose=1,
	  max_epochs=50,
	  
	  on_epoch_finished=[
	  AdjustVariable('update_learning_rate', start=0.02, stop=0.01),
	  #EarlyStopping(patience=20),
	  ],
	  )  
    model = BaggingClassifier(base_estimator=model,n_estimators=1,n_jobs=1,verbose=2,random_state=None,max_samples=1.0,max_features=1.0,bootstrap=False)
    xmodel = XModel("nnet3_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,class_names=None)
    ensemble.append(xmodel)
    """
    
    
    """
    #NN with all features (SSS 8fold)  (SKF 5fold) PL: 
    garbage=["<.*?>", "http", "www","img","border","style","px","margin","left", "right","font","solid","This translation tool is for your convenience only.*?Note: The accuracy and accessibility of the resulting translation is not guaranteed"]
    stop_words  = text.ENGLISH_STOP_WORDS.union(garbage).union(corpus.stopwords.words('english'))
    computeKaggleTopics=["notebook","computer","movie","clothes","media","shoe","kitchen","car","bike","toy","phone","food","sport"]
    (Xtrain, ytrain, Xtest,idx,_)  = prepareDataset(seed=42,nsamples=-1,cleanse='load',doSeparateTFID=['product_title','query'],doSVDseparate=300,useOnlyTrain=False,computeSim=True,computeFeatures=True,computeWord2Vec=True,computeKaggleDistance=True,computeKaggleTopics=computeKaggleTopics, standardize=True,stop_words=stop_words)
    model = nnet_crowd = NeuralNet(layers=[('input', layers.InputLayer),
	  ('dropout0', layers.DropoutLayer),
	  ('hidden1', layers.DenseLayer),
	  ('dropout1', layers.DropoutLayer),
	  ('hidden2', layers.DenseLayer),
	  ('dropout2', layers.DropoutLayer), 
	  ('output', layers.DenseLayer)],
	  input_shape=(None, 633),
	  dropout0_p=0.25,
	  hidden1_num_units=200,

	  dropout1_p=0.25,
	  hidden2_num_units=200,

	  dropout2_p=0.5,

	  output_num_units=4,
	  output_nonlinearity=nonlinearities.softmax,

	  update=adagrad,
	  update_learning_rate=theano.shared(float32(0.02)),
	  #update_momentum=0.9, only used with nesterov_
	  eval_size=0.0,
	  verbose=1,
	  max_epochs=80,
	  
	  on_epoch_finished=[
	  AdjustVariable('update_learning_rate', start=0.02, stop=0.01),
	  #EarlyStopping(patience=20),
	  ],
	  ) 
    
    model = BaggingClassifier(base_estimator=model,n_estimators=10,n_jobs=1,verbose=2,random_state=None,max_samples=1.0,max_features=1.0,bootstrap=False)
    xmodel = XModel("bagnet1_br3",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,class_names=None)
    ensemble.append(xmodel)
    """
    
    #some info
    for m in ensemble:
	m.summary()
    return(ensemble)

    
def finalizeModel(m,use_proba=True):
	"""
	Make predictions and save them
	"""
	print "Make predictions and save them..."
	#oob from crossvalidation
	yoob = m.oob_preds
	#final prediction
	ypred = m.preds
	    
	m.summary()	
	
	#put data to data.frame and save
	#OOB DATA
	m.oob_preds=pd.DataFrame(np.asarray(m.oob_preds),columns=['prediction'])
		
	#TESTSET prediction	
	m.preds=pd.DataFrame(np.asarray(m.preds),columns=['prediction'])
	
	#save final model
	allpred = pd.concat([m.preds, m.oob_preds])
	#submission data is first, train data is last!
	filename="./data/"+m.name+".csv"
	print "Saving oob + predictions as csv to:",filename
	allpred.to_csv(filename,index=False)
	
	#XModel.saveModel(m,"/home/loschen/Desktop/datamining-kaggle/higgs/data/"+m.name+".pkl")
	XModel.saveCoreData(m,"./data/"+m.name+".pkl")
	return(m)
    

def createOOBdata_parallel(ensemble,repeats=2,nfolds=5,n_jobs=1,score_func='log_loss',verbose=False,calibrate=False,bagmode=False,use_proba=True):
    """
    parallel oob creation
    """
    global funcdict

    for m in ensemble:
	print "Computing oob predictions for:",m.name
	print m.classifier.get_params
	if m.class_names is not None:
	    n_classes = len(m.class_names)
	else:
	    n_classes = 1
	print "n_classes",n_classes
	
	oob_preds=np.zeros((m.ytrain.shape[0],n_classes,repeats))
	preds = np.zeros((m.Xtest.shape[0],n_classes,repeats))
	
	ly=m.ytrain
	oobscore=np.zeros(repeats)
	maescore=np.zeros(repeats)
	
	#outer loop
	for j in xrange(repeats):
	    cv = StratifiedKFold(ly, n_folds=nfolds,shuffle=True,random_state=None)
	    #cv = StratifiedShuffleSplit(ly, n_iter=nfolds, test_size=0.25,random_state=j)
	    
	    scores=np.zeros(len(cv))
	    scores2=np.zeros(len(cv))
	    
	    #parallel stuff
	    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
			    pre_dispatch='2*n_jobs')
	    
	    #parallel run, returns a list of oob predictions
	    results = parallel(delayed(fit_and_score)(clone(m.classifier), m.Xtrain, ly, train, test,use_proba=use_proba,returnModel=True)
			  for train, test in cv)

	    for i,(train,test) in enumerate(cv):
		oob_pred,cv_model = results[i]
	        oob_pred = oob_pred.reshape(oob_pred.shape[0],n_classes)
		oob_preds[test,:,j] = oob_pred
		
		scores[i]=funcdict[score_func](ly[test],oob_preds[test,:,j])
		
		if bagmode:
		  print "Using cv models for test set(bagmode)..."
		  if use_proba:
		    p = cv_model.predict_proba(m.Xtest)
		    p = p.reshape(p.shape[0],n_classes)
		    preds[:,:,j] = p
		  else:
		    p = cv_model.predict(m.Xtest)
		    p = p.reshape(p.shape[0],n_classes)
		    preds[:,:,j] = p
		    
		#print "Fold %d - score:%0.3f " % (i,scores[i])
		#scores_mae[i]=funcdict['mae'](ly[test],oob_preds[test,j])

	    oobscore[j]=funcdict[score_func](ly,oob_preds[:,:,j])
	    #maescore[j]=funcdict['mae'](ly,oob_preds[:,j])
	    
	    print "Iteration:",j,
	    print " <score>: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std()),
	    print " score,oob: %0.3f" %(oobscore[j])
	    #print " ## <mae>: %0.3f (+/- %0.3f)" % (scores_mae.mean(), scores_mae.std()),
	    #print " score3,oob: %0.3f" %(maescore[j])
	    
	#simple averaging of blending
	m.oob_preds=np.mean(oob_preds,axis=2)
	#print m.oob_preds[:10]
	if not use_proba:
	  print "Warning: rounding oob data!"
	  m.oob_preds = np.round(m.oob_preds)
	
	#print m.oob_preds[:10]

	score_oob = funcdict[score_func](ly,m.oob_preds)
	print "Summary: <score,oob>: %6.3f +- %6.3f   score,oob-total: %0.3f (after %d repeats)\n" %(oobscore.mean(),oobscore.std(),score_oob,repeats)
	
	if not bagmode:
	    #Train full model on total train data
	    print "Train full model on whole train set..."	
	    if m.sample_weight is not None:
		print "... with sample weights"
		m.classifier.fit(m.Xtrain,ly,m.sample_weight)
	    else:
		m.classifier.fit(m.Xtrain,ly)
	    
	    if use_proba:
	      m.preds = m.classifier.predict_proba(m.Xtest)
	    else:
	      m.preds = m.classifier.predict(m.Xtest)
	  
	else:
	      print "bagmode: averaging all cv classifier results"
	      #print preds[:10]
	      m.preds= np.mean(preds,axis=2)
	      m.preds[:10]
	      if not use_proba:
		print "Warning: rounding test data!"
		m.preds = np.round(m.preds)
	      #print m.preds[:10]
	
	m = finalizeModel(m,use_proba=use_proba)
	
    return(ensemble)
    
def fit_and_score(xmodel,X,y,train,valid,sample_weight=None,scale_wt=None,use_proba=False,returnModel=True):
    """
    Score function for parallel oob creation
    """
    if isinstance(X,pd.DataFrame): 
	Xtrain = X.iloc[train]
	Xvalid = X.iloc[valid]
    else:
	Xtrain = X[train]
	Xvalid = X[valid]
	
    ytrain = y[train]
    
    if sample_weight is not None: 
	wtrain = sample_weight[train]
	xmodel.fit(Xtrain,ytrain,sample_weight=wtrain)
    else:
	xmodel.fit(Xtrain,ytrain)
    
    if use_proba:
	  #saving out-of-bag predictions
	  local_pred = xmodel.predict_proba(Xvalid)
	  #prediction for test set
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
    if returnModel:
       return local_pred, xmodel
    else:
       return local_pred

def trainEnsemble_multiclass(ensemble,mode='linear',score_func='log_loss',useCols=None,addMetaFeatures=False,use_proba=True,dropCorrelated=False,skipCV=False,subfile=""):
    """
    Train the ensemble
    """
    basedir="./data/"

    for i,model in enumerate(ensemble):
	
	print "Loading model:",i," name:",model
	xmodel = XModel.loadModel(basedir+model)
	class_names = xmodel.class_names
	if class_names is None:
	  class_names = ['Class']
	print "OOB data:",xmodel.oob_preds.shape
	print "pred data:",xmodel.preds.shape
	print "y train:",xmodel.ytrain.shape
	
	if i>0:
	    xmodel.oob_preds.columns = [model+"_"+n for n in class_names]
	    Xtrain = pd.concat([Xtrain,xmodel.oob_preds], axis=1)
	    Xtest = pd.concat([Xtest,xmodel.preds], axis=1)
	    
	else:
	    Xtrain = xmodel.oob_preds
	    Xtest = xmodel.preds
	    y = xmodel.ytrain
	    Xtrain.columns = [model+"_"+n for n in class_names]

    print Xtrain.columns
    print Xtrain.shape
    
    print "spearman-correlation:\n",Xtrain.corr(method='spearman')
    print "pearson-correlation :\n",Xtrain.corr(method='pearson')
    #raw_input()
    #print Xtrain.describe()
    print Xtest.shape
    #print Xtest.describe()
   
    if mode is 'classical':
	results=classicalBlend(ensemble,Xtrain,Xtest,y,score_func=score_func,use_proba=use_proba,skipCV=skipCV,subfile=subfile)
    elif mode is 'mean':
	results=linearBlend_multiclass(ensemble,Xtrain,Xtest,y,score_func=score_func,takeMean=True,subfile=subfile)
    elif mode is 'voting':
        results=voting_multiclass(ensemble,Xtrain,Xtest,y,score_func=score_func,n_classes=1,subfile=subfile)
    else:
	results=linearBlend_multiclass(ensemble,Xtrain,Xtest,y,score_func=score_func,takeMean=False,subfile=subfile)
    return(results)

def voting_multiclass(ensemble,Xtrain,Xtest,y,n_classes=9,use_proba=False,score_func='log_loss',plotting=True,subfile=None):
    """
    Voting for multi classifiction result
    """
    if use_proba:
      print "Majority voting for predictions using proba"
      voter = np.reshape(Xtrain.values,(Xtrain.shape[0],-1,n_classes)).swapaxes(0,1)

      for model in voter:
	  max_idx=model.argmax(axis=1)
	  for row,idx in zip(model,max_idx):
	      row[:]=0.0
	      row[idx]=1.0
      
      voter = voter.mean(axis=0)
      print voter
      print voter.shape
    else:
      print "Majority voting for predictions"
      #assuming all classes are predicted
      if Xtrain.shape[1]%2==0:
	  print "Warning: Even number of voters..."
      
      classes = np.unique(Xtrain.values)
      
      votes_train = np.zeros((Xtrain.shape[0],classes.shape[0]))
      votes_test = np.zeros((Xtest.shape[0],classes.shape[0]))
      
      for i,c in enumerate(classes):
	  votes_train[:,i] = np.sum(Xtrain.values==c,axis=1)
	  votes_test[:,i] = np.sum(Xtest.values==c,axis=1)
      
      votes_train = np.argmax(votes_train,axis=1)
      votes_test = np.argmax(votes_test,axis=1)
      
      encoder= preprocessing.LabelEncoder()
      encoder.fit(y)
      ypred = encoder.inverse_transform(votes_train)
      preds = encoder.inverse_transform(votes_test)
      
      score=funcdict[score_func](y, ypred)
      print score_func+": %0.3f" %(score)
      
      
    if subfile is not None:
	analyze_predictions(ypred,preds)
	makePredictions(None,Xtest=preds,filename=subfile)
	
	if plotting:
	  plt.hist(ypred,bins=50,alpha=0.3,label='oob')
	  plt.hist(preds,bins=50,alpha=0.3,label='pred')
	  plt.legend()
	  plt.show()
	
    else:
	return score
      
      
      
def analyze_predictions(ypred,preds):
    ypred = ypred.astype(int)
    oob_total = (ypred<1).sum()+(ypred==2).sum()+(ypred==3).sum()+(ypred==4).sum()
    pred_total = (preds<1).sum()+(preds==2).sum()+(preds==3).sum()+(preds==4).sum()
    oob0 = (ypred<1).sum()
    oob1 = (ypred==1).sum()
    oob2 = (ypred==2).sum()
    oob3 = (ypred==3).sum()
    oob4 = (ypred==4).sum()
    
    pred0 = (preds<1).sum()
    pred1 = (preds==1).sum()
    pred2 = (preds==2).sum()
    pred3 = (preds==3).sum()
    pred4 = (preds==4).sum()
	      
    print "oob : 0:%d 1:%d (%.2f) 2:%s (%.2f) 3:%d (%.2f) 4:%d (%.2f) (total: %d)"%(oob0, oob1,oob1/float(oob_total),oob2,oob2/float(oob_total),oob3,oob3/float(oob_total),oob4,oob4/float(oob_total),oob_total)
    print "pred: 0:%d 1:%d (%.2f) 2:%s (%.2f) 3:%d (%.2f) 4:%d (%.2f) (total: %d)"%(pred0 ,pred1,pred1/float(pred_total),pred2,pred2/float(pred_total),pred3,pred3/float(pred_total),pred4,pred4/float(pred_total),pred_total)
    plt.hist(ypred,bins=50,alpha=0.3,label='oob')
    plt.hist(preds,bins=50,alpha=0.3,label='pred')
    plt.legend()
    plt.show()
  

def classicalBlend(ensemble,oobpreds,testset,ly,use_proba=True,score_func='log_loss',subfile="",cv=5,skipCV=False):
    """
    Blending using sklearn classifier
    """
     
    #blender=LogisticRegression(penalty='l2', tol=0.0001, C=1)
    #blender = Pipeline([('filter', SelectPercentile(f_regression, percentile=25)), ('model', LogisticRegression(penalty='l2', tol=0.0001, C=0.1))])
    #blender=RandomForestClassifier(n_estimators=500,n_jobs=4, max_features='auto',oob_score=False,min_samples_leaf=10,max_depth=None)
    #blender = CalibratedClassifierCV(blender, method='isotonic', cv=3)
    #blender=ExtraTreesClassifier(n_estimators=300,max_depth=None,min_samples_leaf=7,n_jobs=4,criterion='gini', max_features=3,oob_score=False)
    #blender = Pipeline([('ohc', OneHotEncoder(sparse=False)), ('model',ExtraTreesClassifier(n_estimators=300,max_depth=None,min_samples_leaf=7,n_jobs=4,criterion='gini', max_features=3,oob_score=False))])
    #blender = Pipeline([('ohc', OneHotEncoder(sparse=False)), ('model',LogisticRegression(penalty='l2', tol=0.0001, C=1))])
    #blender=RandomForestClassifier(n_estimators=100,max_depth=None,min_samples_leaf=10,n_jobs=1,criterion='entropy', max_features=5,oob_score=False)
    blender = XgboostClassifier(n_estimators=200,learning_rate=0.03,max_depth=2,subsample=.5,n_jobs=1,objective='multi:softmax',eval_metric='error',booster='gbtree',silent=1)
    if not skipCV:
	#blender = CalibratedClassifierCV(baseblender, method='sigmoid', cv=3)
	cv = StratifiedKFold(ly, n_folds=5,shuffle=True)
	#cv=StratifiedShuffleSplit(ly,16,test_size=0.3)
	#score_func = make_scorer(funcdict[score_func], greater_is_better = True)
	parameters = {'n_estimators':[200],'max_depth':[2],'learning_rate':[0.03],'subsample':[0.5]}#XGB
	#parameters = {'n_estimators':[100,500],'max_features':[4],'min_samples_leaf':[1,5],'criterion':['entropy']}#XGB
	#blender=makeGridSearch(blender,oobpreds,ly,n_jobs=2,refit=True,cv=cv,scoring=score_func,random_iter=-1,parameters=parameters)
	
	blend_scores=np.zeros(len(cv))
	n_classes = 1 #oobpreds.shape[1]/len(ensemble)
	blend_oob=np.zeros((oobpreds.shape[0],n_classes))
	print blender
	for i, (train, test) in enumerate(cv):
	    Xtrain = oobpreds.iloc[train]
	    Xtest = oobpreds.iloc[test]
	    blender.fit(Xtrain, ly[train])	
	    if use_proba:
		blend_oob[test] = blender.predict_proba(Xtest)
	    else:
		print "Warning: Using predict, no proba!"
		blend_oob[test] = blender.predict(Xtest).reshape(blend_oob[test].shape)
	    blend_scores[i]=funcdict[score_func](ly[test],blend_oob[test])
	    print "Fold: %3d <%s>: %0.6f" % (i,score_func,blend_scores[i])
	
	print " <"+score_func+">: %0.6f (+/- %0.4f)" % (blend_scores.mean(), blend_scores.std()),
	oob_auc=funcdict[score_func](ly,blend_oob)
	print " "+score_func+": %0.6f" %(oob_auc)
	
	if hasattr(blender,'coef_'):
	  print "%-16s %10s %10s" %("model",score_func,"coef")
	  idx = 0
	  for i,model in enumerate(ensemble):
	    idx_start = n_classes*i
	    idx_end = n_classes*(i+1)
	    coldata=np.asarray(oobpreds.iloc[:,idx_start:idx_end])
	    score=funcdict[score_func](ly, coldata)
	    print "%-16s %10.3f%10.3f" %(model,score,blender.coef_[0][i])
	  print "sum coef: %4.4f"%(np.sum(blender.coef_))
	#plt.plot(range(len(ensemble)),scores,'ro')
    
    if len(subfile)>1:
	#Prediction
	print "Make final ensemble prediction..."
	#make prediction for each classifiers   
	preds=blender.fit(oobpreds,ly)
	#blend results
	if use_proba:
	  preds=blender.predict_proba(testset)
	else:
	  preds=blender.predict(testset)
	#print preds
	makePredictions(blender,testset,filename=subfile)
	
	if not skipCV: plt.hist(blend_oob,bins=50,label='oob')
	plt.hist(preds,bins=50,alpha=0.3,label='pred')
	analyze_predictions(blend_oob,preds)
	plt.legend()
	plt.show()
	
    return(blend_scores.mean())



def multiclass_mult(Xtrain,params,n_classes):
    """
    Multiplication rule for multiclass models
    """
    ypred = np.zeros((len(params),Xtrain.shape[0],n_classes))
    for i,p in enumerate(params):
		idx_start = n_classes*i
		idx_end = n_classes*(i+1)
		ypred[i] = Xtrain.iloc[:,idx_start:idx_end]*p
    ypred = np.mean(ypred,axis=0)
    return ypred

def blend_mult(Xtrain,params,n_classes=None):
    if n_classes <2:
	return np.dot(Xtrain,params)
    else: 
	return multiclass_mult(Xtrain,params,n_classes)

def linearBlend_multiclass(ensemble,Xtrain,Xtest,y,score_func='log_loss',greater_is_better=True,use_proba=False,normalize=False,removeZeroModels=-1,takeMean=False,alpha=None,subfile=None,plotting=False):
    """
    Blending for multiclass systems
    """
    def fopt(params):
	# nxm  * m*1 ->n*1
	if np.isnan(np.sum(params)):
	    print "We have NaN here!!"
	    score=0.0
	else:
	    #print "params:",params
	    #print Xtrain[:15]
	    #print params
	    ypred = blend_mult(Xtrain,params,n_classes)
	    #print ypred
	    #print ypred[:15]
	    if not use_proba: ypred = np.round(ypred).astype(int)
	    #print ypred[:15]
	    #print y[:15]
	    #print ypred
	    score=funcdict[score_func](y,ypred)
	    #print "score: %8.3f"%(score)
	    #raw_input()
	    #regularization
	    if alpha is not None:
	      penalty=alpha*np.sum(np.square(params))
	      print "orig score:%8.3f"%(score),
	      score=score-penalty
	      print " - Regularization - alpha: %8.3f penalty: %8.3f regularized score: %8.3f"%(alpha,penalty,score)
	    if greater_is_better: score = -1*score
	return score

    y = np.asarray(y)
    n_models=len(ensemble)
    n_classes = Xtrain.shape[1]/len(ensemble)
    
    lowerbound=0.0
    upperbound=0.5
    constr=None
    constr=[lambda x,z=i: x[z]-lowerbound for i in range(n_models)]
    constr2=[lambda x,z=i: upperbound-x[z] for i in range(n_models)]
    constr=constr+constr2

    x0 = np.ones((n_models, 1)) / float(n_models)
    
    if not takeMean:
      xopt = fmin_cobyla(fopt, x0,constr,rhoend=1e-7,maxfun=1000)
    else:
      xopt = x0
	

    
    #normalize coefficient
      if normalize: 
	  xopt=xopt/np.sum(xopt)
	  print "Normalized coefficients:",xopt

    if np.isnan(np.sum(xopt)):
	    print "We have NaN here!!"
    
    ypred=blend_mult(Xtrain,xopt,n_classes)
    #ymean= blend_mult(Xtrain,x0,n_classes).flatten()
    ymean=np.mean(Xtrain.values,axis=1)
    
    if takeMean:
      print "Taking the mean..."
      ypred = ymean
    
    
    
    #print ymean[:10]
    if not use_proba:
      ymean = np.round(ymean+1E-2).astype(int)
      ypred = np.round(ypred+1E-6).astype(int)
      
    print "ypred:",ypred.sum()
    print "ypred:",ypred
    print "ymean:",ymean.sum()
    print "ymean:",ymean
    
    score=funcdict[score_func](y,ymean)
    print "->score,mean: %4.4f" %(score)
    oob_score=funcdict[score_func](y,ypred)
    print "->score,opt: %4.4f" %(oob_score)
       
    zero_models=[]
    print "%4s %-48s %6s %6s" %("nr","model","score","coeff")
    for i,model in enumerate(ensemble):
	idx_start = n_classes*i
	idx_end = n_classes*(i+1)
	coldata=np.asarray(Xtrain.iloc[:,idx_start:idx_end])
	score = funcdict[score_func](y,coldata)	
	print "%4d %-48s %6.3f %6.3f" %(i+1,model,score,xopt[i])
	if xopt[i]<removeZeroModels:
	    zero_models.append(model)
    print "##sum coefficients: %4.4f"%(np.sum(xopt))
    print "training -  max: %4.2f mean: %4.2f median: %4.2f min: %4.2f"%(np.amax(ypred),ypred.mean(),np.median(ypred),np.amin(ypred))
    
    if removeZeroModels>0.0:
	print "Dropping ",len(zero_models)," columns:",zero_models
	Xtrain=Xtrain.drop(zero_models,axis=1)
	Xtest=Xtest.drop(zero_models,axis=1)
	return (Xtrain,Xtest)
    
    #prediction flatten makes a n-dim row vector from a nx1 column vector...
    preds = blend_mult(Xtest,xopt,n_classes).flatten()
    if not use_proba: preds = np.round(preds).astype(int)

    if subfile is not None:
	print "predictions - max: %4.2f mean: %4.2f median: %4.2f min: %4.2f"%(np.amax(preds),preds.mean(),np.median(preds),np.amin(preds))
	analyze_predictions(ypred,preds)
	plt.hist(ypred,bins=50,alpha=0.3,label='oob')
	plt.hist(preds,bins=50,alpha=0.3,label='pred')
	plt.legend()
	plt.show()
	makePredictions(None,Xtest=preds,filename=subfile)
    else:
	return oob_score
      
  
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

def selectModelsGreedy(ensemble,startensemble=[],niter=2,mode='classical',useCols=None,dropCorrelated=False,greater_is_better=False):    
    """
    Select best models in a greedy forward selection
    """
    topensemble=startensemble
    score_list=[]
    ens_list=[]
    if greater_is_better:
	bestscore=0.0
    else:
	bestscore=1E15
    for i in range(niter):
	if greater_is_better:
	    maxscore=0.0
	else:
	    maxscore=1E15
	topidx=-1
	for j in range(len(ensemble)):
	    if ensemble[j] not in topensemble:
		actensemble=topensemble+[ensemble[j]]
	    else:
		continue
	    
	    #score=trainEnsemble(actensemble,mode=mode,useCols=useCols,addMetaFeatures=False,dropCorrelated=dropCorrelated)
	    #score=trainEnsemble_multiclass(actensemble,mode=mode,useCols=None,use_proba=False)
	    score = trainEnsemble_multiclass(actensemble,mode=mode,score_func='quadratic_weighted_kappa',use_proba=False,subfile=None)
	    print "##(Current top score: %4.4f | overall best score: %4.4f) actual score: %4.4f  - " %(maxscore,bestscore,score),
	    if greater_is_better:
		if score>maxscore:
		    maxscore=score
		    topidx=j
	    else:
		if score<maxscore:
		    maxscore=score
		    topidx=j
	    
	#pick best set
	#if not maxscore+>bestscore:
	#    print "Not gain in score anymore, leaving..."
	#    break
	topensemble.append(ensemble[topidx])
	print "TOP score: %4.4f" %(maxscore),
	print " - actual ensemble:",topensemble
	score_list.append(maxscore)
	ens_list.append(list(topensemble))
	if greater_is_better:
	  if maxscore>bestscore:
	      bestscore=maxscore
	else:
	  if maxscore<bestscore:
	      bestscore=maxscore
    
    for ens,score in zip(ens_list,score_list):	
	print "SCORE: %4.4f" %(score),
	print ens
	
    plt.plot(score_list)
    plt.show()
    return topensemble

 
def blendSubmissions(fileList,coefList):
    """
    Simple blend dataframes from fileList
    """
    pass
   

if __name__=="__main__":
    #ensemble=createModels()
    #ensemble=createOOBdata_parallel(ensemble,repeats=1,nfolds=5,n_jobs=5,use_proba=False,score_func='quadratic_weighted_kappa',bagmode=True) #oob data averaging leads to significant variance reduction
    #old_models=['knn1_r1','svm1_r1','svm2_r1','svm3_r1','svm4_r1','svm5_r1','rf1_r1','xgb1_r1','xrf1_r1','xrf3_r1','svm6_r1','logreg1_r1','lsvm1_r1','logreg2_r1','bagnet1_r1','nnet1_r1','xrf4_r1']
    #opt_models = ['svm6_r1', 'xrf1_r1', 'svm4_r1', 'xrf2_r1', 'logreg1_r1', 'knn1_r1', 'svm1_r1', 'xrf3_r1', 'svm3_r1']#greedy linear    
    new_models=['knn1_r1','knn2_r1','bagnet1_r1','nnet1_r1','xrf3_r1','xrf4_r1','svm7_r1','logreg1_r1','lsvm1_r1','xgb2_r1']
    br3_models=['nnet1_br3','svm7_br3','xrf4_br3','logreg1_br3']
    br1_models=['knn1_br1','svm1_br1','svm2_br1','svm8_br1','xrf5_br1','logreg1_br1','logreg2_br1','nnet2_br1','nnet3_br1']
    #greedy_opt=['nnet2_br1', 'xrf5_br1']

    models = new_models + br3_models + br1_models
    #models = ['svm7_br3','xrf4_br3','nnet1_br3']
    #models = opt_models2
    useCols=None
    trainEnsemble_multiclass(models,mode='linear',score_func='quadratic_weighted_kappa',useCols=None,addMetaFeatures=False,use_proba=False,dropCorrelated=False,subfile='./submissions/sub06072015e.csv')
    #selectModelsGreedy(models,startensemble=['nnet2_br1'],niter=10,mode='mean',greater_is_better=True)
   