#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Chrissly31415 August 2013


from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import scipy as sp
import numpy as np
import pandas as pd
import pickle

class XModel:
   """
   class holds classifier plus train and test set for later use in ensemble building
   Wrapper for ensemble building
   """
   modelcount = 0

   def __init__(self, name,classifier,Xtrain,Xtest,sample_weight=None,cutoff=None,scale_wt=1.0):
      self.name = name
      self.classifier = classifier
      self.Xtrain=Xtrain
      self.Xtest=Xtest
       
      if isinstance(Xtrain,sp.sparse.csr.csr_matrix) or isinstance(Xtrain,sp.sparse.csc.csc_matrix):
	self.sparse=True
      else:
	self.sparse=False
      self.sample_weight=sample_weight
      
      self.oob_preds=np.zeros((Xtrain.shape[0],1))
      self.preds=np.zeros((Xtest.shape[0],1))
      self.ytrain=np.zeros((Xtrain.shape[0],1))
      
      self.cutoff=cutoff
      if cutoff is not None:
	self.use_proba=True
      self.scale_wt=scale_wt
	
      XModel.modelcount += 1

   def summary(self):
      print ">>Name<<     :" ,  self.name
      print "classifier   :" , type(self.classifier)
      print "Train data   :" , self.Xtrain.shape,
      print " type         :" , type(self.Xtrain)
      print "Test data    :" , self.Xtest.shape,
      print " type         :" , type(self.Xtest)
      if self.sample_weight is not None:
	  print "sample_weight :" , self.sample_weight.shape,
	  print " type         :" , type(self.sample_weight)
      #print "sparse data  :" , self.sparse 
      print "proba cutoff  :" , self.cutoff
      print "scale weight  :" , self.scale_wt
      
      print "predictions mean %6.3f :" %(np.mean(self.preds)),
      print " Dim:", self.preds.shape
      print "oob preds mean %6.3f  :" %(np.mean(self.oob_preds)),
      print " Dim:", self.oob_preds.shape
      
      
      
   def __repr__(self):
      self.summary()
   
   #static function for saving
   def saveModel(xmodel,filename):
      if not hasattr(xmodel,'xgboost_model'):
	  pickle_out = open(filename.replace('.csv',''), 'wb')
	  pickle.dump(xmodel, pickle_out)
	  pickle_out.close()
	  
  
   #static function for saving only the important parameters
   def saveCoreData(xmodel,filename):
      if not hasattr(xmodel,'xgboost_model'):
	  #reset not needed stuff
	  xmodel.classifier=None
	  xmodel.Xtrain=None
	  xmodel.Xtest=None
	  xmodel.sample_weight=None
	  #keep only parameters and predictions
	  pickle_out = open(filename.replace('.csv',''), 'wb')
	  pickle.dump(xmodel, pickle_out)
	  pickle_out.close()
  
   #static function for loading
   def loadModel(filename):
      my_object_file = open(filename+'.pkl', 'rb')
      xmodel = pickle.load(my_object_file)
      my_object_file.close()
      return xmodel

      
      
"""
Just a test for subclassing
"""
class LogisticRegressionMod(LogisticRegression):
      #subclassing init must repeat all keyword arguments...
      #def __init__(self):
	#pass	
      def fit(self, X, y,sample_weight=None):
	  N = X.shape[0]
	  #print "N:",N	 
	  if sample_weight is None:
	      sample_weight=np.ones(N) 
	  #print "max weight:",np.max(sample_weight)
	  #print "min weight:",np.min(sample_weight)
	  #print "avg weight:",np.mean(sample_weight)
	  #print sample_weight
	  if isinstance(X,sp.sparse.csr.csr_matrix):
	      mat1 = sp.sparse.dia_matrix(([sample_weight], [0]), shape=(N, N))
	  else:
	      mat1 = np.diagflat([1.0/sample_weight])
	  #* operator does not define a matrix multiplication with arrays
	  Xmod = np.copy(X)
	  Xmod = np.dot(mat1, Xmod)
	  
	  pd.DataFrame(Xmod).to_csv("Xmod.csv")
	  pd.DataFrame(X).to_csv("X.csv")
	  lmodel=super(LogisticRegression, self).fit(Xmod,y)
	  #print lmodel.coef_
	  pred=lmodel.predict_proba(Xmod)[:,1]
	  print "AUC:",roc_auc_score(y,pred)	  
	  return lmodel

