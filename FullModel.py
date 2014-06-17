#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Chrissly31415 August 2013


from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import scipy as sp
import numpy as np
import pandas as pd

class XModel:
   """
   class holds classifier plus train and test set for later use in ensemble building
   """
   modelcount = 0

   def __init__(self, name,classifier,Xtrain,Xtest,weights=None):
      self.name = name
      self.classifier = classifier
      self.Xtrain=Xtrain
      self.Xtest=Xtest
      if isinstance(Xtrain,sp.sparse.csr.csr_matrix) or isinstance(Xtrain,sp.sparse.csc.csc_matrix):
	self.sparse=True
      else:
	self.sparse=False
      self.oobpreds=np.zeros((Xtrain.shape[0],1))
      self.preds=np.zeros((Xtest.shape[0],1))
	
      XModel.modelcount += 1

   def summary(self):
      print ">>Name<<     :" ,  self.name
      print "classifier   :" , type(self.classifier)
      print "Train data   :" , self.Xtrain.shape
      print "type         :" , type(self.Xtrain)
      print "Test data    :" , self.Xtest.shape
      print "type         :" , type(self.Xtest)
      print "sparse data  :" , self.sparse 
      print "oob preds    :" , np.mean(self.oobpreds)
      print "predictions  :" , np.mean(self.preds)
      print "weights 	   :" , self.weights.shape

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


        
        

#class FullEstimator(Estimator):
#      def __init__(self):
#        Estimator.__init__(self)