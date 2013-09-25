#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Chrissly31415 August 2013


from sklearn.linear_model import LogisticRegression
import scipy as sp
import numpy as np

class XModel:
   """
   class holds classifier plus train and test set for later use in ensemble building
   """
   modelcount = 0

   def __init__(self, name,classifier,Xtrain,Xtest):
      self.name = name
      self.classifier = classifier
      self.Xtrain=Xtrain
      self.Xtest=Xtest
      if isinstance(Xtrain,sp.sparse.csr.csr_matrix):
	self.sparse=True
      else:
	self.sparse=False
      self.oobpreds=np.zeros((Xtrain.shape[0],1))
      self.pres=np.zeros((Xtest.shape[0],1))
	
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

"""
Just a test for subclassing
"""
class LogisticRegressionMod(LogisticRegression):
      pass
      #subclassing init must repeat all keyword arguments...
      #def __init__(self):
	#pass
      
      def fit(self, X, y,sample_weight=None):
	return super(LogisticRegression, self).fit(X, y)
	#pass
	#print "Creating LR subclass..."
	#print self
	#super(LogisticRegression, self).__init__()

        
        

#class FullEstimator(Estimator):
#      def __init__(self):
#        Estimator.__init__(self)