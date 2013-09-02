#!/usr/bin/env python
# -*- coding: utf-8 -*-
class FullModel:
   """
   class holds classifier plus train and test set for later use in ensemble building
   """
   modelcount = 0

   def __init__(self, name,classifier,Xtrain,Xtest,y):
      self.name = name
      self.classifier = classifier
      self.Xtrain=Xtrain
      self.Xtest=Xtest
      self.y=y
      FullModel.modelcount += 1

   def summary(self):
      print "Name : ", self.name
      print "classifier:" , type(self.classifier)
      print "Training data:" , self.Xtrain.shape
      print "type:" , type(self.Xtrain)
      print "Test data:" , self.Xtest.shape
      print "type:" , type(self.Xtest)