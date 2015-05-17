#!/usr/bin/python 
# coding: utf-8

from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils
from sklearn.base import BaseEstimator


'''
    This demonstrates how to reach a score of 0.4890 (local validation)
    on the Kaggle Otto challenge, with a deep net using Keras.
    Compatible Python 2.7-3.4 
    Recommended to run on GPU: 
        Command: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python kaggle_otto_nn.py
        On EC2 g2.2xlarge instance: 19s/epoch. 6-7 minutes total training time.
    Best validation score at epoch 21: 0.4881 
    Try it at home:
        - with/without BatchNormalization (BatchNormalization helps!)
        - with ReLU or with PReLU (PReLU helps!)
        - with smaller layers, largers layers
        - with more layers, less layers
        - with different optimizers (SGD+momentum+decay is probably better than Adam!)
    Get the data from Kaggle: https://www.kaggle.com/c/otto-group-product-classification-challenge/data
'''

class KerasNN(Sequential,BaseEstimator):
    
    def __init__(self,dims=93,nb_classes=9):
	Sequential.__init__(self)
	self.dims = dims
	self.nb_classes = nb_classes
	print('Initializing Keras Deep Net with %d features and %d classes'%(self.dims,self.nb_classes))

	self.add(Dropout(0.1))
	self.add(Dense(dims, 512, init='glorot_uniform'))
	self.add(PReLU((512,)))
	self.add(BatchNormalization((512,)))
	self.add(Dropout(0.5))

	self.add(Dense(512, 512, init='glorot_uniform'))
	self.add(PReLU((512,)))
	self.add(BatchNormalization((512,)))
	self.add(Dropout(0.5))

	self.add(Dense(512, 512, init='glorot_uniform'))
	self.add(PReLU((512,)))
	self.add(BatchNormalization((512,)))
	self.add(Dropout(0.25))

	self.add(Dense(512, nb_classes, init='glorot_uniform'))
	self.add(Activation('softmax'))

	sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True)
	self.compile(loss='categorical_crossentropy', optimizer=sgd)

    def fit(self,X,y):
	y = np_utils.to_categorical(y)
	Sequential.fit(self,X, y, nb_epoch=50, batch_size=128, validation_split=0.0)
	

    def predict_proba(self,Xtest):
	#ypred = Sequential.predict_proba(self,Xtest,batch_size=128,verbose=1)
	ypred = Sequential.predict_proba(self,Xtest)
	print(ypred.shape)
	return ypred
    
    #problem keras general predict function...
    #def predict(self,Xtest):
    #	print("this is predict")
    #	#ypred = Sequential.predict_classes(self,Xtest,batch_size=128,verbose=1)
    #	ypred = Sequential.predict_classes(self,Xtest)
    #	return ypred
    


