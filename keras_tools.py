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

'''
From kaggle forum:

NN is the average of 30 neural networks with the same parameters fed by x^(2/3) transformed features and by results of KNN with N = 27 (KNN gained .002 for my best solution). NN was implemented on Keras, I've found this library very nice and fast (with CUDA-enabled Theano). Layers were (512,256,128), the score was .428

Dropout(.15) -> Dense(n_in, l1, activation='tanh') -> BatchNormalization((l1,)) -> Dropout(.5) -> Dense(l1, l2) -> PReLU((l2,)) -> BatchNormalization((l2,)) -> Dropout(.3) -> Dense(l2, l3) -> PReLU((l3,)) -> BatchNormalization((l3,)) -> Dropout(.1) -> Dense(l3, n_out) -> Activation('softmax')
sgd = SGD(lr=0.004, decay=1e-7, momentum=0.99, nesterov=True)

'''

class KerasNN(Sequential,BaseEstimator):
    
    def __init__(self,dims=93,nb_classes=9,nb_epoch=50,learning_rate=0.02,validation_split=0.0,batch_size=128,verbose=1):
	Sequential.__init__(self)
	self.dims = dims
	self.nb_classes = nb_classes
	self.nb_epoch = nb_epoch
	self.learning_rate = learning_rate
	self.validation_split = validation_split
	self.batch_size = batch_size
	self.verbose = verbose
	print('Initializing Keras Deep Net with %d features and %d classes'%(self.dims,self.nb_classes))

	self.add(Dropout(0.10))
	self.add(Dense(dims, 900, init='glorot_uniform'))
	self.add(PReLU((900,)))
	self.add(BatchNormalization((900,)))
	self.add(Dropout(0.5))

	self.add(Dense(900, 512, init='glorot_uniform'))
	self.add(PReLU((512,)))
	self.add(BatchNormalization((512,)))
	self.add(Dropout(0.25))

	self.add(Dense(512, 256, init='glorot_uniform'))
	self.add(PReLU((256,)))
	self.add(BatchNormalization((256,)))
	self.add(Dropout(0.25))

	self.add(Dense(256, nb_classes, init='glorot_uniform'))
	self.add(Activation('softmax'))

	sgd = SGD(lr=self.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
	self.compile(loss='categorical_crossentropy', optimizer=sgd)

    def fit(self,X,y):
	y = np_utils.to_categorical(y)
	Sequential.fit(self,X, y, nb_epoch=self.nb_epoch, batch_size=self.batch_size, validation_split=self.validation_split)
	

    def predict_proba(self,Xtest):
	#ypred = Sequential.predict_proba(self,Xtest,batch_size=128,verbose=1)
	ypred = Sequential.predict_proba(self,Xtest,batch_size=self.batch_size,verbose=self.verbose)
	print(ypred.shape)
	return ypred
    

class KerasNN2(Sequential,BaseEstimator):
    
    def __init__(self,dims=93,nb_classes=9,nb_epoch=50,learning_rate=0.02,validation_split=0.0,batch_size=128,verbose=1):
	Sequential.__init__(self)
	self.dims = dims
	self.nb_classes = nb_classes
	self.nb_epoch = nb_epoch
	self.learning_rate = learning_rate
	self.validation_split = validation_split
	self.batch_size = batch_size
	self.verbose = verbose
	print('Initializing Keras Deep Net with %d features and %d classes'%(self.dims,self.nb_classes))

	self.add(Dropout(0.2))
	self.add(Dense(dims, 900, init='glorot_uniform'))
	self.add(PReLU((900,)))
	self.add(BatchNormalization((900,)))
	self.add(Dropout(0.5))

	self.add(Dense(900, 500, init='glorot_uniform'))
	self.add(PReLU((500,)))
	self.add(BatchNormalization((500,)))
	self.add(Dropout(0.25))

	self.add(Dense(500, 250, init='glorot_uniform'))
	self.add(PReLU((250,)))
	self.add(BatchNormalization((250,)))
	self.add(Dropout(0.25))

	self.add(Dense(250, nb_classes, init='glorot_uniform'))
	self.add(Activation('softmax'))

	self.compile(loss='categorical_crossentropy', optimizer="adam")

    def fit(self,X,y):
	y = np_utils.to_categorical(y)
	Sequential.fit(self,X, y, nb_epoch=self.nb_epoch, batch_size=self.batch_size, validation_split=self.validation_split)
	

    def predict_proba(self,Xtest):
	#ypred = Sequential.predict_proba(self,Xtest,batch_size=128,verbose=1)
	ypred = Sequential.predict_proba(self,Xtest,batch_size=self.batch_size,verbose=self.verbose)
	print(ypred.shape)
	return ypred


    #problem keras general predict function...
    #def predict(self,Xtest):
    #	print("this is predict")
    #	#ypred = Sequential.predict_classes(self,Xtest,batch_size=128,verbose=1)
    #	ypred = Sequential.predict_classes(self,Xtest)
    #	return ypred
    


