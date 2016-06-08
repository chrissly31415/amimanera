#!/usr/bin/python 
# coding: utf-8

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from keras.models import Sequential
from keras.optimizers import SGD,Adagrad,RMSprop
from keras.layers.core import Dense, Dropout, Activation, MaxoutDense

from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU,LeakyReLU
from keras.utils import np_utils, generic_utils
from sklearn.base import BaseEstimator
import theano.tensor as T

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


Rossmann 3d place: https://github.com/entron/category-embedding-rossmann/blob/master/models.py "categorical embedding"

'''

def RMSE(y_true, y_pred):
    loss = T.sqrt(T.sqr(y_true - y_pred).mean(axis=-1))
    #print(loss)
    return loss


class KerasNN(BaseEstimator):
    def __init__(self, dims=66, nb_classes=1, nb_epoch=30, learning_rate=0.5, validation_split=0.0, batch_size=64,
                 loss='categorical_crossentropy', layers=[32,32], activation='relu',  dropout=[0.2,0.2],verbose=1):

        self.dims = dims
        self.nb_classes = nb_classes
        self.classes_ = None # list containing classes
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.loss = loss
        self.layers = layers
        self.activation = activation
        self.dropout = dropout
        self.verbose = verbose

        self.model = Sequential()
        # Keras model
        for i,dropout in enumerate(self.dropout):
            if i>0:
                dims = self.layers[i-1]

            if 'maxout' in self.activation:
                self.model.add(MaxoutDense(output_dim=layers[i], nb_feature=4, input_dim=dims))
            else:
                self.model.add(Dense(output_dim=layers[i], input_dim=dims, init='glorot_uniform'))
                #https://www.reddit.com/r/MachineLearning/comments/22u1yt/is_deep_learning_basically_just_neural_networks/
                #https://www.kaggle.com/c/job-salary-prediction/forums/t/4208/congratulations-to-the-preliminary-winners?page=2
                if 'PReLU' in self.activation:
                    self.model.add(PReLU())
                elif 'LeakyReLU' in self.activation:
                    self.model.add(LeakyReLU(alpha=0.3))
                else:
                    self.model.add(Activation(self.activation))

            self.model.add(BatchNormalization())
            if dropout>1E-15:
                self.model.add(Dropout(dropout))


        if 'categorical_crossentropy' in loss:
            self.model.add(Dense(output_dim=nb_classes))
            self.model.add(Activation('softmax'))
            self.model.compile(loss=loss, optimizer="adadelta")

        else:
            self.model.add(Dense(output_dim=1))
            self.model.add(Activation('linear'))
            #optimizer = Adagrad(lr=self.learning_rate) # 0.01
            #optimizer = Adagrad()
            print("Learning rate:",self.learning_rate)
            optimizer = RMSprop(lr=self.learning_rate) # 0.001
            #optimizer = RMSprop()
            if 'rmse' in self.loss:
                self.model.compile(loss=RMSE, optimizer=optimizer)
            else:
                self.model.compile(loss=self.loss, optimizer=optimizer)

        # tanh better for regression?

        #sgd = SGD(lr=self.learning_rate, decay=1e-7, momentum=0.99, nesterov=True)
        print('Compiling Keras Deep Net with loss: %s and activation: %s' % (str(self.loss),self.activation))


    def fit(self, X, y, sample_weight=None):
        print('Fitting  Keras Deep Net for regression with batch_size %d, epochs %d  and learning rate: %f' % (
        self.batch_size, self.nb_epoch, self.learning_rate))
        if self.nb_classes>1:
            y = np_utils.to_categorical(y)
            self.classes_ = np.unique(y)
        self.model.fit(X, y, nb_epoch=self.nb_epoch, batch_size=self.batch_size,
                       validation_split=self.validation_split,verbose=self.verbose)

    def predict_proba(self, Xtest):
        ypred = self.model.predict_proba(Xtest, batch_size=self.batch_size, verbose=self.verbose)
        return ypred

    def predict(self, Xtest):
        ypred = self.model.predict(Xtest, batch_size=self.batch_size, verbose=self.verbose)
        if self.nb_classes>1:
            ypred = np_utils.probas_to_classes(ypred)
        else:
            ypred = ypred.flatten()
        return ypred
