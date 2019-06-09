#!/usr/bin/python 
# coding: utf-8



import numpy as np
from keras.models import Sequential, load_model
from keras.optimizers import SGD,Adagrad,RMSprop,Adam
from keras.layers.core import Dense, Dropout, Activation
from keras import optimizers

from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU,LeakyReLU
from keras.utils import np_utils, generic_utils
from sklearn.base import BaseEstimator
import types
import tempfile
import keras.models

from keras import callbacks


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

avito challenge https://www.kaggle.com/rightfit/avito-duplicate-ads-detection/get-hash-from-images/code

'''

def RMSE(y_true, y_pred):
    loss = T.sqrt(T.sqr(y_true - y_pred).mean(axis=-1))
    #print(loss)
    return loss



def make_keras_picklable():
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = { 'model_str': model_str }
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__


    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__


#https://gist.github.com/MaxHalford/9bfaa8daf8b4bc17a7fb7ba58c880675#file-fit-py
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='auto')

def create_classification_model(input_dim=64,learning_rate=0.001,activation='relu',batchnorm=False,layers=[256,256],dropouts=[0.0,0.0],optimizer=None):
    # create model
    model = Sequential()
    for i,(layer,dropout) in enumerate(zip(layers,dropouts)):
        if i==0:
            model.add(Dense(layer, input_dim=input_dim, kernel_initializer='uniform'))
            if batchnorm: model.add(BatchNormalization())  # problem with CUDA?
            model.add(Activation(activation))
            model.add(Dropout(dropout))

        else:
            model.add(Dense(layer, kernel_initializer='uniform'))
            if batchnorm: model.add(BatchNormalization())
            model.add(Activation(activation))
            model.add(Dropout(dropout))

    if batchnorm: model.add(BatchNormalization())
    model.add(Dense(1, kernel_initializer='uniform',activation='sigmoid'))
    # Compile model
    if optimizer is None:
        optimizer = optimizers.SGD(lr=learning_rate, momentum=0.0, decay=0.0, nesterov=False)  # normal
    elif 'adam' in optimizer:
        optimizer = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) # deep nets
    elif 'adadelta' in optimizer:
        optimizer = optimizers.Adadelta(lr=learning_rate, rho=0.95, epsilon=1e-08, decay=0.0)
    elif 'adagrad' in optimizer:
        optimizer = Adagrad(lr=self.learning_rate)
    else:
        optimizer = optimizers.SGD(lr=learning_rate, momentum=0.0, decay=0.0, nesterov=False)  # normal

    model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    return model

def create_regression_model(input_dim=64,learning_rate=0.001,activation='sigmoid',layers=[256,256],dropouts=[0.0,0.0],optimizer=None):
    # create model
    model = Sequential()
    for i,(layer,dropout) in enumerate(zip(layers,dropouts)):
        if i==0:
            model.add(Dropout(dropout))
            model.add(Dense(layer, input_dim=input_dim, kernel_initializer='normal', activation=activation))
        else:
            model.add(Dropout(dropout))
            model.add(Dense(layer, kernel_initializer='normal', activation=activation))

    model.add(Dense(1, kernel_initializer='normal',activation='linear'))
    # Compile model
    #model.compile(loss='mean_squared_error', optimizer=optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0))
    #model.compile(loss='mean_squared_error', optimizer=Adagrad(lr=self.learning_rate) # 0.01
    if optimizer is None:
        optimizer = optimizers.RMSprop(lr=learning_rate)
    model.compile(loss='mean_squared_error',optimizer=optimizer)
    return model

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
        self.hist = ""

        self.model = Sequential()
        # Keras model
        for i,dropout in enumerate(self.dropout):
            if i>0:
                dims = self.layers[i-1]

            if 'maxout' in self.activation:
                print("Currently not implemented...")
                #self.model.add(MaxoutDense(output_dim=layers[i], nb_feature=4, input_dim=dims))
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
            y = np.reshape(y,(y.shape[0],-1))

        #pandas hack
        if not isinstance(X,np.ndarray):
            X = X.values

        self.model.fit(X, y,batch_size=self.batch_size, nb_epoch=self.nb_epoch, verbose=self.verbose,callbacks=[],
                       validation_split=self.validation_split,validation_data=None,shuffle=True,class_weight=None,sample_weight=sample_weight)

    def predict_proba(self, X):
        if not isinstance(X,np.ndarray):
            X = X.values
        ypred = self.model.predict_proba(X, batch_size=self.batch_size, verbose=self.verbose)
        return ypred

    def predict(self, X):
        if not isinstance(X,np.ndarray):
            X = X.values
        ypred = self.model.predict(X, batch_size=self.batch_size, verbose=self.verbose)
        if self.nb_classes>1:
            ypred = np_utils.probas_to_classes(ypred)
        else:
            ypred = ypred.flatten()
        return ypred

    def save_model(self,filename):
        self.model.save(filename)

    def load_model(self,filename):
        self.model = load_model(filename)


