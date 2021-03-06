#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Chrissly31415 August 2013


from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator

from scipy.optimize import fmin, fmin_cobyla, minimize

from qsprLib import funcdict, save_sparse_csr, load_sparse_csr

import scipy as sp
import numpy as np
import pandas as pd
import pickle as pickle
#import dill as pickle

class XModel:
    """
    class holds classifier plus train and test set for later use in ensemble building
    Wrapper for ensemble building
    """
    modelcount = 0

    def __init__(self, name, classifier, Xtrain=None, Xtest=None, ytrain=None, sample_weight=None, Xval=None, yval=None,
                 cutoff=None, class_names=None, cv_labels=None, bag_mode=False, fit_on_validation=True, params=None,generators=None):
        """
        Old constructor

        :param name:
        :param classifier:
        :param Xtrain:
        :param Xtest:
        :param ytrain:
        :param sample_weight:
        :param Xval:
        :param yval:
        :param cutoff:
        :param class_names:
        :param cv_labels:
        :param bag_mode:
        :param fit_on_validation:
        :param params:

        """
        self.name = name
        self.classifier = classifier
        self.Xtrain = Xtrain
        self.Xtest = Xtest
        self.ytrain = ytrain
        self.sample_weight = sample_weight
        self.Xval = Xval
        self.yval = yval

        self.cutoff = cutoff
        if cutoff is not None:
            self.use_proba = True

        self.class_names = class_names
        self.cv_labels = cv_labels  # for special cv like leave pout
        self.bag_mode = bag_mode  # bagging i.e. cross-validated models for prediction
        self.fit_on_validation = fit_on_validation
        self.params = params
        self.generators = generators

        if params is None:
            self.initialize()

    def initialize(self):
        if isinstance(self.Xtrain, sp.sparse.csr.csr_matrix) or isinstance(self.Xtrain, sp.sparse.csc.csc_matrix):
            self.sparse = True
        else:
            self.sparse = False

        self.oob_preds = np.zeros((self.Xtrain.shape[0], 1))
        self.preds = np.zeros((self.Xtest.shape[0], 1))
        if self.Xval is not None:
            self.val_preds = np.zeros((self.Xval.shape[0], 1))

        if self.Xval is not None and self.sample_weight is not None:
            raise Exception("Holdout and sampleweight currently not supported!")

        XModel.modelcount += 1

    def summary(self):
        sum_str = "\n##########################################################################\n"
        sum_str += ">>name         : %s\n"%(self.name)
        sum_str+=  ">>classifier   : %s\n"%(self.classifier)
        sum_str += ">>parameters for feature generatation:\n %r\n"%(self.params)

        if self.Xtrain is not None:
            sum_str += ">>train data   : %s %s\n" %(self.Xtrain.shape)

        if self.Xtest is not None:
            sum_str += ">>test data    : %s %s\n"%(self.Xtest.shape)

        if self.Xval is not None:
            sum_str += ">>valid data   : %s %s\n" % (self.Xval.shape)

        if self.Xtrain is not None:
            sum_str += ">>columns: %r\n" % (list(self.Xtrain.columns))

        if self.sample_weight is not None:
            sum_str +="sample_weight: %r\n" %(self.sample_weight.shape)

        if self.ytrain is not None:
            sum_str += ">>target(y)   : %r\n"%(self.ytrain.shape)

        # print "sparse data  :" , self.sparse
        if self.cutoff is not None: sum_str +=("proba cutoff  :", self.cutoff)
        if self.class_names is not None: sum_str +=("class names  :", self.class_names)

        if self.preds is not None:
            sum_str += ">>predictions,mean       : %6.3f " % (np.mean(self.preds))
            sum_str +=f" shape: {self.preds.shape}  \n"

        sum_str +=     ">>oob predictions,mean   : %6.3f " % (np.mean(self.oob_preds))
        sum_str +=f" shape: {self.oob_preds.shape}\n"

        if self.val_preds is not None:
            sum_str += ">>valid predictions,mean : %6.3f " % (np.mean(self.val_preds))
            sum_str += f" shape: {self.val_preds.shape}\n"

        #if self.cv_labels is not None:
        #    sum_str += ">>cv_labels : %s " % (self.cv_labels[:10])
        #    sum_str += f" cv_labels: {self.val_preds.shape}\n"

        sum_str +="##########################################################################"
        return sum_str

    def __repr__(self):
        return self.summary()

    def set_feature_params(self,params_dict,generator_dict):
        self.params = params_dict
        self.generator_dict = generator_dict

    def generate_features(self):
        print(("Generating features for model: %s \n"%(self.name)))
        Xtest, Xtrain, ytrain, cv_labels, sample_weight, Xval, yval = self.generators['prepareDataset'](**self.params)

        self.Xtrain = Xtrain
        self.Xtest = Xtest
        self.ytrain = ytrain
        self.sample_weight = sample_weight
        self.Xval = Xval
        self.yval = yval
        self.cv_labels = cv_labels

        self.initialize()
        print(("Feature generation finished: " + self.name))

    @staticmethod
    def saveDataSet(xmodel, basedir='./share/'):
        if xmodel.sparse:
            save_sparse_csr(basedir + "Xtrain_" + xmodel.name + "_sparse.csv", xmodel.Xtrain)
            save_sparse_csr(basedir + "Xval" + xmodel.name + "_sparse.csv", xmodel.Xval)
            save_sparse_csr(basedir + "Xtest_" + xmodel.name + "_sparse.csv", xmodel.Xtest)
        else:
            xmodel.Xtrain.to_csv(basedir + "Xtrain_" + xmodel.name + ".csv")
            xmodel.Xval.to_csv(basedir + "Xval_" + xmodel.name + ".csv")
            xmodel.Xtest.to_csv(basedir + "Xtest_" + xmodel.name + ".csv")

    @staticmethod
    def loadDataSet(xmodel, basedir='./share/'):
        if xmodel.sparse:
            xmodel.Xtrain = load_sparse_csr(basedir + "Xtrain_" + xmodel.name + "_sparse.csv.npz")
            xmodel.XVal = load_sparse_csr(basedir + "Xval_" + xmodel.name + "_sparse.csv.npz")
            xmodel.Xtest = load_sparse_csr(basedir + "Xtest_" + xmodel.name + "_sparse.csv.npz")
        else:
            xmodel.Xtrain = pd.read_csv(basedir + "Xtrain_" + xmodel.name + ".csv",index_col=0)
            xmodel.Xval = pd.read_csv(basedir + "Xval_" + xmodel.name + ".csv", index_col=0)
            xmodel.Xtest = pd.read_csv(basedir + "Xtest_" + xmodel.name + ".csv",index_col=0)

        return (xmodel.Xtrain, xmodel.Xval, xmodel.Xtest)

    # static function for saving
    @staticmethod
    def saveModel(xmodel, filename):
            pickle_out = open(filename.replace('.csv', ''), 'wb')
            pickle.dump(xmodel, pickle_out)
            pickle_out.close()

    # static function for saving only the important data
    @staticmethod
    def saveCoreData(xmodel, filename):
        if xmodel.classifier is not None and 'm__epochs' in xmodel.classifier.get_params():
            print((type(xmodel.classifier.get_params())))

        xmodel.Xtrain = None
        #xmodel.ytrain = None
        xmodel.Xval = None
        #xmodel.yval = None
        xmodel.Xtest = None
        #xmodel.ytest = None

        xmodel.sample_weight = None
        # keep only parameters and predictions
        pickle_out = open(filename.replace('.csv', ''), 'wb')
        pickle.dump(xmodel, pickle_out)
        pickle_out.close()

    # static function for loading
    @staticmethod
    def loadModel(filename):
        my_object_file = open(filename + '.pkl', 'rb')
        xmodel = pickle.load(my_object_file)
        my_object_file.close()
        return xmodel

    # static function for loading
    @staticmethod
    def loadCoreData(filename):
        my_object_file = open(filename + '.pkl', 'rb')
        xmodel = pickle.load(my_object_file)
        my_object_file.close()
        return xmodel


class FeaturePredictor(BaseEstimator):
    """
    Yields single column(s) with Feature
    """

    def __init__(self, fname, pos=0):
        self.fname = fname
        self.pos = pos

    def fit(self, lX, ly=None, sample_weight=None):
        pass

    def predict(self, lX):
        if isinstance(lX,pd.DataFrame):
            return lX[self.fname].values.astype(int)
        else:
            preds = lX[:,self.pos].astype(int)
            return preds


class NothingTransform(BaseEstimator):
    """
    Yields single column(s) with Feature
    """

    def fit(self, lX, ly=None, sample_weight=None):
        pass

    def fit_transform(self, lX, ly=None, sample_weight=None):
        return lX

    def transform(self, lX):
        return lX

    def predict(self, lX):
        pass


class ConstrainedLinearRegressor(BaseEstimator):
    """
        Constrained linear regression
        """

    def __init__(self, lowerbound=0, upperbound=1.0, n_classes=1, alpha=None, corr_penalty=None, normalize=False,
                 loss='rmse', greater_is_better=False):
        self.normalize = normalize
        self.greater_is_better = greater_is_better
        self.lowerbound = lowerbound
        self.upperbound = upperbound
        self.n_classes = n_classes
        self.alpha = alpha
        self.loss = loss
        self.coef_ = 0
        self.corr_penalty = corr_penalty

    def fit(self, lX, ly, sample_weight=None):
        n_cols = lX.shape[1]
        x0 = np.ones((n_cols, 1))
        constr_lb = [lambda x, z=i: x[z] - self.lowerbound for i in range(n_cols)]
        constr_ub = [lambda x, z=i: self.upperbound - x[z] for i in range(n_cols)]
        constr = constr_lb + constr_ub
        # constr=constr_lb
        self.coef_ = fmin_cobyla(self.fopt, x0, constr, args=(lX, ly), consargs=(), rhoend=1e-10, maxfun=10000, disp=0)
        # coef_ = minimize(fopt, x0,method='COBYLA',constraints=self.constr)
        # normalize coefficient
        if self.normalize:
            self.coef_ = self.coef_ / np.sum(self.coef_)
            # print "Normalizing coefficients:",self.coef_

        if np.isnan(np.sum(self.coef_)):
            print("We have NaN here...")

    def predict(self, lX):
        ypred = self.blend_mult(lX, self.coef_, self.n_classes)
        return ypred.flatten()

    def predict_proba(self, lX):
        yt = self.predict(lX)
        ypred = np.zeros((yt.shape[0],2))
        ypred[:,0] = 1.0-yt
        ypred[:,1] = yt
        return ypred


    def fopt(self, params, X, y):
        # nxm  * m*1 ->n*1
        ypred = self.blend_mult(X, params, self.n_classes)
        score = funcdict[self.loss](y, ypred)
        # if not use_proba: ypred = np.round(ypred).astype(int)
        # regularization
        if self.alpha is not None:
            # cc = np.corrcoef(X.values,rowvar=0)
            # cc = np.power(cc,2)
            # cc = self.corr_penalty * np.mean(cc)
            l2 = self.alpha * np.sum(np.power(params, 2))
            # l1 = self.alpha *np.sum(np.absolute(params))
            score = score + l2
        # print "score: %6.2f alpha: %6.2f cc: %r l2: %6.3f"%(score,self.alpha,cc,l2)
        # raw_input()
        if self.greater_is_better: score = -1 * score
        return score

    def blend_mult(self, X, params, n_classes=None):
        if n_classes < 2:
            return np.dot(X, params)
            #    else:
            #        return multiclass_mult(X,params,n_classes)
