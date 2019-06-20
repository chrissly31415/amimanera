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

from nmr import *
from FullModel import XModel, FeaturePredictor

import sys
import itertools
from random import randint
import argparse

from scipy.optimize import fmin, fmin_cobyla, minimize

from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn import preprocessing

from qsprLib import funcdict

lbl = LabelEncoder()
tmp = pd.read_csv('./data/train.csv')
lbl.fit(list(tmp['type'].values))


def createModels(models = []):

    rseed = 666
    ensemble = []
    fit_on_validation = True

    generators = {'prepareDataset' : prepareDataset}

    # type
    if "type" in models:
        params = {'seed': rseed, 'nsamples': -1,'makeLabelEncode': True}
        model = FeaturePredictor('type')
        xmodel = XModel("type", classifier=model, bag_mode=False, fit_on_validation=fit_on_validation, params=params,
                        generators=generators)
        ensemble.append(xmodel)

    # rt1 
    if "rt1" in models:
        params = {'seed': rseed, 'nsamples': -1, 'makeDistMat': True, 'makeTrainType': True, 'plotDist': False,
                  'makeDistMean': True, 'makeMolNameMean': True, 'makeLabelEncode': True, 'makeRDKitFeatures': True,
                  'makeRDKitFingerPrints': False, 'oneHotenc': None, 'removeLowVariance': False,
                  'dropFeatures': ['atom_index_0', 'atom_index_1', 'molblock']}
        model = Pipeline([('rt', RandomTreesEmbedding(n_estimators=40, max_depth=2)),
                          ('m', LinearRegression(C=0.01, penalty='l2', solver='lbfgs'))])
        xmodel = XModel("rt1", classifier=model, bag_mode=False, fit_on_validation=fit_on_validation, params=params,
                        generators=generators)
        ensemble.append(xmodel)

    # ridge1 
    if "ridge1" in models:
        params = {'seed': rseed, 'nsamples': -1, 'makeDistMat': True, 'makeTrainType': True, 'plotDist': False,
                  'makeDistMean': True, 'makeMolNameMean': True, 'makeLabelEncode': True, 'makeRDKitFeatures': False,
                  'makeRDKitFingerPrints': False, 'oneHotenc': None, 'removeLowVariance': False,
                  'dropFeatures': ['atom_index_0', 'atom_index_1', 'molblock']}

        model = RidgeCV()
        xmodel = XModel("ridge1", classifier=model, bag_mode=False, fit_on_validation=fit_on_validation, params=params,
                        generators=generators)
        ensemble.append(xmodel)


    if "lgb1" in models:#defaul
        params = {'seed': rseed, 'nsamples': -1, 'makeDistMat': True, 'makeTrainType': True, 'plotDist': False,
                  'makeDistMean': True, 'makeMolNameMean': True, 'makeLabelEncode': True, 'makeRDKitFeatures': False,
                  'makeRDKitFingerPrints': False, 'oneHotenc': None, 'removeLowVariance': False,
                  'dropFeatures': ['atom_index_0', 'atom_index_1', 'molblock']}

        lgb_params = {'num_leaves': 128, 'min_child_samples': 79, 'objective': 'regression', 'max_depth': 12,
                      'learning_rate': 0.2, "boosting_type": "gbdt", "subsample_freq": 1, "subsample": 1.0,
                      "bagging_seed": 11, "metric": 'mae', "verbosity": -1, 'reg_alpha': 0.1, 'reg_lambda': 0.3,
                      'colsample_bytree': 1.0, 'n_estimators': 10000,'n_jobs':4}

        model = lgb.LGBMRegressor(**lgb_params)
        xmodel = XModel("lgb1", classifier=model, bag_mode=False, fit_on_validation=fit_on_validation, params=params,
                        generators=generators)
        ensemble.append(xmodel)

    if "lgb2" in models:#makeRDKitFingerPrints
        params = {'seed': rseed, 'nsamples': -1, 'makeDistMat': True, 'makeTrainType': True, 'plotDist': False,
                  'makeDistMean': True, 'makeMolNameMean': True, 'makeLabelEncode': True, 'makeRDKitFingerPrints': True,
                  'dropFeatures': ['atom_index_0', 'atom_index_1', 'molblock']}

        lgb_params = {'num_leaves': 128, 'min_child_samples': 79, 'objective': 'regression', 'max_depth': 10,
                      'learning_rate': 0.2, "boosting_type": "gbdt", "subsample_freq": 1, "subsample": 1.0,
                      "bagging_seed": 11, "metric": 'mae', "verbosity": -1, 'reg_alpha': 0.1, 'reg_lambda': 0.3,
                      'colsample_bytree': 1.0, 'n_estimators': 5000,'n_jobs':-1}

        model = lgb.LGBMRegressor(**lgb_params)
        xmodel = XModel("lgb2", classifier=model, bag_mode=False, fit_on_validation=fit_on_validation, params=params,
                        generators=generators)
        ensemble.append(xmodel)

    if "lgb3" in models: #bruteForceFeatures
        params = {'seed': rseed, 'nsamples': -1, 'makeDistMat': True, 'makeTrainType': True,
                  'makeDistMean': True, 'makeMolNameMean': True, 'makeLabelEncode': True, 'bruteForceFeatures': True,
                  'dropFeatures': ['atom_index_0', 'atom_index_1', 'molblock']}

        lgb_params = {'num_leaves': 128, 'min_child_samples': 79, 'objective': 'regression', 'max_depth': 9,
                      'learning_rate': 0.2, "boosting_type": "gbdt", "subsample_freq": 1, "subsample": 0.9,
                      "bagging_seed": 11, "metric": 'mae', "verbosity": -1, 'reg_alpha': 0.1, 'reg_lambda': 0.3,
                      'colsample_bytree': 1.0, 'n_estimators': 5000,'n_jobs':-1}

        model = lgb.LGBMRegressor(**lgb_params)
        xmodel = XModel("lgb3", classifier=model, bag_mode=False, fit_on_validation=fit_on_validation, params=params,
                        generators=generators)
        ensemble.append(xmodel)

    if "lgb4" in models: #obCharges
        params = {'seed': rseed, 'nsamples': -1, 'makeDistMat': True, 'makeTrainType': True,
                  'makeDistMean': True, 'makeMolNameMean': True, 'makeLabelEncode': True, 'obCharges': True,
                  'dropFeatures': ['atom_index_0', 'atom_index_1', 'molblock']}

        lgb_params = {'num_leaves': 128, 'min_child_samples': 79, 'objective': 'regression', 'max_depth': 10,
                      'learning_rate': 0.2, "boosting_type": "gbdt", "subsample_freq": 1, "subsample": 1.0,
                      "bagging_seed": 11, "metric": 'mae', "verbosity": -1, 'reg_alpha': 0.1, 'reg_lambda': 0.3,
                      'colsample_bytree': 1.0, 'n_estimators': 5000,'n_jobs':-1}
    if "lgb5" in models:  # makeRDKitAtomicFeatures
        params = {'seed': rseed, 'nsamples': -1, 'makeDistMat': True, 'makeTrainType': True,
                  'makeDistMean': True, 'makeMolNameMean': True, 'makeLabelEncode': True, 'makeRDKitAtomFeatures': True,
                  'dropFeatures': ['atom_index_0', 'atom_index_1', 'molblock']}

        lgb_params = {'num_leaves': 128, 'min_child_samples': 79, 'objective': 'regression', 'max_depth': 10,
                      'learning_rate': 0.2, "boosting_type": "gbdt", "subsample_freq": 1, "subsample": 1.0,
                      "bagging_seed": 11, "metric": 'mae', "verbosity": -1, 'reg_alpha': 0.1, 'reg_lambda': 0.3,
                      'colsample_bytree': 1.0, 'n_estimators': 5000, 'n_jobs': -1}

        model = lgb.LGBMRegressor(**lgb_params)
        xmodel = XModel("lgb5", classifier=model, bag_mode=False, fit_on_validation=fit_on_validation, params=params,
                        generators=generators)
        ensemble.append(xmodel)

    if "lgb6" in models:  # makeRDKitFeatures
        params = {'seed': rseed, 'nsamples': -1, 'makeDistMat': True, 'makeTrainType': True,
                  'makeDistMean': True, 'makeMolNameMean': True, 'makeLabelEncode': True, 'makeRDKitFeatures': True,
                  'keepFeatures' : descRDkitTop80, 'dropFeatures': ['atom_index_0', 'atom_index_1', 'molblock']}

        lgb_params = {'num_leaves': 128, 'min_child_samples': 79, 'objective': 'regression', 'max_depth': 10,
                      'learning_rate': 0.2, "boosting_type": "gbdt", "subsample_freq": 1, "subsample": 1.0,
                      "bagging_seed": 11, "metric": 'mae', "verbosity": -1, 'reg_alpha': 0.1, 'reg_lambda': 0.3,
                      'colsample_bytree': 1.0, 'n_estimators': 5000, 'n_jobs': -1}

        model = lgb.LGBMRegressor(**lgb_params)
        xmodel = XModel("lgb6", classifier=model, bag_mode=False, fit_on_validation=fit_on_validation, params=params,
                        generators=generators)
        ensemble.append(xmodel)

    if "xgb1" in models:
        params = {'seed': rseed, 'nsamples': -1, 'makeDistMat': True, 'makeTrainType': True, 'plotDist': False,
                  'makeDistMean': True, 'makeMolNameMean': True, 'makeLabelEncode': True, 'makeRDKitFeatures': False,
                  'makeRDKitFingerPrints': False, 'oneHotenc': None, 'removeLowVariance': False,
                  'dropFeatures': ['atom_index_0', 'atom_index_1', 'molblock']}

        model = XGBRegressor(n_estimators=5000, learning_rate=0.01, max_depth=10, NA=0, subsample=.5,
                             colsample_bytree=1.0, min_child_weight=5, n_jobs=4,
                             eval_metric='mae', booster='gbtree', silent=1, eval_size=0.0)
        xmodel = XModel("xgb1", classifier=model, bag_mode=False, fit_on_validation=fit_on_validation, params=params,
                        generators=generators)
        ensemble.append(xmodel)

    """
    #tsnelib= {'numDims': 2, 'perplexity': 30, 'theta':0.5, 'pcaDims':21}
    #Xtest,Xtrain,ytrain,idx,sample_weight,Xval,yval = prepareDataset(quickload=False,data_id=data_id,seed=51176, nsamples='shuffle',removeCor = True,makeTSNE = tsnelib,renameFeatures = True)
    #model = LogisticRegression(C=1.0,penalty='l2')
    #model = Pipeline([('scaler', StandardScaler()), ('m',model)])
    #xmodel = XModel("lr3",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,Xval=Xval,yval=yval,cv_labels=None,bag_mode=False)
    #ensemble.append(xmodel)
    """

    return (ensemble)


def finalizeModel(m, use_proba=True,basedir="./data/"):
    """
    Make predictions and save them
    """
    print("Make predictions and save them...")

    print(m.summary())
    logging.info(m.summary())

    # put OOB DATA to data.frame and save
    m.oob_preds = pd.DataFrame(np.asarray(m.oob_preds), columns=['oob'])

    # store validation
    if hasattr(m, 'val_preds') and m.val_preds is not None:
        m.val_preds = pd.DataFrame(np.asarray(m.val_preds), columns=['val'])

    # TESTSET prediction
    m.preds = pd.DataFrame(np.asarray(m.preds), columns=['prediction'])

    # save final model
    allpred = pd.concat([m.preds, m.oob_preds],sort=False)
    # submission data is first, train data is last!
    filename =  basedir + m.name + ".csv"
    print("Saving oob + predictions as csv to:", filename)
    allpred.to_csv(filename, index=False)

    XModel.saveCoreData(m, basedir + m.name + ".pkl")
    return (m)


def saveTrainData(ensemble):
    """
    save training data
    """
    for m in ensemble:
        print("Saving data for model:", m.name)
        XModel.saveDataSet(m)


def loadDataSet(ensemble):
    """
    load saved training data
    """
    basedir = "./data/"
    for i, model in enumerate(ensemble):

        xmodel = XModel.loadModel(basedir + model)
        Xtrain, Xtest = XModel.loadDataSet(xmodel)
        print("model: %-20s %20r %20r %20r" % (xmodel.name, Xtrain.shape, Xtest.shape, type(xmodel.classifier)))


def createOOBdata(ensemble, repeats=1, n_folds=10, n_jobs=1, score_func='log_loss', verbose=False, calibrate=False,
                  use_proba=True):
    """
    Creates parallel out-of-bag prediction
    """

    global funcdict

    print("List of models:")
    for i,m in enumerate(ensemble):
        print(("%5d %s"%(i,m.name)))

    for m in ensemble:
        m.generate_features()

        bag_mode = m.bag_mode
        print(("\nComputing oob predictions for:"+m.name))
        # set nprocs=1 for NN
        print(m.classifier.get_params())
        if 'm__epochs' in m.classifier.get_params():
            print("Switching to serial...")
            n_jobs = 1

        if m.class_names is not None:
            n_classes = len(m.class_names)
            print(("Number of classes:"+n_classes))
        else:
            n_classes = 1

        oob_preds = np.zeros((m.ytrain.shape[0], n_classes, repeats),dtype=np.float32)
        preds = np.zeros((m.Xtest.shape[0], n_classes, repeats))
        val_preds = None
        if m.Xval is not None:
            val_preds = np.zeros((m.yval.shape[0], n_classes, repeats))

        oobscore = np.zeros(repeats)
        maescore = np.zeros(repeats)

        # outer loop
        for j in range(repeats):
            if m.cv_labels is not None:
                print("CV using labels...")
                cv = KLabelFolds(labels=m.cv_labels, n_folds=8, repeats=1)

            else:
                print("KFOLD  ...")
                cv = KFold(n_splits=n_folds, shuffle=True, random_state = 42)

            scores = np.zeros(n_folds)

            # parallel stuff
            parallel = Parallel(n_jobs=n_jobs, verbose=True,
                                pre_dispatch='2*n_jobs')

            # parallel run, returns a list of oob predictions
            results = parallel(
                delayed(fit_and_score)(clone(m.classifier), m.Xtrain, m.ytrain, train, test,
                                       sample_weight=m.sample_weight, use_proba=use_proba, returnModel=bag_mode) for train, test in cv.split(m.Xtrain))

            for i, (train, test) in enumerate(cv.split(m.Xtrain)):
                oob_pred, cv_model = results[i]
                scores[i] = funcdict[score_func](m.ytrain.iloc[test], oob_pred, m.Xtrain.iloc[test]['type'])
                if use_proba:
                    oob_pred = oob_pred[:,1]
                oob_pred = oob_pred.reshape(oob_pred.shape[0], n_classes)
                oob_preds[test, :, j] = oob_pred

                if bag_mode:
                    print("Using cv models for test set(bag_mode)...")
                    if use_proba:
                        p = cv_model.predict_proba(m.Xtest)
                        p = p.reshape(p.shape[0], n_classes)
                        preds[:, :, j] = p
                    else:
                        p = cv_model.predict(m.Xtest)
                        p = p.reshape(p.shape[0], n_classes)
                        preds[:, :, j] = p

                        if m.Xval is not None:
                            raise Exception("Currently not supported in Bag mode!")

            oobscore[j] = funcdict[score_func](m.ytrain, oob_preds[:, :, j].ravel(),m.Xtrain['type'])

            score_str = "Repeat: %2d <score>: %0.4f (+/- %0.4f) score,oob: %0.4f" % (j,scores.mean(), scores.std(),oobscore[j])
            print(score_str)
        logging.info(score_str)

        # simple averaging of blending
        m.oob_preds = np.mean(oob_preds, axis=2)

        score_oob = funcdict[score_func](m.ytrain, m.oob_preds.ravel(),m.Xtrain['type'])

        sum_str = "Summary: <score,oob>: %6.4f +- %6.4f   score,oob-total: %0.4f (after %d repeats)\n" % (
            oobscore.mean(), oobscore.std(), score_oob, repeats)
        print(sum_str)
        logging.info(sum_str)

        orig_classifier = clone(m.classifier)
        m.classifier = clone(orig_classifier)
        if not bag_mode:
            # Train full model on total train data
            print("Training on full train set...really?")
            Xtrain_ = m.Xtrain
            ly_ = m.ytrain
            if m.sample_weight is not None:
                print("... with sample weights")
                sample_weight_ = m.sample_weight

                m.classifier.fit(Xtrain_, ly_, sample_weight_)
            else:
                m.classifier.fit(Xtrain_, ly_)

            if m.Xval is not None:
                print("\n>>Prediction for val set... ", end=' ')
                if use_proba:
                    m.val_preds = m.classifier.predict_proba(m.Xval)[:,1]
                else:
                    m.val_preds = m.classifier.predict(m.Xval)
                # check
                score = funcdict[score_func](m.yval, m.val_preds,m.Xval['type'])
                val_str = "score,validation: %0.4f\n" % (score)
                print(val_str)
                logging.info(val_str)


            if m.Xval is not None and m.fit_on_validation:
                print("Re-training on train & val set...")
                Xtrain_ = pd.concat([m.Xtrain, m.Xval])
                ly_ = np.hstack((m.ytrain.ravel(), m.yval.ravel()))
                if m.sample_weight is not None:
                    raise Exception("Not supported for now...")

                # here we need to clone and retrain!
                m.classifier = clone(orig_classifier)
                m.classifier.fit(Xtrain_, ly_)
            else:
                print("Skipping validation set for model fit...")

            print("Predicting on test set...")
            if use_proba:
                m.preds = m.classifier.predict_proba(m.Xtest)[:,1]
            else:
                m.preds = m.classifier.predict(m.Xtest)

        else:
            print("bag_mode: averaging all cv classifier results")
            # print preds[:10]
            m.preds = np.mean(preds, axis=2)
            if m.Xval is not None:
                raise Exception("Currently not supported for Holdout....")

        m = finalizeModel(m, use_proba=use_proba)
        del oob_preds
        del preds
        if m.Xval is not None:
            del val_preds
    return ensemble


def fit_and_score(xmodel, X, y, train, valid, sample_weight=None, use_proba=False, returnModel=True):
    """
    Score function for parallel oob creation
    """
    if isinstance(X, pd.DataFrame):
        Xtrain = X.iloc[train]
        Xvalid = X.iloc[valid]
    else:
        Xtrain = X[train]
        Xvalid = X[valid]

    if isinstance(y, pd.Series):
        ytrain = y.iloc[train]
        yvalid = y.iloc[valid]
    else:
        ytrain = y[train]
        yvalid = y[valid]

    if sample_weight is not None:
        print("Using sample weight...", sample_weight[train])
        xmodel.fit(Xtrain, ytrain, sample_weight=sample_weight[train])
    else:
        xmodel.fit(Xtrain, ytrain)

    if use_proba:
        # saving out-of-bag predictions
        oob_pred = xmodel.predict_proba(Xvalid)
    # prediction for test set
    # classification/regression
    else:
        oob_pred = xmodel.predict(Xvalid)

    #mae = funcdict['group_mae'](yvalid, oob_pred, Xvalid['type'])
    #print("mae: %6.4f"%(mae))

    if returnModel:
        return oob_pred, xmodel
    else:
        return oob_pred, None


def trainEnsemble(ensemble,basedir = "./data_numerai/", mode='linear', score_func='log_loss', useCols=None, addMetaFeatures=False, use_proba=True,
                  dropCorrelated=False, skipCV=False, subfile="",model_dict=dict()):
    """
    Train the ensemble
    """

    for i, model in enumerate(ensemble):
        print(''.join(['-'] * 60))

        key = basedir + model
        if key not in model_dict:
            print("Loading model:", i, " name:", model)
            xmodel = XModel.loadModel(key)
            model_dict[key] = xmodel
        else:
            print("From cache -  model:", i, " name:", model)
            xmodel = model_dict[key]
        class_names = xmodel.class_names
        if class_names is None:
            class_names = ['Class']
        print("OOB data:", xmodel.oob_preds.shape)
        if hasattr(xmodel, 'val_preds') and xmodel.val_preds is not None:
            print("Holdout data:", xmodel.val_preds.shape)
        print("pred data:", xmodel.preds.shape)
        print("y train:", xmodel.ytrain.shape)

        if i > 0:
            xmodel.oob_preds.columns = [model + "_" + n for n in class_names]
            Xtrain = pd.concat([Xtrain, xmodel.oob_preds], axis=1)
            Xtest = pd.concat([Xtest, xmodel.preds], axis=1)
            if hasattr(xmodel, 'val_preds') and xmodel.val_preds is not None:
                Xval = pd.concat([Xval, xmodel.val_preds], axis=1)

        else:
            Xtrain = xmodel.oob_preds
            Xtest = xmodel.preds
            y = xmodel.ytrain
            colnames = [model + "_" + n for n in class_names]
            Xtrain.columns = colnames
            Xval = None
            yval = None
            if hasattr(xmodel, 'val_preds') and xmodel.val_preds is not None:
                Xval = xmodel.val_preds
                yval = xmodel.yval
                print(Xval.shape)

    Xtest.columns = Xtrain.columns
    if hasattr(xmodel, 'val_preds') and xmodel.val_preds is not None:
        Xval.columns = Xtrain.columns

    # print "spearman-correlation:\n",Xtrain.corr(method='spearman')
    if dropCorrelated: print("pearson-correlation :\n", Xtrain.corr(method='pearson'))

    if mode is 'classical':
        results = classicalBlend(ensemble, Xtrain, Xtest, y, valpreds=Xval, yval=yval,
                                 use_proba=use_proba, score_func=score_func,
                                 subfile=subfile,skipCV=skipCV, cv_labels=xmodel.cv_labels, dropCorrelated=dropCorrelated, fit_on_validation=xmodel.fit_on_validation)

    elif mode is 'mean':
        results = linearBlend(ensemble, Xtrain, Xtest, y, Xval=Xval, yval=yval, score_func=score_func, takeMean=True,
                              subfile=subfile,
                              dropCorrelated=dropCorrelated,fit_on_validation=xmodel.fit_on_validation)
    elif mode is 'voting':
        results = voting_multiclass(ensemble, Xtrain, Xtest, y, score_func=score_func, n_classes=1, subfile=subfile,
                                    dropCorrelated=dropCorrelated)
    elif mode is 'oob':
        return (Xtest, Xtrain, y, None, xmodel.cv_labels, None)

    else:
        results = linearBlend(ensemble, Xtrain, Xtest, y, Xval=Xval, yval=yval, score_func=score_func, takeMean=False,
                              subfile=subfile,
                              dropCorrelated=dropCorrelated)
    return (results,model_dict)


def voting_multiclass(ensemble, Xtrain, Xtest, y, n_classes=9, use_proba=False, score_func='log_loss', plotting=True,
                      subfile=None):
    """
    Voting for multi classifiction result
    """
    if use_proba:
        print("Majority voting for predictions using proba")
        voter = np.reshape(Xtrain.values, (Xtrain.shape[0], -1, n_classes)).swapaxes(0, 1)

        for model in voter:
            max_idx = model.argmax(axis=1)
            for row, idx in zip(model, max_idx):
                row[:] = 0.0
                row[idx] = 1.0

        voter = voter.mean(axis=0)
        print(voter)
        print(voter.shape)
    else:
        print("Majority voting for predictions")
        # assuming all classes are predicted
        if Xtrain.shape[1] % 2 == 0:
            print("Warning: Even number of voters...")

        classes = np.unique(Xtrain.values)

        votes_train = np.zeros((Xtrain.shape[0], classes.shape[0]))
        votes_test = np.zeros((Xtest.shape[0], classes.shape[0]))

        for i, c in enumerate(classes):
            votes_train[:, i] = np.sum(Xtrain.values == c, axis=1)
            votes_test[:, i] = np.sum(Xtest.values == c, axis=1)

        votes_train = np.argmax(votes_train, axis=1)
        votes_test = np.argmax(votes_test, axis=1)

        encoder = preprocessing.LabelEncoder()
        encoder.fit(y)
        ypred = encoder.inverse_transform(votes_train)
        preds = encoder.inverse_transform(votes_test)

        score = funcdict[score_func](y, ypred)
        print(score_func + ": %0.3f" % (score))

    if subfile is not None:
        analyze_predictions(ypred, preds)
        makePredictions(None, Xtest=preds, idx=idx,filename=subfile)

        if plotting:
            plt.hist(ypred, bins=50, alpha=0.3, label='oob')
            plt.hist(preds, bins=50, alpha=0.3, label='pred')
            plt.legend()
            plt.show()

    else:
        return score


def analyze_predictions(ypred, preds):
    # ypred = ypred.astype(int)
    plt.hist(ypred, bins=50, alpha=0.3, label='oob')
    plt.hist(preds, bins=50, alpha=0.3, label='pred')
    plt.legend()
    plt.show()


def preprocess(oobpreds, testset, verbose=False):
    # print "Clipping data  data..."
    # lowerb = 0.41
    # upperb = 6.91
    # oobpreds = oobpreds.clip(lower=lowerb,upper=upperb,axis=0)
    # testset = pd.DataFrame(np.clip(testset.values,lowerb, upperb))#all labels are the same!

    # overfittet models
    noise_columns = []  # ['bagxgb5_br1_Class','nn7_br25_Class']
    print("Adding random noise:", noise_columns)
    for col in noise_columns:
        if col in oobpreds.columns:
            oobpreds[col] = oobpreds[col].map(lambda x: x + np.random.normal(loc=0.0, scale=.05))

    if verbose:
        oobpreds.describe()
        showCorrelations(oobpreds)

    return oobpreds, testset



def classicalBlend(ensemble, oobpreds, testset, ly, valpreds=None, yval=None, use_proba=True, score_func='log_loss',subfile=None,skipCV=False,cv_labels=None, **kwargs):
    """
    Blending using sklearn API
    """
    showAVGCorrelations(oobpreds)

    if kwargs['dropCorrelated']:
        #showCorrelations(oobpreds)
        #oobpreds, testset, valpreds = removeCorrelations(oobpreds, testset,valpreds, 0.995)
        print(oobpreds.shape)

    #blender = XgboostClassifier(n_estimators=200, learning_rate=0.01, max_depth=2, NA=0, subsample=.5,colsample_bytree=1.0, min_child_weight=5, n_jobs=4, objective='binary:logistic',eval_metric='logloss', booster='gbtree', silent=1, eval_size=0.0)
    #blender = Pipeline([('rt', RandomTreesEmbedding(n_estimators=20, max_depth=1)),
    #                  ('m', LogisticRegression(C=1E-4, penalty='l2', solver='lbfgs'))])

    blender = RidgeCV()
    #blender = ConstrainedLinearRegressor(lowerbound=0, upperbound=.2, n_classes=1, alpha=None, corr_penalty=None,normalize=False, loss='log_loss', greater_is_better=False)  # 0.216467
    #blender =ExtraTreesRegressor(n_estimators=250, max_depth=6, min_samples_leaf=5, n_jobs=4, max_features=3, oob_score=False)
    #blender=RandomForestRegressor(n_estimators=100,max_depth=None,min_samples_leaf=10,n_jobs=1,criterion='entropy', max_features=5,oob_score=False)
    #blender = KerasClassifier(build_fn=create_classification_model, layers=[4], dropouts=[0.0], input_dim=oobpreds.shape[1], activation='relu', learning_rate=0.25, epochs=25, batch_size=1024, verbose=2, validation_split=0.0)
    #blender = KerasClassifier(build_fn=create_classification_model, layers=[64, 64], dropouts=[0.5, 0.5],input_dim=oobpreds.shape[1], activation='relu', learning_rate=0.2, epochs=25,batch_size=512, verbose=0, validation_split=0.0)
    #blender = Pipeline([('scaler', StandardScaler()), ('m',blender)])
    #blender = BaggingClassifier(base_estimator=blender, n_estimators=20, n_jobs=1, verbose=0, random_state=None,max_samples=0.75, max_features=1.0, bootstrap=False)

    if not skipCV:
        n_folds = 5
        cv = KFold(n_splits=n_folds, shuffle=True, random_state = 42)
        parameters = {'m__C': [1E-2,1E-3,1E-4,1E-5,1E-6]}
        #parameters = {'m__epochs': [25], 'm__layers': [[4],[2]], 'm__dropouts': [[0.0]],
        #              'm__batch_size': [1024], 'm__learning_rate': [0.3,0.25], 'm__activation': ['relu']}
        #parameters = {'n_estimators':[200],'max_depth':[2,3],'learning_rate':[0.01,0.1,0.001],'subsample':[0.5],'colsample_bytree':[1.0],'min_child_weight':[5]}#XGB
        # parameters = {'n_estimators':[200,300],'max_features':[5,7],'min_samples_leaf':[1,5,10],'criterion':['mse']}#XGB
        # parameters = {'max_features':[0.9,0.95,1.0],'max_samples':[0.9,0.95,1.0],'bootstrap':[False,True]}#XGB
        # parameters = {'model__hidden1_num_units': [128],'model__dropout1_p':[0.0],'model__hidden2_num_units': [128],'model__dropout2_p':[0.0],'model__max_epochs':[75],'model__objective_alpha':[0.002]}
        #parameters = {'model__max_epochs':[5,10,15]}
        #blender=makeGridSearch(blender,oobpreds,ly,n_jobs=1,refit=True,cv=cv,scoring=score_func,parameters=parameters,random_iter=-1)
        #buildXvalModel(blender,oobpreds,ly,sample_weight=None,class_names=None,refit=True,cv=cv)
        blend_scores = np.zeros(n_folds)
        n_classes = 1
        blend_oob = np.zeros((oobpreds.shape[0], n_classes))

        for i, (train, test) in enumerate(cv.split(oobpreds)):
            clf = clone(blender)
            Xtrain = oobpreds.iloc[train]
            Xtest = oobpreds.iloc[test]
            clf.fit(Xtrain.values, ly.iloc[train])
            if use_proba:
                t = clf.predict_proba(Xtest)[:,1]
                blend_oob[test] = t.reshape(blend_oob[test].shape)
            else:
                blend_oob[test] = clf.predict(Xtest).reshape(blend_oob[test].shape)
            blend_scores[i] = funcdict[score_func](ly.iloc[test], blend_oob[test].ravel(),Xtest['type_Class'])
            print("Fold: %3d <%s>: %0.6f ~mean: %6.4f std: %6.4f" % (
                i, score_func, blend_scores[i], blend_scores[:i + 1].mean(), blend_scores[:i + 1].std()))

        print(" <" + score_func + ">: %0.5f (+/- %0.4f)" % (blend_scores.mean(), blend_scores.std()), end=' ')
        oob_auc = funcdict[score_func](ly, blend_oob.ravel(),oobpreds['type_Class'])
        # showMisclass(ly,blend_oob,oobpreds,index=kwargs['cv_labels'])
        print("\ntotal out-of-bag " + score_func + ": %0.5f" % (oob_auc))

        penalty = 0.0
        if subfile is not None or kwargs['fit_on_validation'] is not None:
            print("Make model fit on full out-of-bag data...")
            blender.fit(oobpreds, ly)
            if valpreds is not None:
                print("Evaluate full model on validation data...", end=' ')
                if use_proba:
                    y_val_pred = blender.predict_proba(valpreds)[:,1]
                else:
                    y_val_pred = blender.predict(valpreds)
                score = funcdict[score_func](yval, y_val_pred,valpreds['type_Class'])
                print(" " + score_func + ": %0.5f" % (score))

                if kwargs['fit_on_validation']:
                    print("Make model fit on out-of-bag & validation data...")

                    oobpreds = pd.concat([oobpreds, valpreds], axis=0)
                    ly = np.hstack((ly.ravel(), yval.ravel()))
                    blender.fit(oobpreds, ly)

                else:
                    print("Skipping fit on validation set..")

        if hasattr(blender, 'coef_'):
            print("%-3s %-24s %10s %10s" % ("nr", "model", score_func, "coef"))
            for i, model in enumerate(oobpreds.columns):
                coldata = np.asarray(oobpreds.iloc[:, i])
                score = funcdict[score_func](ly, coldata,oobpreds['type_Class'])
                print("%-3d %-24s %10.4f%10.4f" % (i + 1, model.replace("_Class", ""), score, blender.coef_.flatten()[i]))
            print("sum coef: %4.4f" % (np.sum(blender.coef_)))

        if subfile is not None:
            info_dist(ly, "orig")
            info_dist(blender.predict(oobpreds), "fit")

    if subfile is not None:
        print("Make final ensemble prediction...")
        # blend results
        if use_proba:
            preds = blender.predict_proba(testset)[:,1]
        else:
            preds = blender.predict(testset)
            preds = preds.flatten()

        info_dist(preds, "preds")
        makePredictions(None, preds, filename=subfile)
        #analyze_predictions(blend_oob, preds)

    return (blend_scores.mean())


def multiclass_mult(Xtrain, params, n_classes):
    """
    Multiplication rule for multiclass models
    """
    ypred = np.zeros((len(params), Xtrain.shape[0], n_classes))
    for i, p in enumerate(params):
        idx_start = n_classes * i
        idx_end = n_classes * (i + 1)
        ypred[i] = Xtrain.iloc[:, idx_start:idx_end] * p
    ypred = np.mean(ypred, axis=0)
    return ypred


def blend_mult(Xtrain, params, n_classes=None):
    if n_classes < 2:
        return np.dot(Xtrain, params)
    else:
        return multiclass_mult(Xtrain, params, n_classes)


def linearBlend(ensemble, Xtrain, Xtest, y, Xval=None, yval=None, score_func='log_loss', greater_is_better=False,use_proba=True,normalize=False, removeZeroModels=-1, takeMean=False, alpha=None, subfile=None, plotting=False, **kwargs):
    """
    Blending for multiclass systems
    """

    def fopt(params):
        # nxm  * m*1 ->n*1
        if np.isnan(np.sum(params)):
            print("We have NaN here!!")
            score = 0.0
        else:
            ypred = blend_mult(Xtrain, params, n_classes)
            # if not use_proba: ypred = np.round(ypred).astype(int)
            score = funcdict[score_func](y, ypred,Xtrain['type_Class'])
            #print "orig score:%8.5f" % (score)
            # regularization
            if alpha is not None:
                penalty = alpha * np.sum(np.square(params))

                score = score - penalty
                print(" - Regularization - alpha: %8.3f penalty: %8.3f regularized score: %8.3f" % (
                    alpha, penalty, score))
            if greater_is_better: score = -1 * score
        return score

    y = np.asarray(y)
    n_models = len(ensemble)
    n_classes = Xtrain.shape[1] / len(ensemble)

    lowerbound = 0.0
    upperbound = 1.0
    #constr = None
    constr = [lambda x, z=i: x[z] - lowerbound for i in range(n_models)]
    constr2 = [lambda x, z=i: upperbound - x[z] for i in range(n_models)]
    constr = constr + constr2

    #cons = ({'type': 'ineq', 'fun': [lambda x, z=i: x[z] - lowerbound for i in range(n_models)]}, {'type': 'ineq', 'fun': [lambda x, z=i: upperbound - x[z] for i in range(n_models)]})

    x0 = np.ones((n_models, 1)) / float(n_models)

    if not takeMean:
        xopt = fmin_cobyla(fopt, x0, constr, rhoend=1e-5, maxfun=200)
        # xopt = minimize(fopt, x0,method='Nelder-Mead')
        # xopt = minimize(fopt, x0,method='COBYLA',constraints=cons)
        print(xopt)
    # xopt = xopt.x
    else:
        xopt = x0

    # normalize coefficient
    if normalize:
        xopt = xopt / np.sum(xopt)
        print("Normalized coefficients:", xopt)

    if np.isnan(np.sum(xopt)):
        print("We have NaN here!!")

    ypred = blend_mult(Xtrain, xopt, n_classes)
    ymean = np.mean(Xtrain.values, axis=1)
    # ymean=np.median(Xtrain.values,axis=1)

    if takeMean:
        print("Taking the mean/median...")
        ypred = ymean

    score = funcdict[score_func](y, ymean, Xtrain['type_Class'])
    pscore = funcdict['mae'](y, ymean)
    print(">>score,mean: %4.4f" % (score))
    if n_classes==1: ypred = ypred.flatten()
    oob_score = funcdict[score_func](y, ypred, Xtrain['type_Class'])
    print(">>score,opt: %4.4f" % (oob_score))

    #plotting = True
    #if plotting:
    #    plot_types(y, ymean, Xtrain['type_Class'].astype(int))

    if Xval is not None:
        print("\nEvaluating on validation set...")
        yval_mean = np.mean(Xval.values, axis=1)
        pred_score = funcdict[score_func](yval, yval_mean, Xval['type_Class'])
        ppred_score = funcdict['mae'](yval, yval_mean)
        print(">>score,mean: %4.4f" % (pred_score))
        yval_pred = blend_mult(Xval, xopt, n_classes)
        if n_classes ==1: yval_pred = yval_pred.flatten()
        pred_score = funcdict[score_func](yval, yval_pred, Xval['type_Class'])
        print(">>score,opt: %4.4f" % (pred_score))

        #if plotting:
        #    plot_types(yval, yval_mean, Xval['type_Class'].astype(int))

    zero_models = []
    print("%4s %-48s %7s %6s" % ("nr", "model", "score", "coeff"))
    for i, model in enumerate(ensemble):
        n_classes = int(n_classes) # not sure when n_classes becomes float..
        idx_start = n_classes * i
        idx_end = n_classes * (i + 1)
        coldata = np.asarray(Xtrain.iloc[:, idx_start:idx_end])
        if n_classes == 1: coldata = coldata.flatten()
        score = funcdict[score_func](y, pd.Series(coldata), Xtrain['type_Class'])
        print("%4d %-48s %7.4f %6.3f" % (i + 1, model, score, xopt[i]), end='')
        if xopt[i] < removeZeroModels:
            zero_models.append(model)
        if Xval is not None:
            coldata_val = np.asarray(Xval.iloc[:, idx_start:idx_end])
            if n_classes == 1: coldata_val = coldata_val.flatten()
            score = funcdict[score_func](yval, pd.Series(coldata_val), Xval['type_Class'])
            print("(val: %6.4f)" % (score))
        else:
            print("")

    print("##sum coefficients: %4.4f" % (np.sum(xopt)))

    if removeZeroModels > 0.0:
        print("Dropping ", len(zero_models), " columns:", zero_models)
        Xtrain = Xtrain.drop(zero_models, axis=1)
        Xtest = Xtest.drop(zero_models, axis=1)
        return (Xtrain, Xtest)

    # prediction flatten makes a n-dim row vector from a nx1 column vector...
    if takeMean:
        print("Taking the mean/median for predictions...")
        preds = np.mean(Xtest.values, axis=1)
    else:
        preds = blend_mult(Xtest, xopt, n_classes).flatten()
    # if not use_proba: preds = np.round(preds).astype(int)


    if subfile is not None:
        info_dist(y, "orig")
        info_dist(ypred, "fit ")
        info_dist(preds, "pred")
        #plt.hist(y,bins=50)
        #plt.hist(preds,bins=50)
        #plt.show()

        makePredictions(None, Xtest=preds,filename=subfile)
    else:
        if Xval is not None:
            #print "Returning the validation score...!"
            return oob_score
        else:
            return oob_score


def info_dist(y, info):
    print(info + "-  max: %4.2f mean: %4.2f median: %4.2f min: %4.2f" % (np.amax(y), y.mean(), np.median(y), np.amin(y)))


def selectModels(ensemble, startensemble=[], niter=10, mode='linear', useCols=None):
    """
    Random mode for best model selection
    """
    randBinList = lambda n: [randint(0, 1) for b in range(1, n + 1)]
    auc_list = [0.0]
    ens_list = []
    cols_list = []
    for i in range(niter):
        print("iteration %5d/%5d, current max_score: %6.3f" % (i + 1, niter, max(auc_list)))
        actlist = randBinList(len(ensemble))
        actensemble = [x for x in itertools.compress(ensemble, actlist)]
        actensemble = startensemble + actensemble
        print(actensemble)
        # print actensemble
        score = trainEnsemble(actensemble, mode=mode, useCols=useCols, addMetaFeatures=False, dropCorrelated=False)
        auc_list.append(score)
        ens_list.append(actensemble)
    # cols_list.append(actCols)
    max_score = 0.0
    topens = None
    topcols = None
    for ens, score in zip(ens_list, auc_list):
        print("SCORE: %4.4f" % (score), end=' ')
        print(ens)
        if score > max_score:
            maxauc = score
            topens = ens
            # topcols=col
    print("\nTOP ensemble:", topens)
    print("TOP score: %4.4f" % (max_score))


def selectModelsGreedy(ensemble, startensemble=[], niter=2, mode='mean', useCols=None, dropCorrelated=False,
                       greater_is_better=False,score_func='log_loss', replacement=False):
    """
    Select best models in a greedy forward selection
    """

    model_dict = dict()

    topensemble = startensemble
    score_list = []
    ens_list = []
    if greater_is_better:
        bestscore = 0.0
    else:
        bestscore = 1E15
    for i in range(niter):
        if greater_is_better:
            maxscore = 0.0
        else:
            maxscore = 1E15
        topidx = -1
        for j in range(len(ensemble)):
            if not replacement:
                if ensemble[j] not in topensemble:
                    actensemble = topensemble + [ensemble[j]]
                else:
                    continue
            else:
                actensemble = topensemble + [ensemble[j]]

            score, model_dict = trainEnsemble(actensemble, mode=mode, score_func=score_func, use_proba=True, useCols=None,
                                  subfile=None, dropCorrelated=dropCorrelated,model_dict=model_dict)
            print("##(Current top score: %7.5f | overall best score: %7.5f) current score: %4.4f  - " % (
                maxscore, bestscore, score))
            if greater_is_better:
                if score > maxscore:
                    maxscore = score
                    topidx = j
            else:
                if score < maxscore:
                    maxscore = score
                    topidx = j

        # pick best set
        # if not maxscore+>bestscore:
        #    print "Not gain in score anymore, leaving..."
        #    break
        topensemble.append(ensemble[topidx])
        print("TOP score: %7.5f" % (maxscore), end=' ')
        print(" - actual ensemble:", topensemble)
        score_list.append(maxscore)
        ens_list.append(list(topensemble))
        if greater_is_better:
            if maxscore > bestscore:
                bestscore = maxscore
        else:
            if maxscore < bestscore:
                bestscore = maxscore

    for ens, score in zip(ens_list, score_list):
        print("SCORE: %7.5f" % (score), end=' ')
        print(ens)

    plt.plot(score_list)
    plt.show()
    return topensemble

# def updatePredictions(model_list=[],basedir='./data/',storedir='./data/',score_func='log_loss',use_proba=True):
#     #download_data(numerai.DATA_ID)
#     for model in model_list:
#         xmodel = XModel.loadModel(basedir+model)
#         xmodel.summary()
#         xmodel.params['data_id'] = numerai.DATA_ID
#         model = createPredictionData(xmodel,score_func,use_proba=use_proba)
#         finalizeModel(model, use_proba=True, basedir=storedir)

def createPredictionData(m,score_func,use_proba=True):
    #Reset old prediction
    m.preds = None
    m.Xtest = None
    m.ytest = None
    m.Xval = None
    m.yval = None

    # numerai training data is the same but shuffled
    # hence backup old training data!
    y_bk = m.ytrain.copy()
    oob_preds_bk = m.oob_preds.copy()
    m.generate_features()
    m.ytrain = y_bk
    m.oob_preds = oob_preds_bk
    score_oob = funcdict[score_func](m.ytrain, m.oob_preds)
    print(">>score,oob-total,after: %0.4f\n" % (score_oob))

    #print m.generators
    print(type(m.classifier))
    classifier = m.classifier
    #predict new data
    if m.Xval is not None:
        print("\n>>Prediction on val set...", end=' ')
        if use_proba:
            m.val_preds = classifier.predict_proba(m.Xval)[:, 1]
        else:
            m.val_preds = classifier.predict(m.Xval)
        # check
        score = funcdict[score_func](m.yval, m.val_preds)
        print(" score,validation set: %0.4f\n" % (score))

    print("Prediction on test set...")
    if use_proba:
        m.preds = m.classifier.predict_proba(m.Xtest)[:, 1]
    else:
        m.preds = m.classifier.predict(m.Xtest)

    return m


def main():
    logging.basicConfig(filename='log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    np.random.seed(123)
    #make_keras_picklable()
    subfile = 'nmr.csv'
    plt.interactive(False)

    models = ['ridge1','xgb1','lgb1','lgb2','lgb3','lgb4','type']
    act_model = ['lgb6']

    """
    # 1nd LEVEL MODEL BUILDING
    """
    ensemble = createModels(act_model)
    ensemble = createOOBdata(ensemble, repeats=1, n_folds=5, n_jobs=1, use_proba=False,score_func='group_mae')
    #createDataSets()
    #saveTrainData(ensemble)
    basedir = './data/'

    """
    # 1nd LEVEL ENSEMBLING
    """

    #trainEnsemble(models, basedir=basedir, mode='classical', score_func='group_mae', useCols=None, addMetaFeatures=False, use_proba=False, dropCorrelated=False, subfile=subfile)

    # selectModelsGreedy(act_model,startensemble=[],niter=12,mode='mean',greater_is_better=False, replacement = True)
    #upload_submission(filename=subfile, account=account)


if __name__ == "__main__":
    np.random.seed(123)
    plt.interactive(False)
    main()

