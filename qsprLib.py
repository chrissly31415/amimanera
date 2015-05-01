#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
Code fragment collection for QPSR. Using sklearn, pandas and numpy
"""

from time import time
import itertools
from random import choice

import numpy as np
import numpy.random as nprnd
import pandas as pd
import scipy as sp
from scipy import sparse

import matplotlib.pyplot as plt
import matplotlib as mpl
import pylab as pl

from sklearn.preprocessing import StandardScaler,PolynomialFeatures,OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer,HashingVectorizer,TfidfVectorizer
from sklearn.feature_extraction import FeatureHasher
#from sklearn import metrics
from sklearn import cross_validation,grid_search
from sklearn.cross_validation import StratifiedKFold,KFold,StratifiedShuffleSplit,ShuffleSplit,train_test_split
from sklearn.metrics import roc_auc_score,classification_report,make_scorer,f1_score,precision_score,mean_squared_error,accuracy_score,log_loss
#from sklearn.utils.extmath import density
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import TruncatedSVD,PCA,RandomizedPCA, FastICA, MiniBatchSparsePCA, SparseCoder,DictionaryLearning,MiniBatchDictionaryLearning
from sklearn.pipeline import Pipeline

from sklearn.feature_selection import SelectKBest,SelectPercentile, chi2, f_classif,f_regression,GenericUnivariateSelect,VarianceThreshold
from sklearn.naive_bayes import BernoulliNB,MultinomialNB,GaussianNB
from sklearn.cluster import k_means
from sklearn.isotonic import IsotonicRegression

from sklearn.linear_model import LogisticRegression,RandomizedLogisticRegression,SGDClassifier,Perceptron,SGDRegressor,RidgeClassifier,LinearRegression,Ridge,BayesianRidge,ElasticNet,RidgeCV,LassoLarsCV,Lasso,LarsCV
from sklearn.cross_decomposition import PLSRegression,PLSSVD
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,ExtraTreesClassifier,AdaBoostClassifier,ExtraTreesRegressor,GradientBoostingRegressor,BaggingRegressor,BaggingClassifier,RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.svm import LinearSVC,SVC,SVR

from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn.learning_curve import learning_curve
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier

from sklearn.base import clone

from sklearn.calibration import CalibratedClassifierCV

    
def removeCorrelations(X_all,threshhold):
    """
    Remove correlations, we could improve it by only removing the variable frmo two showing the highest correlations with others
    """
    print "Removing correlated columns with threshhold:",threshhold
    c = X_all.corr().abs()
    #print c
    corcols={}
    for col in range(len(c.columns)):
	for row in range(len(c.index)):
	    if c.columns[col] in corcols or c.index[row] in corcols:
		continue
	    if row<=col:
		continue
	    if c.iloc[row,col]<1 and c.iloc[row,col]>threshhold:
		corcols[c.index[row]]=c.columns[col]+" :"+str("%4.3f"%(c.iloc[row,col]))

    for el in corcols.keys():
	print "Dropped: %32s due to Col1: %32s"%(el,corcols[el])
    X_all=X_all.drop(corcols,axis=1)
    return(X_all)
    
    
def makePredictions(lmodel,lXs_test,lidx,filename):
    """
    Uses priorily fit model to make predictions
    """
    print "Saving predictions to: ",filename
    print "Final test dataframe:",lXs_test.shape
    preds = lmodel.predict_proba(lXs_test)[:,1]
    pred_df = pd.DataFrame(preds, index=lidx, columns=['label'])
    pred_df.to_csv(filename)
    

def make_calibration(lmodel,Xtrain,ytrain):
    """
    Make Platt/isotonic calibration with sklearn
    """
    calli_clf = CalibratedClassifierCV(lmodel, method='isotonic', cv=3)
    calli_clf.fit(Xtrain,ytrain)
    print calli_clf.calibrated_classifiers_
    

    
def analyzeModel(lmodel,feature_names):
    """
    Analysis of data if feature_names are available
    """
    if hasattr(lmodel, 'coef_'): 
	print("Analysis of data...")
	print("Dimensionality: %d" % lmodel.coef_.shape[1])
	print("Density: %4.3f" % density(lmodel.coef_))
	if feature_names is not None:
	  top10 = np.argsort(lmodel.coef_)[0,-10:][::-1]
	  #print model.coef_[top10b]
	  for i in xrange(top10.shape[0]):
	      print("Top %2d: coef: %0.3f %20s" % (i+1,lmodel.coef_[0,top10[i]],feature_names[top10[i]]))
	      

def sigmoid(z):
    """
    classical sigmoid
    """
    g = 1.0/(1.0+np.exp(-z));
    return(g)
	         
def ensembleBuilding(lXs,ly):
    """
    train ensemble
    """
    print "Ensemble training..."
    folds=8
    parameters=np.logspace(-15, -10, num=200, base=2.0)
    #iterpos=[10,20,30,40,50]
    #regula=[0.001,0.1,1.0,10.0,100.0]
    parameters=nprnd.choice([0.1,10,1.0],10)
    loss=['l2','l2']
    #parameters=nprnd.choice(parameters, 20)
    classifiers = {}
    for p in parameters:
        l1ratio=nprnd.ranf()
        perc=85.0+nprnd.ranf()*15.0
        #iterations=choice(iterpos)
        l1l2=choice(loss)
	dic = {'LOG_C'+str(p)+'_perc'+str(perc)+'_loss'+str(l1l2): Pipeline([('filter', SelectPercentile(chi2, percentile=perc)), ('model', LogisticRegression(penalty=l1l2, tol=0.0001, C=p))])}
	#dic ={'SDG_alpha'+str(p)+'_L1'+str(l1ratio): SGDClassifier(alpha=p, n_iter=choice(iterpos),penalty='elasticnet',l1_ratio=l1ratio,shuffle=True,random_state=np.random.randint(0,100),loss='log')}
	#dic ={'PIP_SDG_iter'+str(p)+'_perc'+str(perc): Pipeline([('filter', SelectPercentile(chi2, percentile=perc)), ('model', SGDClassifier(alpha=0.00014, n_iter=p,shuffle=True,random_state=p,loss='log',penalty='elasticnet',l1_ratio=l1ratio))])}
	#dic ={'PIP_SDG_alpha'+str(p)+'_perc'+str(perc)+'_iter'+str(iterations): Pipeline([('filter', SelectPercentile(chi2, percentile=perc)),('model', SGDClassifier(alpha=p, n_iter=iterations,penalty='elasticnet',l1_ratio=l1ratio,shuffle=True,random_state=np.random.randint(0,100),loss='log'))])}
	classifiers.update(dic)
    #dic ={'NB': BernoulliNB(alpha=1.0)}
    #classifiers.update(dic)
    dic ={'LG1': LogisticRegression(penalty='l2', tol=0.0001, C=1.0)}
    classifiers.update(dic)
    dic ={'SDG1': SGDClassifier(alpha=0.0001, n_iter=50,shuffle=True,loss='log',penalty='l2')}
    classifiers.update(dic)
    dic ={'SDG2': SGDClassifier(alpha=0.00014, n_iter=50,shuffle=True,loss='log',penalty='elasticnet',l1_ratio=0.99)}
    classifiers.update(dic)
    #dic ={'LG2': LogisticRegression(penalty='l1', tol=0.0001, C=1.0,random_state=42)}
    #classifiers.update(dic)
    dic = {'NB2': Pipeline([('filter', SelectPercentile(f_classif, percentile=25)), ('model', BernoulliNB(alpha=0.1))])}
    classifiers.update(dic)    
    #dic ={'KNN': Pipeline([('filter', SelectPercentile(f_classif, percentile=25)), ('model', KNeighborsClassifier(n_neighbors=150))])}
    #classifiers.update(dic)  
    
    oobpreds=np.zeros((lXs.shape[0],len(classifiers)))
    for j,(key, lmodel) in enumerate(classifiers.iteritems()):
        #print lmodel.get_params()
        #cv = StratifiedKFold(ly, n_folds=folds, indices=True)
        cv = KFold(lXs.shape[0], n_folds=folds, indices=True,random_state=j,shuffle=True)
	scores=np.zeros(folds)	
	for i, (train, test) in enumerate(cv):
	    Xtrain = lXs[train]
	    Xtest = lXs[test]
	    lmodel.fit(Xtrain, ly[train])
	    oobpreds[test,j] = lmodel.predict_proba(Xtest)[:,1]
	    scores[i]=roc_auc_score(ly[test],oobpreds[test,j])
	    #print "AUC: %0.2f " % (scores[i])
	    #save oobpredictions
	print "Iteration:",j," model:",key,
	print " <AUC>: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std()),
	oobscore=roc_auc_score(ly,oobpreds[:,j])
	print " AUC oob: %0.3f" %(oobscore)
	
    scores=[roc_auc_score(ly,oobpreds[:,j]) for j in xrange(len(classifiers))]
    #simple averaging of blending
    oob_avg=np.mean(oobpreds,axis=1)
    print " AUC oob, simple mean: %0.3f" %(roc_auc_score(ly,oob_avg))
    
    #do another crossvalidation for weights and train blender...?
    #blender=LogisticRegression(penalty='l2', tol=0.0001, C=1.0)
    blender=AdaBoostClassifier(learning_rate=0.1,n_estimators=150,algorithm="SAMME.R")
    #blender=ExtraTreesRegressor(n_estimators=200,max_depth=None,n_jobs=1, max_features='auto',oob_score=False,random_state=42)
    #blender=ExtraTreesClassifier(n_estimators=200,max_depth=None,min_samples_leaf=10,n_jobs=1,criterion='entropy', max_features='auto',oob_score=False,random_state=42)
    cv = KFold(lXs.shape[0], n_folds=folds, indices=True,random_state=42)
    blend_scores=np.zeros(folds)
    blend_oob=np.zeros((lXs.shape[0]))
    for i, (train, test) in enumerate(cv):
	Xtrain = oobpreds[train]
	Xtest = oobpreds[test]
	blender.fit(Xtrain, ly[train])
	if hasattr(blender,'predict_proba'):
	    blend_oob[test] = blender.predict_proba(Xtest)[:,1]
	else:
	    blend_oob[test] = blender.predict(Xtest)
	blend_scores[i]=roc_auc_score(ly[test],blend_oob[test])
    print " <AUC>: %0.3f (+/- %0.3f)" % (blend_scores.mean(), blend_scores.std()),
    print " AUC oob after blending: %0.3f" %(roc_auc_score(ly,blend_oob))
    if hasattr(blender,'coef_'):
      print "Coefficients:",blender.coef_
    
    plt.plot(range(len(classifiers)),scores,'ro')
    return(classifiers,blender,oob_avg)
    
def ensemblePredictions(classifiers,blender,lXs,ly,lXs_test,lidx,lidx_train,oob_avg,filename):
    """   
    Makes prediction
    """ 
    print "Make final ensemble prediction..."
    #make prediction for each classifiers
    preds=np.zeros((lXs_test.shape[0],len(classifiers)))
    for j,(key, lmodel) in enumerate(classifiers.iteritems()):
        lmodel.fit(lXs,ly)
	preds[:,j]=lmodel.predict_proba(lXs_test)[:,1]
    #blend results
    finalpred=blender.predict_proba(preds)[:,1]   
    print "Saving predictions to: ",filename
    print "Final test dataframe:",lXs_test.shape
    oob_avg = pd.DataFrame(oob_avg,index=lidx_train,columns=['label'])
    pred_df = pd.DataFrame(finalpred, index=lidx, columns=['label'])
    pred_df = pd.concat([pred_df, oob_avg])
    pred_df.index.name='urlid'
    pred_df.to_csv(filename)
    

def weightedGridsearch(lmodel,lX,ly,lw,fitWithWeights=False,nfolds=5,useProba=False,scale_wt='auto',n_jobs=1,local_scorer='roc_auc'):
    """
    Uses sample weights and individual scoring function, used in Higgs challenge, needs modification cross_Validation.py
    """
    if not 'sample_weight' in inspect.getargspec(lmodel.fit).args:
	  print("WARNING: Fit function ignores sample_weight!")
	  
    fit_params = {}	
    fit_params['scoring_weight']=lw	
    fit_params['fitWithWeights']=fitWithWeights
    
    #parameters = {'n_estimators':[150,300], 'max_features':[5,10]}#rf
    #parameters = {'n_estimators':[250], 'max_features':[6,8,10],'min_samples_leaf':[5,10]}#xrf+xrf
    #parameters = {'max_depth':[7], 'learning_rate':[0.08],'n_estimators':[100,200,300],'subsample':[0.5],'max_features':[10],'min_samples_leaf':[50]}#gbm
    #parameters = {'max_depth':[7], 'learning_rate':[0.08],'n_estimators':[200],'subsample':[1.0],'max_features':[10],'min_samples_leaf':[20]}#gbm
    parameters = {'max_depth':[6], 'learning_rate':[0.1,0.08,0.05],'n_estimators':[300,500,800],'subsample':[1.0],'loss':['deviance'],'min_samples_leaf':[100],'max_features':[8]}#gbm
    #parameters = {'max_depth':[10], 'learning_rate':[0.001],'n_estimators':[500],'subsample':[0.5],'loss':['deviance']}#gbm
    #parameters = {'max_depth':[15,20,25], 'learning_rate':[0.1,0.01],'n_estimators':[150,300],'subsample':[1.0,0.5]}#gbm
    #parameters = {'max_depth':[20,30], 'learning_rate':[0.1,0.05],'n_estimators':[300,500,1000],'subsample':[0.5],'loss':['exponential']}#gbm
    #parameters = {'max_depth':[15,20], 'learning_rate':[0.05,0.01,0.005],'n_estimators':[250,500],'subsample':[1.0,0.5]}#gbm
    #parameters = {'n_estimators':[100,200,400], 'learning_rate':[0.1,0.05]}#adaboost
    #parameters = {'filter__percentile':[20,15]}#naives bayes
    #parameters = {'filter__percentile': [15], 'model__alpha':[0.0001,0.001],'model__n_iter':[15,50,100],'model__penalty':['l1']}#SGD
    #parameters['model__n_neighbors']=[40,60]}#knn
    #parameters['model__alpha']=[1.0,0.8,0.5,0.1]#opt nb
    #parameters = {'n_neighbors':[10,30,40,50],'algorithm':['ball_tree'],'weights':['distance']}#knn
    clf_opt=grid_search.GridSearchCV(lmodel, parameters,n_jobs=n_jobs,verbose=1,scoring=local_scorer,cv=nfolds,fit_params=fit_params,refit=True)   
    clf_opt.fit(lX,ly)
    #dir(clf_opt)
    for params, mean_score, scores in clf_opt.grid_scores_:       
        print("%0.3f (+/- %0.3f) for %r" % (mean_score, scores.std(), params))
    
    scores = cross_validation.cross_val_score(lmodel, lX, ly, fit_params=fit_params,scoring=local_scorer,cv=nfolds)
    print "Score: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std())
    return(clf_opt.best_estimator_)
    
    
    
def buildRegressionModel(lmodel,lXs,ly,sample_weights=None,scoring=None,cv=5):
    """   
    Final model building part
    """ 
    print "Xvalidation..."
    scores = cross_validation.cross_val_score(lmodel, lXs, ly, cv=cv, scoring=scoring,n_jobs=1)
    #scores = (-1*scores)**0.5
    print "SCORE: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std())
    print "Building model with all instances..."
    lmodel.fit(lXs,ly)
    return(lmodel)


def buildModel(clf,lX,ly,cv=None,scoring=None,n_jobs=8,trainFull=True,verbose=False):
    score = cross_validation.cross_val_score(clf,lX,ly,fit_params=None, scoring=scoring,cv=cv,n_jobs=n_jobs)
    if verbose: print "cv-score: %6.3f +/- %6.3f" %(-1*score.mean(),score.std())
    if trainFull:
      print "Train on all data..."
      clf.fit(lX,ly)
      return(clf)
    else:
      return score

def buildClassificationModel(clf_orig,lX,ly,class_names=None,trainFull=False,cv=None):
  """   
  Final model building part
  """ 
  print "Training the model..."
  print class_names
  if isinstance(lX,pd.DataFrame): lX  = lX.values

  ypred = np.zeros((len(ly),))
  yproba = np.zeros((len(ly),len(set(ly))))
  mll = np.zeros((len(cv),1))
  for i,(train, test) in enumerate(cv):  
      clf = clone(clf_orig)
      ytrain, ytest = ly[train], ly[test]
      clf.fit(lX[train,:], ytrain)
      ypred[test] = clf.predict(lX[test,:])
      yproba[test] = clf.predict_proba(lX[test,:])
      mll[i] = multiclass_log_loss(ly[test], yproba[test])
      acc = accuracy_score(ly[test], ypred[test])
      print "train set: %2d samples: %5d/%5d mll: %4.3f accuracy: %4.3f"%(i,lX[train,:].shape[0],lX[test,:].shape[0],mll[i],acc)
      
  print classification_report(ly, ypred, target_names=class_names)
  mll_oob = multiclass_log_loss(ly, yproba)
  
  print "oob multiclass logloss: %6.3f" %(mll_oob)
  print "avg multiclass logloss: %6.3f +/- %6.3f" %(mll.mean(),mll.std())
  #training on all data
  if trainFull:
    clf_orig.fit(lX, ly)
  return(clf_orig)


def shuffle(df, n=1, axis=0):     
        df = df.copy()
        for _ in range(n):
	    df.apply(np.random.shuffle, axis=axis)
            return df

    
def density(m):
    """
    For sparse & dense matrices
    """
    if isinstance(m,pd.DataFrame) or isinstance(m,np.ndarray):
      nz = np.count_nonzero(m.values)
      print "Non-zeros     : %12d"%(nz)
      te = m.shape[0]*m.shape[1]
      print "Total elements: %12d"%(te)
      print "Ratio         : %12.2f"%(float(nz)/float(te))
    else:
      entries=m.shape[0]*m.shape[1]
      return "Density      : %12.3f"%(m.nnz/float(entries))


def buildWeightedModel(lmodel,lXs,ly,lw=None,fitWithWeights=True,nfolds=8,useProba=True,scale_wt=None,n_jobs=1,verbose=False,local_scorer='roc_auc'):
    """   
    Build model using sample weights, can use weights for scoring function
    """ 
    
    fit_params = {}	
    fit_params['scoring_weight']=lw	
    fit_params['fitWithWeights']=fitWithWeights
    
    print "Xvalidation..."
    scores = cross_validation.cross_val_score(lmodel,lXs,ly,fit_params=fit_params, scoring=local_scorer,cv=nfolds,n_jobs=n_jobs)
    print "<SCORE>= %0.4f (+/- %0.4f)" % (scores.mean(), scores.std())
    print "Building model with all instances..."
    
    if fitWithWeights:
	    print "Use sample weights for final model..."
	    lmodel.fit(lXs,ly,sample_weight=lw)
    else:
	    lmodel.fit(lXs,ly)
    
    #analysis of final predictions
    if useProba:
	print "Using predic_proba for final model..."
	probs = lmodel.predict_proba(lXs)[:,1]
        #plot it
        plt.hist(probs,label='final model',bins=50,color='b')
        plt.legend()
        plt.draw()
    
    return(lmodel)
    

def group_sparse(Xold,Xold_test, degree=2,append=True):
    """ 
    multiply columns of sparse data
    """
    print "Columnwise min of data..."
    #only for important data
    (lXs,lXs_test) = linearFeatureSelection(model,Xold,Xold_test,10)
    #also transform old data
    #(Xold,Xold_test) = linearFeatureSelection(model,Xold,Xold_test,5000)
    Xtmp=sparse.vstack((lXs_test,lXs),format="csc")
    Xtmp=pd.DataFrame(np.asarray(Xtmp.todense()))
    new_data = None
    m,n = Xtmp.shape
    for indices in itertools.combinations(range(n), degree):
	indices=Xtmp.columns[list(indices)]
	print indices
	if not isinstance(new_data,pd.DataFrame):
	  new_data=pd.DataFrame(Xtmp[indices].apply(np.min, axis=1))
	else:
	  new_data = pd.concat([new_data, pd.DataFrame(Xtmp[indices].apply(np.min, axis=1))],axis=1)
	print new_data.shape
    
    
    #making test data
    Xreduced_test = new_data[:Xold_test.shape[0]]
    if append: 
	Xreduced_test=sparse.hstack((Xold_test,Xreduced_test),format="csr")
    print "New test data:",Xreduced_test.shape
    
    #making train data
    Xreduced = new_data[Xold_test.shape[0]:]
    if append:
	Xreduced=sparse.hstack((Xold,Xreduced),format="csr")
    print "New test data:",Xreduced.shape   
    return(Xreduced,Xreduced_test)
    
def rfFeatureImportance(forest,Xold,Xold_test,n):
    """ 
    Selects n best features from a model which has the attribute feature_importances_
    """
    print "Feature importance..."
    if not hasattr(forest,'feature_importances_'): 
      print "Missing attribute feature_importances_"
      return
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)#perhas we need it later
    
    indices = np.argsort(importances)[::-1]
    print indices
    # Print the feature ranking
    print("Feature ranking:")

    for i,f in enumerate(indices):
	print("%3d. feature %16s %3d - %6.4f +/- %6.4f" % (i, Xold.columns[f], f, importances[f] , std[f]))
	
    # Plot the feature importances of the forest  
    plt.bar(left=np.arange(len(indices)),height=importances[indices] , width=0.35, color='r',yerr=std[indices])
    plt.ylabel('Importance')
    plt.title("Feature importances")
    #stack train and test data
    Xreduced = pd.concat([Xold_test, Xold])
    #sorting features
    n=len(indices)-n
    print "Selection of ",n," top features..."
    Xreduced=Xreduced.iloc[:,indices[0:n]]
    print Xreduced.columns
    #split train and test data
    #pd slicing sometimes confusing...last element in slicing is inclusive!!! use iloc for integer indexing (i.e. in case index are float or not ordered)
    pdrowidx=Xold_test.shape[0]-1
    Xreduced_test = Xreduced[:len(Xold_test.index)]
    Xreduced = Xreduced[len(Xold_test.index):]
    #print "Xreduced_test:",Xreduced_test
    print "Xreduced_test:",Xreduced_test.shape
    print "Xreduced_train:",Xreduced.shape
    return(Xreduced,Xreduced_test)
    
def linearFeatureSelection(lmodel,Xold,Xold_test,n):
    """
    Analysis of data if coef_ are available for sparse matrices, better use t-scores
    """
    print "Selecting features based on important coefficients..."
    if hasattr(lmodel, 'coef_') and isinstance(Xold,sparse.csr.csr_matrix): 
	print("Dimensionality before: %d" % lmodel.coef_.shape[1])
	indices = np.argsort(lmodel.coef_)[0,-n:][::-1]
	#print model.coef_[top10b]
	#for i in xrange(indices.shape[0]):
	#    print("Top %2d: coef: %0.3f col: %2d" % (i+1,lmodel.coef_[0,indices[i]], indices[i]))
	plt.bar(left=np.arange(len(indices)),height=lmodel.coef_[0,indices] , width=0.35, color='r')
	plt.ylabel('Importance')
	#stack train and test data
	#Xreduced=np.vstack((Xold_test,Xold))
	Xreduced=sparse.vstack((Xold_test,Xold),format="csr")
	#sorting features
	#print indices[0:n]
	Xtmp=Xreduced[:,indices[0:n]] 
	print("Dimensionality after: %d" % Xtmp.shape[1])
	#split train and test data
	Xreduced_test = Xtmp[:Xold_test.shape[0]]
	Xreduced = Xtmp[Xold_test.shape[0]:]
	return(Xreduced,Xreduced_test)


# The step callback function, this function
# will be called every step (generation) of the GA evolution
def evolve_callback(ga_engine):
   generation = ga_engine.getCurrentGeneration()
   if generation % 1 == 0:
      pop = ga_engine.getPopulation()
      column_names = pop.oneSelfGenome.getParam("Xtrain").columns
      binary_list = [True if value==1 else False for value in ga_engine.bestIndividual()]
      print "Best individual:",column_names[binary_list]
      print "Best individual:",ga_engine.bestIndividual()
   return False


def eval_func(genome):
  """
  Evaluation function for GA
  """
  model = genome.getParam("model")
  Xtrain = genome.getParam("Xtrain")
  ytrain = genome.getParam("ytrain")
  cv = genome.getParam("cv")
  scoring_func = genome.getParam("scoring_func")
  n_jobs = genome.getParam("n_jobs")

  #print genome
  binary_list = [True if value==1 else False for value in genome]
  Xact = Xtrain.iloc[:,binary_list]
  #print "New shape:",Xact.shape
  #print "Columns:",Xact.columns
  t0 = time()
  score = buildModel(model,Xact,ytrain,cv=cv,scoring=scoring_func,n_jobs=n_jobs,trainFull=False)
  run_time = time() - t0
  
  #print "cv-score: %6.3f +/- %6.3f genome: %s" %(-1*score.mean(),score.std(),[value for value in genome])
  print "cv-score: %6.3f +/- %6.3f cv-runs: %4d time: %4.2f" %(-1*score.mean(),score.std(),len(score),run_time)
  return -1/score.mean()


def genetic_feature_selection(model,Xtrain,ytrain,Xtest,pool_features=None,start_features=None,scoring_func='mean_squared_error',cv=None,n_iter=3,n_pop=20,n_jobs=1):
    """
    Genetic feature selection
    """
    from pyevolve import G1DBinaryString
    from pyevolve import Initializators, Mutators
    from pyevolve import GSimpleGA
    from pyevolve import Selectors
    from pyevolve import Statistics
    from pyevolve import DBAdapters
    from pyevolve.GenomeBase import GenomeBase
    
    print cv
    print model
    
    if pool_features==None:
	pool_features = Xtrain.columns
    
    genome = G1DBinaryString.G1DBinaryString(Xtrain.shape[1])

    genome.evaluator.set(eval_func)
    #genome.mutator.set(Mutators.G1DBinaryStringMutatorFlip)
    
    #start_features='11111111111111111111111111101111101111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111101111111111111111111111111111110111111111111111111111111111111111101111111111111111110111111111111111111011111111111111111011110111111111111111101111011111111111111011111111111111111111111111011111111111100111111111111101111111101111111111111011111111111111111111111111111111111111111011111111111101111111111111111111111111111111111111111111111111101111111111101111111111101111111111111111011111111111111111111111111111111111111111111111111111011111111111111111111111111110111111110111111111111111111111111111111111111111111111111111111111111111011111111111111111111111101111111111011110111111111011110111001111101101101001100111011110010000101000110011111111010000001110110100111101000000001111111110010010000111100100010100011011001001011100101111010000001110001001011110011101101100001111101110010001000010100111001010011011110101100100011111101001111001000010000100100111111111100111110000101001000000101001100111011110010000100101001110010101100010100101101101001000110100111001000010111101111000110001110110101101111010110010101010011010100110100111110001110101011100010011100100000011000100101010111110001010001010010111000000110101011000100110000000100110000000010101101111001001100111011011110110101111111011011100111110010110100101110'
    #if isinstance(start_features,str):
	#binary_list = [True if value=='1' else False for value in start_features]
	#start_features = list(Xtrain.columns[binary_list])
    
    genome.setParams(model=model,Xtrain=Xtrain,ytrain=ytrain,start_features=start_features,pool_features=pool_features,scoring_func=scoring_func,cv=cv,n_jobs=n_jobs)
    genome.initializator.set(Initializators.G1DBinaryStringInitializator)
    # Genetic Algorithm Instance
    ga = GSimpleGA.GSimpleGA(genome)
    ga.stepCallback.set(evolve_callback)
    #GSimpleGA.GSimpleGA.setMultiProcessing(ga)
    ga.selector.set(Selectors.GTournamentSelector)
    ga.setGenerations(n_iter)
    ga.setPopulationSize(n_pop)
    # Do the evolution, with stats dump
    # frequency of 10 generations
    ga.evolve(freq_stats=1)
    
    # Best individual
    #print ga.bestIndividual()
    binary_list = [True if value==1 else False for value in ga.bestIndividual()]
    print "Best features:"
    print Xtrain.columns[binary_list]
    print "Selected %d out of %d features"%(sum(binary_list),Xtrain.shape[1])
    buildModel(model,Xtrain.iloc[:,binary_list],ytrain,cv=cv,scoring=scoring_func,n_jobs=n_jobs,trainFull=False,verbose=True)
    


def greedyFeatureSelection(lmodel,lX,ly,itermax=10,itermin=5,pool_features=None ,start_features=None,verbose=False, cv=5, n_jobs=4,scoring_func='mean_squared_error'):
	features=[]
	
	if pool_features is None:
	    pool_features = lX.columns
	
	if start_features is not None:
	    features=start_features
	else:
	    features=[col for col in lX.columns if not col in pool_features]
	
	scores = []
	score_opt=1E10
	for i in xrange(itermax):
	    print "Round %4d"%(i)
	    score_best=1E9
	    for a,k in enumerate(xrange(len(pool_features))):
		act_feature = pool_features[k]
		if act_feature in set(features):continue
		
		features.append(act_feature)
		
		t0 = time()
		score = cross_validation.cross_val_score(lmodel,lX.loc[:,features],ly,fit_params=None, scoring=scoring_func,cv=cv,n_jobs=n_jobs)
		run_time = time() - t0
		
		if 'mean_squared_error' in str(scoring_func):
		  score = -1*(score)**0.5
		else:
		  score = -1*score
		
		
		if verbose:
		    print "(%4d/%4d) TARGET: %-12s - <score>= %0.4f (+/- %0.5f) score,iteration best= %0.4f score,overall best: %0.4f features: %5d time: %6.2f" % (a+1,len(pool_features),act_feature, score.mean(), score.std(),score_best,score_opt,lX.loc[:,features].shape[1],run_time)
		
		if score.mean()<score_best:
		  score_best=score.mean()
		  new_feat=act_feature
		  features_best=features[:]
		del features[-1]
	    features.append(new_feat)
	    scores.append(score_best)
	    
	    if (i>itermin and (score_opt>score_best)):
		    print "Converged with threshold: %0.6f"%(np.abs(score_opt-score_best))
		    score_opt=score_best
		    opt_list=features_best
		    break	    
	    if score_best<score_opt:
		    score_opt=score_best
		    opt_list=features_best
		
	    print " nr features: %5d "%(len(features)),
	    print " score,iteration best= %0.4f score,overall best: %0.4f \n%r" % (score_best,score_opt,features)
	 
	print "Scores:",scores
	
	print "Best score: %6.4f with %5d features:\n%r"%(score_opt,len(opt_list),opt_list)
	plt.plot(scores)
	plt.show()

def iterativeFeatureSelection(lmodel,Xold,Xold_test,ly,iterations=5,nrfeats=1,scoring=None,cv=None,n_jobs=8):
	"""
	Iterative feature selection e.g. via random Forest
	"""
	for i in xrange(iterations):
	    print ">>>Iteration: ",i,"<<<"
	    #lmodel = buildModel(lmodel,Xold,ly,cv=cv,scoring=scoring,n_jobs=n_jobs,trainFull=True)
	    lmodel.fit(Xold,ly)
	    (Xold,Xold_test)=rfFeatureImportance(lmodel,Xold,Xold_test,nrfeats)
	    #Xold.to_csv("../stumbled_upon/data/Xlarge_"+str(i)+".csv")
	    #Xold_test.to_csv("../stumbled_upon/data/XXlarge_test_"+str(i)+".csv")
	return(Xold,Xold_test)
	
def removeInstances(lXs,ly,preds,t,returnSD=True):
	"""
	Removes examples from train set either due to prediction error or due to standard deviation
	Preds should come from repeated CV.
	"""
	if returnSD:
	    res=preds
	else:
	    res=np.abs(ly-preds)
	d={'abs_err' : pd.Series(res)}
	res=pd.DataFrame(d)
	res.index=lXs.index
	lXs_reduced=pd.concat([lXs,res], axis=1)
	boolindex=lXs_reduced['abs_err']<t
	lXs_reduced=lXs_reduced[boolindex]
	#print lXs_reduced.shape
	#ninst[i]=len(Xtrain.index)-len(lXs_reduced.index)
	lXs_reduced = lXs_reduced.drop(['abs_err'], axis=1)
	#print "New dim:",lXs_reduced.shape
	ly_reduced=ly[np.asarray(boolindex)]
	return (lXs_reduced,ly_reduced)

def removeLowVariance(X_all,threshhold=1E-5):
	"""
	remove useless data
	"""
	
	if isinstance(X_all,sparse.csc_matrix):
	    print "Making matrix dense again..."
	    X_all = pd.DataFrame(X_all.toarray())
	
	idx = np.asarray(X_all.std()<=threshhold)
	if len(X_all.columns[idx])>0:
		print "Dropped %4d zero variance columns (threshold=%6.3f): %r"%(np.sum(idx),threshhold,X_all.columns[idx])
		X_all.drop(X_all.columns[idx], axis=1,inplace=True)
	else:
		print "Variance filter dropped nothing (threshhold = %6.3f)."%(threshhold)
	
	return(X_all)
	

def pcAnalysis(X,Xtest,y=None,w=None,ncomp=2,transform=False,classification=False):
    """
    PCA 
    """  
    pca = PCA(n_components=ncomp)
    if transform:
        print "PC reduction"
        X_all = pd.concat([Xtest, X])
        
        X_r = pca.fit_transform(np.asarray(X_all)) 
        print(pca.explained_variance_ratio_)
        #split
        X_r_train = X_r[len(Xtest.index):]
        X_r_test = X_r[:len(Xtest.index)]
        return (X_r_train,X_r_test)
    
    elif classification:
        print "PC analysis for classification"
        X_all = pd.concat([Xtest, X])
        #this is transformation is necessary otherwise PCA gives rubbish!!
        ytrain = np.asarray(y)
        
        X_r = pca.fit_transform(np.asarray(X_all))  
        
        if w is None:
            plt.scatter(X_r[ytrain == 0,0], X_r[ytrain == 0,1], c='r', label="1",alpha=0.1)
            plt.scatter(X_r[ytrain == 1,0], X_r[ytrain == 1,1], c='g',label="0",alpha=0.1)
        else:
            plt.scatter(X_r[ytrain == 0,0], X_r[ytrain == 0,1], c='r', label="background",s=w[ytrain==0]*25.0,alpha=0.1)
            plt.scatter(X_r[ytrain == 1,0], X_r[ytrain == 1,1], c='g',label="signal",s=w[ytrain==1]*1000.0,alpha=0.1)

        print(pca.explained_variance_ratio_) 
        plt.legend()
        #plt.xlim(-3500,2000)
        #plt.ylim(-1000,2000)
        plt.draw()
    else:
	print "PC analysis for train/test"
	X_all = pd.concat([Xtest, X])
	X_r = pca.fit_transform(np.asarray(X_all))
	plt.scatter(X_r[len(Xtest.index):,0], X_r[len(Xtest.index):,1], c='r', label="train",alpha=0.5)
	plt.scatter(X_r[:len(Xtest.index),0], X_r[:len(Xtest.index),1], c='g',label="test",alpha=0.5)
	print("Explained variance:",pca.explained_variance_ratio_) 
	plt.legend()
	plt.show()

	
def root_mean_squared_error(x,y):
	return mean_squared_error(x,y)**0.5
	
def mean_absolute_error(x,y):
	x = x.flatten()
	y = y.flatten()
	return np.mean(np.abs(x-y))


def multiclass_log_loss(y_true, y_pred, eps=1e-15):
    return log_loss(y_true,y_pred, eps=eps, normalize=True)

def getOOBCVPredictions(lmodel,lXs,ly,folds=8,repeats=1,returnSD=True,score_func='rmse'):
	"""
	Get cv oob predictions for classifiers
	"""
	funcdict={}
	if score_func=='rmse':
	    funcdict['scorer_funct']=root_mean_squared_error
	else:
	    funcdict['scorer_funct']=roc_auc_score
	
	
	print "Computing oob predictions..."
	if isinstance(lmodel,RandomForestClassifier) or isinstance(lmodel,SGDClassifier):
		lmodel.set_params(n_jobs=4)
	oobpreds=np.zeros((lXs.shape[0],repeats))
	for j in xrange(repeats):
	    #print lmodel.get_params()
	    cv = KFold(lXs.shape[0], n_folds=folds,random_state=j,shuffle=True)
	    scores=np.zeros(folds)	
	    for i, (train, test) in enumerate(cv):
		Xtrain = lXs.iloc[train]
		Xtest = lXs.iloc[test]
		#print Xtest['avglinksize'].head(3)
		lmodel.fit(Xtrain, ly[train])
		if score_func=='rmse':
		    oobpreds[test,j] = lmodel.predict(Xtest)
		    scores[i]=funcdict['scorer_funct'](ly[test],oobpreds[test,j])
		else:  
		    oobpreds[test,j] = lmodel.predict_proba(Xtest)[:,1]
		    scores[i]=funcdict['scorer_funct'](ly[test],oobpreds[test,j])
		#print "AUC: %0.2f " % (scores[i])
		#save oobpredictions
	    print "Iteration:",j,
	    print " <score>: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std()),
	    
	    oobscore=funcdict['scorer_funct'](ly,oobpreds[:,j])
	    print " score,oob: %0.3f" %(oobscore)
	    
	scores=[funcdict['scorer_funct'](ly,oobpreds[:,j]) for j in xrange(repeats)]
	#simple averaging of blending
	oob_avg=np.mean(oobpreds,axis=1)
	print "Summary: <score,oob>: %0.3f (%d repeats)" %(funcdict['scorer_funct'](ly,oob_avg),repeats,)	
	if returnSD:
	    oob_avg=np.std(oobpreds,axis=1)
	return(oob_avg)
	
def lofFilter(pred,threshhold=10.0,training=True):
	"""
	filter data according to local outlier frequency as computed by R...did not work
	"""
	indices=[]
	global test_indices
	lof = pd.read_csv("../stumbled_upon/data/lof.csv", sep=",", index_col=0)
	lof = lof[len(test_indices):]
	avg=np.mean(pred)
	for i in xrange(len(lof.index)):
	    #print lof.iloc[i,0]
	    if lof.iloc[i,0]>threshhold:
		pred[i]=avg
		indices.append(i)
	#print indices
	print "threshhold:,",threshhold,"n,changed:",len(indices)," mean:",avg
	return pred
      
def filterClassNoise(lmodel,lXs,lXs_test,ly):
	"""
	Removes training samples which could be class noise
	Done in outer XVal loop
	precision: Wieviel falsche habe ich erwischt
	recall: wieviele richtige sind durch die Lappen gegangen
	"""
	threshhold=[0.045,0.04,0.035]
	folds=10
	print "Filter strongly misclassified classes..."
	#rdidx=random.sample(xrange(1000), 20)
	#print rdidx
	#lXs = lXs.iloc[rdidx]
	#ly = ly[rdidx]
	preds=getOOBCVPredictions(lmodel,Xs,Xs_test,y,folds,10)
	print preds
	plt.hist(preds,bins=20)
	plt.show()
	#print "stdev:",std
	#should be oob or cvalidated!!!!
	#preds = lmodel.predict_proba(lXs)[:,1]
	scores=np.zeros((folds,len(threshhold)))
	oobpreds=np.zeros((lXs.shape[0],folds))
	for j,t in enumerate(threshhold):
	    #XValidation
	    cv = KFold(lXs.shape[0], n_folds=folds, indices=True,random_state=j,shuffle=True)	    	    
	    ninst=np.zeros(folds)	    
	    for i, (train, test) in enumerate(cv):
		Xtrain = lXs.iloc[train]
		ytrain=  ly[train]		
		#now remove examples from train
		lXs_reduced,ly_reduced = removeInstances(Xtrain,ytrain,preds[train],t)
		ninst[i]=len(Xtrain.index)-len(lXs_reduced.index)
		lmodel.fit(lXs_reduced, ly_reduced)
		
		#testing data, not manipulated
		Xtest = lXs.iloc[test]
		oobpreds[test,j] = lmodel.predict_proba(Xtest)[:,1]
		
		scores[i,j]=roc_auc_score(ly[test],oobpreds[test,j])

	    print "Threshhold: %0.3f  <AUC>: %0.3f (+/- %0.3f) removed instances: %4.2f" % (t, scores[:,j].mean(), scores[:,j].std(), ninst.mean() ),
	    print " AUC oob: %0.3f" %(roc_auc_score(ly,oobpreds[:,j]))
	scores=np.mean(scores,axis=0)
	print scores
	plt.plot(threshhold,scores,'ro')
	top = np.argsort(scores)
	optt = threshhold[top[-1]]
	print "Optimum threshhold %4.2f index: %d with score: %4.4f" %(optt,top[-1],scores[top[-1]])
	lXs_reduced,ly_reduced = removeInstances(lXs,ly,preds,optt)	
	return(lXs_reduced,lXs_test,ly_reduced)

	
def showMisclass(lmodel,lXs,ly,t=0.0,bubblesizes=None):
    """
    Show bubble plot of strongest misclassifications...
    """
    folds=4
    repeats=1
    print "Show strongly misclassified classes..."
    preds=getOOBCVPredictions(lmodel,lXs,ly,folds,repeats,returnSD=False)

    abs_err=pd.DataFrame({'abs_err' : pd.Series(np.abs(ly-preds))})
    residue=pd.DataFrame({'residue' : pd.Series((ly-preds))})
    ly=pd.DataFrame({'y' : pd.Series(ly)})
    preds=pd.DataFrame({'preds' : pd.Series(preds)})
    
    lXs_plot=pd.concat([ly,preds,residue,abs_err], axis=1)
    lXs_plot.index=lXs.index
    lXs_plot=pd.concat([lXs_plot,lXs], axis=1)
    lXs_plot.sort(columns='abs_err',inplace=True)
    
    boolindex=lXs_plot['abs_err']>t
    
    lXs_plot=lXs_plot[boolindex]
    print "Number of instances left:",lXs_plot.shape[0]
    col1='preds'
    col2='residue'
    #bubblesizes=lXs_plot['y']*50
    bubblesizes=30
    
    #sct = plt.scatter(lXs_plot[col1], lXs_plot[col2],c=lXs_plot['abs_err'],s=bubblesizes, linewidths=2, edgecolor='black')
    sct = plt.scatter(lXs_plot[col1], lXs_plot[col2],s=bubblesizes, linewidths=2, edgecolor='black')
    sct.set_alpha(0.75)
    
    print "%4s %6s %6s %8s"%("index",'y','preds','residue')
    for row_index, row in lXs_plot.iterrows():
	plt.text(row[col1], row[col2],row_index,size=10,horizontalalignment='center')
	print "%4d %6.3f %6.3f %8.3f"%(row_index,row['y'],row['preds'],row['residue'])
    print "%4s %6s %6s %8s"%("index",'y','preds','residue')
    
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title("error")
    plt.show()
    
    

    
def scaleData(lXs,lXs_test=None,cols=None,normalize=False,epsilon=0.0):
    """
    standard scaling of data, also possible with sklearn StandardScaler but not with dataframe
    """
    if cols is None:
	cols = lXs.columns

    if lXs_test is not None:
	lX_all = pd.concat([lXs_test, lXs])
    else:
	lX_all = lXs

    if normalize:
	print "Normalize data..."
	lX_all[cols] = (lX_all[cols] - lX_all[cols].min()) / (lX_all[cols].max() - lX_all[cols].min())
    #standardize
    else:
	if epsilon>1E-15:
	  print "Standardize data with epsilon:",epsilon
	  lX_all[cols] = (lX_all[cols] - lX_all[cols].mean()) / np.sqrt(lX_all[cols].var()+epsilon)
	else:
	  print "Standardize data"
	  lX_all[cols] = (lX_all[cols] - lX_all[cols].mean()) / lX_all[cols].std()
    
    #print lX_all[cols].describe()
    
    if lXs_test is not None:
	lXs = lX_all[len(lXs_test.index):]
	lXs_test = lX_all[:len(lXs_test.index)]
	return (lXs,lXs_test)
    else:
	return lX_all


def data_binning(X,binning):
    """
    Bin dat in n=binning bins
    """
    for col in X.columns:
	    #print Xall[col]
	    tmp = pd.cut(X[col].values, binning,labels=False)
	    X[col] = np.asarray(tmp)
	    #print Xall[col]
	    #raw_input()
	    #groups = Xall.groupby(tmp)
	    #print groups
	    #print groups.describe()
	    
    return X

def binarizeProbs(a,cutoff):   
    """
    turn probabilities to 1 and 0
    """
    if a>cutoff: return 1.0
    else: return 0.0
    
def make_polynomials(Xtrain,Xtest=None,degree=2,cutoff=100):
    """
    Generate polynomial features
    """
    if Xtest is not None: Xtrain = Xtrain[len(Xtest.index):]
    m,n = Xtrain.shape
    indices = list(itertools.combinations(range(n), degree))
    new_data=[]
    colnames=[]
    for i,j in indices:
      ##Xnew = (Xtrain.values[:, np.newaxis, i] * Xtrain.values[:, j, np.newaxis]).reshape(len(Xtrain), -1)
      name = str(Xtrain.columns[i])+"x"+str(Xtrain.columns[j])
      Xnew = (Xtrain.values[:, i] * Xtrain.values[:, j])
      #Xnew = (Xtrain.iloc[:, i] * Xtrain.iloc[:, j])
      n_nonnull = (Xnew != 0).astype(int).sum()
      if n_nonnull>cutoff:
	new_data.append(Xnew)
	colnames.append(name)
      else:
	print "Dropped:",name
      
    new_data = pd.DataFrame(np.array(new_data).T,columns=colnames)
    
    return new_data


# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

def makeGridSearch(lmodel,lX,ly,n_jobs=1,refit=True,cv=5,scoring='roc_auc',random_iter=-1,parameters={}):
    print "Start GridSearch..."
    if parameters is None:
      #parameters = {'filter__param': [30,50],'model__C':[1]}
      #parameters = {'pca__n_components':[30],'model__C':[1]}
      #parameters = {'model__C':[1,0.1,0.01,0.001]}#Linear SVC+LOGREG
      #parameters = {'C':[10]}#Linear SVC+LOGREG
      #parameters = {'C':[1.0],'penalty':['l2']}#Linear SVC+LOGREG
      #parameters = {'n_estimators':[50],'alpha_L1':[1E-1],'lambda_L2':[1E-1]}#XGBOOST GBLINEAR
      #parameters = {'alpha':[1E-6,1E-8],'n_iter':[100],'penalty':['l2']}#SGD
      #parameters = {'alpha':[1,1E-2,1E-4],'n_iter':[250],'penalty':['l2']}#SGD
      #parameters = {'learn_rates':[0.3,0.2],'learn_rate_decays':[1.0,0.9],'epochs':[40]}#DBN
      #parameters = {'hidden1_num_units': [600],'dropout1_p':[0.0,0.1,0.2,0.3,0.4,0.5],'maxout1_ds':[2,3,4],'hidden2_num_units': [600],'dropout2_p':[0.0],'maxout2_ds':[2,3,4],'hidden3_num_units': [600],'dropout3_p':[0.0],'maxout3_ds':[2,3,4],'max_epochs':[50,100,150],'update_learning_rate':[0.001,0.002,0.004]}#Lasagne
      #parameters = {'hidden1_num_units': [300],'dropout1_p':[0.25,0.5],'hidden2_num_units': [300],'dropout2_p':[0.5,0.25],'update_learning_rate':[0.01,0.008],'objective_alpha':[1E-10]}#Lasagne
      #parameters = {'hidden1_num_units': [500],'dropout1_p':[0.5],'hidden2_num_units': [500],'dropout2_p':[0.5,0.0],'hidden3_num_units': [500],'dropout3_p':[0.5,0.0],'max_epochs':[150,100],'objective_alpha':[1E-3,1E-6,1E-9],'update_learning_rate':[0.004,0.003,0.002]}#Lasagne
      #parameters = {'hidden1_num_units': [500,1000],'update_learning_rate':[0.0001,0.0005],'max_epochs':[500],'dropout1_p':[0.0,.2]}#Lasagne
      #parameters = {'hidden1_num_units': [500,500],'update_learning_rate':[0.0001,0.0005],'max_epochs':[500],'dropout1_p':[0.0,.2]}#Lasagne
      #parameters = {'hidden1_num_units': [200,300],'max_epochs':[1000],'dropout1_p':[0.1,0.2,0.3]}#Lasagne
      #parameters = {'n_estimators':[500], 'max_features':[18,20,22],'max_depth':[None],'max_leaf_nodes':[None],'min_samples_leaf':[1],'min_samples_split':[2],'criterion':['gini']}#xrf+xrf
      #parameters = {'class_weight': [{0: 1.,1: 1., 2: 1.,3: 1.,4: 1.,5: 1.,6: 1.,7: 1.,8: 1.,9: 1.},{0: 2.,1: 1., 2: 2.,3: 1.,4: 1.,5: 1.,6: 1.,7: 1.,8: 1.,9: 1.}]}
      #parameters = {'n_estimators':[300,400],'max_depth':[8,9,10],'learning_rate':[0.015,0.02,0.025,0.03],'subsample':[0.5,1.0]}#XGB+GBC
      #parameters = {'n_estimators':[400],'max_depth':[10],'learning_rate':[0.1,0.05,0.01],'subsample':[0.5]}#XGB+GBC
      parameters = {'n_estimators':[200,400],'max_depth':[6,8],'learning_rate':[0.05,0.03],'subsample':[0.5]}#XGB

    
    if random_iter<0:
	search  = grid_search.GridSearchCV(lmodel, parameters,n_jobs=n_jobs,verbose=2,scoring=scoring,cv=cv,refit=refit)
    else:
	search  = grid_search.RandomizedSearchCV(lmodel, param_distributions=parameters,n_jobs=n_jobs,verbose=2, scoring=scoring,cv=cv,refit=refit,n_iter=random_iter)
    
    search.fit(lX,ly)
    best_score=1.0E5
    print("%6s %6s %6s %r" % ("OOB", "MEAN", "SDEV", "PARAMS"))
    for params, mean_score, cvscores in search.grid_scores_:
	oob_score = -1* mean_score
	cvscores = -1 * cvscores
	mean_score = cvscores.mean()
	print("%6.3f %6.3f %6.3f %r" % (oob_score, mean_score, cvscores.std(), params))
	#if mean_score < best_score:
	#    best_score = mean_score
	#    scores[i,:] = cvscores
    
    #report(search.grid_scores_)

    if refit:
      return search.best_estimator_
    else:
      return None


def df_info(X):
    if isinstance(X,pd.DataFrame): X = X.values 
    print "Shape:",X.shape, " size (MB):",float(X.nbytes)/1.0E6, " dtype:",X.dtype


def analyzeLearningCurve(model,X,y,cv=8,score_func='roc_auc_score'):
    """
    make a learning curve according to http://scikit-learn.org/dev/auto_examples/plot_learning_curve.html
    """

    #cv = cross_validation.ShuffleSplit(X.shape[0], n_iter=10, test_size=0.2, random_state=0)
    #cv = KFold(X.shape[0], n_folds=folds,shuffle=True)  
    plot_learning_curve(model, "learning curve", X, y, ylim=(0.1, 1.01), cv=cv, n_jobs=1,scoring='accuracy')
    
 


 
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, scoring=f1_score, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,scoring=scoring, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
    
    
#some global vars
funcdict={}
funcdict['rmse']=root_mean_squared_error
funcdict['auc']=roc_auc_score
funcdict['mae']=mean_absolute_error
funcdict['msq']=mean_squared_error
funcdict['log_loss']=multiclass_log_loss
