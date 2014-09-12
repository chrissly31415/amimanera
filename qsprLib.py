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

from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.feature_extraction.text import CountVectorizer,HashingVectorizer,TfidfVectorizer
#from sklearn import metrics
from sklearn import cross_validation,grid_search
from sklearn.cross_validation import StratifiedKFold,KFold,StratifiedShuffleSplit,ShuffleSplit
from sklearn.metrics import roc_auc_score,classification_report,make_scorer,f1_score,precision_score,mean_squared_error
#from sklearn.utils.extmath import density
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import TruncatedSVD,PCA
from sklearn.pipeline import Pipeline

from sklearn.feature_selection import SelectKBest,SelectPercentile, chi2, f_classif,f_regression
from sklearn.naive_bayes import BernoulliNB,MultinomialNB,GaussianNB
from sklearn.cluster import k_means
from sklearn.isotonic import IsotonicRegression

from sklearn.linear_model import LogisticRegression,RandomizedLogisticRegression,SGDClassifier,Perceptron,SGDRegressor,RidgeClassifier,LinearRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,ExtraTreesClassifier,AdaBoostClassifier,ExtraTreesRegressor,GradientBoostingRegressor,BaggingClassifier,RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.learning_curve import learning_curve

def one_hot_encoder(data, col, replace=False):
    """ Takes a dataframe and a list of columns that need to be encoded.
    Returns a 3-tuple comprising the data, the vectorized data,{
    and the fitted vectorizor.
    credits to https://gist.github.com/kljensen/5452382
    """
    vec=DictVectorizer()
    tmp=[]
    print data.columns
    for row in data.loc[:,[col]].itertuples():
	d=dict({'cat': row[1]})
	tmp.append(d)  
    tmp=vec.fit_transform(tmp).toarray()
    vecData = pd.DataFrame(tmp)
    vecData.columns = vec.get_feature_names()
    print "New features:",vecData.columns
    print vecData.describe()
    vecData.index = data.index
    if replace is True:
	data = data.drop(col, axis=1)
	data = data.join(vecData)
    return data

    
def removeCorrelations(X_all,threshhold):
    #filter correlated data
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
	      
	      
def modelEvaluation(lmodel,lXs,ly):
    """
    MODEL EVALUATION
    """
    ly = np.asarray(ly)
    print "Model evaluation..."
    folds=8
    #parameters=np.logspace(-14, -7, num=8, base=2.0)#SDG
    #parameters=np.logspace(-7, 0, num=8, base=2.0)#LG
    #parameters=[250,500,1000,2000]#rf
    parameters=[100.0,10.0,8.0,5.0,2.0,1.5,1.2]#chi2
    #parameters=[2,3,4,5]#gbm
    #parameters=np.logspace(-7, -0, num=8, base=2.0)
    print "Parameter space:",parameters
    #feature selection within xvalidation
    oobpreds=np.zeros((lXs.shape[0],len(parameters)))
    for j,p in enumerate(parameters):
	#if isinstance(lmodel,SGDClassifier):
	#    lmodel.set_params(alpha=p)
	#if (isinstance(lmodel,LogisticRegression) or isinstance(lmodel,SVC)) and p<1000:
	#    lmodel.set_params(C=p)
	#if isinstance(lmodel,RandomForestClassifier) :
	#    lmodel.set_params(max_features=p)
	#if isinstance(lmodel,GradientBoostingClassifier):
	#    lmodel.set_params(max_depth=p)
        #print lmodel.get_params()
        cv = KFold(lXs.shape[0], n_folds=folds,indices=True, random_state=j)
	scores=np.zeros(folds)	
	for i, (train, test) in enumerate(cv):
	    #print("Extracting %s best features by a chi-squared test" % p)
	    #ch2 = SelectKBest(chi2, k=p)
	    #ch2 = SelectPercentile(chi2,percentile=p)
	    #Xtrain = ch2.fit_transform(lXs[train], ly[train])
	    #Xtest = ch2.transform(lXs[test]) 
	    Xtrain = lXs.iloc[train]
	    Xtest = lXs.iloc[test]
	    lmodel.fit(Xtrain, ly[train])
	    oobpreds[test,j] = lmodel.predict_proba(Xtest)[:,1]
	    scores[i]=roc_auc_score(ly[test],oobpreds[test,j])
	    #print "AUC: %0.2f " % (scores[i])
	    #save oobpredictions
	print "Iteration:",j," parameter:",p,
	print " <AUC>: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std()),
	print " AUC oob: %0.3f" %(roc_auc_score(ly,oobpreds[:,j]))
	#Filter data
	#predmod=lofFilter(oobpreds[:,j],p)
	#print " AUC calibrated: %0.3f" %(roc_auc_score(ly,predmod))
	
    scores=[roc_auc_score(ly,oobpreds[:,j]) for j in xrange(len(parameters))]
    plt.plot(parameters,scores,'ro')
    
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
    
def pyGridSearch(lmodel,lXs,ly):  
    """   
    Grid search with sklearn internal tool
    """ 
    print "Grid search..."
    #parameters = {'C':[1000,10000,100], 'gamma':[0.001,0.0001]}
    #parameters = {'max_depth':[3,6,9], 'learning_rate':[0.5,0.1,0.01,0.05],'n_estimators':[150,300,500]}#gbm
    #parameters = {'max_depth':[2], 'learning_rate':[0.01,0.001],'n_estimators':[3000]}#gbm
    #parameters = {'n_estimators':[500], 'max_features':[5,10,15]}#rf
    #parameters = {'n_estimators':[2000,3000], 'learning_rate':[0.08,0.04,0.02]}#adaboost
    #parameters = {'n_estimators':[400,600,800], 'max_features':[10,15,20],'min_samples_leaf':[10,20]}#rf
    parameters = {'alpha':[0.0001,0.001,0.01,0.1],'n_iter':[5,10,100,150],'penalty':['l1','l2']}#SGD
    #parameters = {'C':[0.1,1,10]}#SVC
    #parameters = {'filter__percentile': [100,80,50,25] , 'model__alpha':[1.0,0.8,0.5,0.1]}#opt nb
    #parameters = {'filter__percentile': [16,15,14,13,12] , 'model__n_neighbors':[125,130,135,150,200]}#knn
    #parameters = {'n_neighbors':[1,2,3,5,8,10]}#knn
    #parameters = {'filter__percentile': [6,5,4,3,2,1], 'model__n_estimators': [500], 'model__max_features':['auto'], 'model__min_samples_leaf':[10] }#rf
    #parameters = {'filter__percentile': [100,95,80,70,60,50,25], 'model__C': [0.5,1.0, 10.0], 'model__intercept_scaling': [0.1,1.0,10,100,1000] }#pipeline
    #parameters = {'filter__percentile': [100,98,95,80,70,60,50,25], 'model__C': [0.5,1.0, 10.0,0.1],'model__penalty': ['l1','l2'] }#pipeline
    #parameters = {'filter__percentile': [90,80], 'model__n_estimators': [600,500],'model__learning_rate': [0.1] }#pipeline
    
    clf_opt = grid_search.GridSearchCV(lmodel, parameters,cv=8,scoring='roc_auc',n_jobs=4,verbose=1)
    clf_opt.fit(lXs,ly)
    
    for params, mean_score, scores in clf_opt.grid_scores_:
        print("%0.3f (+/- %0.3f) for %r"
              % (mean_score.mean(), scores.std(), params))
    return(clf_opt.best_estimator_)
    
    
def buildModel(lmodel,lXs,ly,sweights=None,feature_names=None):
    """   
    Final model building part
    """ 
    print "Xvalidation..."
    scores = cross_validation.cross_val_score(lmodel, lXs, ly, cv=5, scoring='roc_auc',n_jobs=1)
    print "AUC: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std())
    print "Building model with all instances..."
    if isinstance(lmodel,RandomForestClassifier) or isinstance(lmodel,SGDClassifier):
	    lmodel.set_params(n_jobs=4)
	    lmodel.fit(lXs,ly,sample_weight=sweights)
    else:
	    lmodel.fit(lXs,ly)
    
    #analyzeModel(lmodel,feature_names)
    return(lmodel)
    
def density(m):
    """
    For sparse matrices
    """
    entries=m.shape[0]*m.shape[1]
    return m.nnz/float(entries)
  
  
def group_sparse_old(Xold,Xold_test, degree=2,append=True):
    """ 
    multiply columns of sparse data
    """
    print "Grouping sparse data..."
    #only for important data
    (lXs,lXs_test) = linearFeatureSelection(model,Xold,Xold_test,200)
    #also transform old data
    #(Xold,Xold_test) = linearFeatureSelection(model,Xold,Xold_test,5000)
    
    Xtmp=sparse.vstack((lXs_test,lXs),format="csr")
    #turn into pandas dataframe for grouping
    new_data=None
    m,n = Xtmp.shape
    for indices in itertools.combinations(range(n), degree):
        #print "idx:",indices
	col1,col2 =indices
	out1 = Xtmp.tocsc()[:,col1]
	out1 = out1.transpose(copy=False)
	out2 = Xtmp.tocsc()[:,col2]
	tmp = np.ravel(np.asarray(out2.todense()))
	diag2 = sparse.spdiags(tmp,[0],out2.shape[0],out2.shape[0],format="csc")
	#out1+diag2-max(out1,diag2)
	prod = out1*diag2
	prod = prod.transpose()
	dens=density(prod)
	#print " Non-zeros: %4.3f " %(dens)
	if new_data is None:  
	    new_data=sparse.csc_matrix(prod)
	elif dens>0.0:
	    new_data=sparse.hstack((new_data,prod),format="csr")
	
    print "Shape of interactions matrix:",new_data.shape,
    print " Non-zeros: %4.3f " %(density(new_data))

    #makting test data
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
    #std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)#perhas we need it later
    indices = np.argsort(importances)[::-1]
    # Print the feature ranking
    print("Feature ranking:")

    for f in range(len(indices)):
	print("%d. feature %64s %d (%f)" % (f + 1, Xold.columns[indices[f]], indices[f], importances[indices[f]]))
	
    # Plot the feature importances of the forest  
    plt.bar(left=np.arange(len(indices)),height=importances[indices] , width=0.35, color='r')
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
    Analysis of data if coef_ are available for sparse matrices
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
	
def iterativeFeatureSelection(lmodel,Xold,Xold_test,ly,iterations,nrfeats):
	"""
	Iterative feature selection e.g. via random Forest
	"""
	for i in xrange(iterations):
	    print ">>>Iteration: ",i,"<<<"
	    lmodel = buildModel(lmodel,Xold,ly)
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

def removeZeroVariance(X,Xtest):
	"""
	remove useless data
	"""
	X_all = pd.concat([Xtest, X])
	idx = np.asarray(X_all.std()==0)
	print "Dropped zero variance columns:",X_all.columns[idx]
	X_all.drop(X_all.columns[idx], axis=1,inplace=True)
	X = X_all[len(Xtest.index):]
        Xtest = X_all[:len(Xtest.index)]
	return(X,Xtest)
	
	
def getOOBCVPredictions(lmodel,lXs,lXs_test,ly,folds=8,repeats=1,returnSD=True):
	"""
	Get cv oob predictions for classifiers
	"""
	print "Computing oob predictions..."
	if isinstance(lmodel,RandomForestClassifier) or isinstance(lmodel,SGDClassifier):
		lmodel.set_params(n_jobs=4)
	oobpreds=np.zeros((lXs.shape[0],repeats))
	for j in xrange(repeats):
	    #print lmodel.get_params()
	    cv = KFold(lXs.shape[0], n_folds=folds, indices=True,random_state=j,shuffle=True)
	    scores=np.zeros(folds)	
	    for i, (train, test) in enumerate(cv):
		Xtrain = lXs.iloc[train]
		Xtest = lXs.iloc[test]
		#print Xtest['avglinksize'].head(3)
		lmodel.fit(Xtrain, ly[train])
		oobpreds[test,j] = lmodel.predict_proba(Xtest)[:,1]
		scores[i]=roc_auc_score(ly[test],oobpreds[test,j])
		#print "AUC: %0.2f " % (scores[i])
		#save oobpredictions
	    print "Iteration:",j,
	    print " <AUC>: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std()),
	    oobscore=roc_auc_score(ly,oobpreds[:,j])
	    print " AUC,oob: %0.3f" %(oobscore)
	scores=[roc_auc_score(ly,oobpreds[:,j]) for j in xrange(repeats)]
	#simple averaging of blending
	oob_avg=np.mean(oobpreds,axis=1)
	print "Summary: <AUC,oob>: %0.3f (%d repeats)" %(roc_auc_score(ly,oob_avg),repeats,)	
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
	
def showMisclass(lXs,lXs_test,ly,t=0.95):
    """
    Show bubble plot of strongest misclassifications...
    """
    folds=8
    repeats=3
    print "Show strongly misclassified classes..."
    preds=getOOBCVPredictions(model,Xs,Xs_test,y,folds,repeats,returnSD=False)
    res=np.abs(ly-preds)
    err=pd.DataFrame({'abs_err' : pd.Series(res)})
    ly=pd.DataFrame({'y' : pd.Series(y)})
    preds=pd.DataFrame({'preds' : pd.Series(preds)})
    lXs_plot=pd.concat([ly,preds,err], axis=1)
    lXs_plot.index=lXs.index
    lXs_plot=pd.concat([lXs_plot,lXs], axis=1)
    boolindex=lXs_plot['abs_err']>t
    lXs_plot=lXs_plot[boolindex]
    print "Number of instances left:",lXs_plot.shape[0]
    col1='1'
    col2='2'
    #bubblesizes=lXs_plot['non_markup_alphanum_characters']
    bubblesizes=lXs_plot['body_length']
    print bubblesizes
    #print lXs_plot.ix[3155]
    sct = plt.scatter(lXs_plot[col1], lXs_plot[col2],c=lXs_plot['y'],s=bubblesizes, linewidths=2, edgecolor='black')
    sct.set_alpha(0.75)
    for row_index, row in lXs_plot.iterrows():
	plt.text(row[col1], row[col2],row_index,size=10,horizontalalignment='center')
	print "INDEX:",row_index," Y:",row['y']," PRED:",row['preds']," body_length:",row['body_length']
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title("red:1 blue:0")
    plt.show()
    #now got to page interactively
    
def scaleData(lXs,lXs_test,cols=None):
    """
    standard+transformation scaling of data, also possible with sklearn StandardScaler
    """
    print "Data scaling..."
    lX_all = pd.concat([lXs_test, lXs])   
    lX_all[cols].hist()   
    lX_all[cols] = (lX_all[cols] - lX_all[cols].min()+10e-10) 
    print lX_all[cols].describe()
    lX_all[cols]=lX_all[cols].apply(np.sqrt)
    lX_all[cols] = (lX_all[cols] - lX_all[cols].mean()) / (lX_all[cols].max() - lX_all[cols].min()) 
    print lX_all[cols].describe()
    lX_all[cols].hist()
    plt.show()
    
    #divide again
    lXs = lX_all[len(lXs_test.index):]
    lXs_test = lX_all[:len(lXs_test.index)]
    return (lXs,lXs_test)
    
       
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