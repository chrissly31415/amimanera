#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""  Modified starter code from BSMan@Kaggle
"""

from time import time

import json
import numpy as np
import numpy.random as nprnd
import pandas as pd
from scipy import sparse

import matplotlib.pyplot as plt
import pylab as pl

from sklearn.feature_extraction.text import CountVectorizer,HashingVectorizer,TfidfVectorizer
from sklearn import metrics
from sklearn import cross_validation,grid_search
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.utils.extmath import density
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import TruncatedSVD

from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression,RandomizedLogisticRegression,SGDClassifier,Perceptron
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,ExtraTreesClassifier,AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

#TODO build class containing classifiers + dataset
#TODO remove infrequent sparse features to new class
#TODO look at amazon challenge winners
   
def prepareDatasets(vecType='hV',useSVD=0):
    """
    Load Data into pandas and Preprocess Features
    """
    print "Reading dataset..."
    X = pd.read_csv('../stumbled_upon/data/train.tsv', sep="\t", na_values=['?'], index_col=1)
    X_test = pd.read_csv('../stumbled_upon/data/test.tsv', sep="\t", na_values=['?'], index_col=1)
    y = X['label']
    X = X.drop(['label'], axis=1)
    # Combine test and train while we do our preprocessing
    X_all = pd.concat([X_test, X])
    
    #take only boilerplate data
    X_all['boilerplate'] = X_all['boilerplate'].apply(json.loads)

    #print X_all['boilerplate']
    #print X_all['boilerplate'][2]
    # Initialize the data as a unicode string
    X_all['body'] = u'empty'
    extractBody = lambda x: x['body'] if x.has_key('body') and x['body'] is not None else u'empty'
    X_all['body'] = X_all['boilerplate'].map(extractBody)
    
    X_all['title'] = u'empty'
    extractBody = lambda x: x['title'] if x.has_key('title') and x['title'] is not None else u'empty'
    X_all['title'] = X_all['boilerplate'].map(extractBody)
    
    X_all['url'] = u'empty'
    extractBody = lambda x: x['url'] if x.has_key('url') and x['url'] is not None else u'empty'
    X_all['url'] = X_all['boilerplate'].map(extractBody)
    
    #vectorize data
    #vectorizer = HashingVectorizer(ngram_range=(1,2), non_negative=True)
    if vecType=='hV':
	vectorizer = HashingVectorizer(stop_words='english',ngram_range=(1,1), non_negative=True, norm='l2', n_features=2**18)
    elif vecType=='tV':
	vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1,1),stop_words='english',max_features=2**18)
    else:
	vectorizer = CountVectorizer(ngram_range=(1,2),analyzer=u'word',max_features=2**19)
   
    body_counts = vectorizer.fit_transform(X_all['body'])
    print "Dim after body:",body_counts.shape
    
    title_counts = vectorizer.fit_transform(X_all['title'])
    print "Dim after title:",title_counts.shape
    
    url_counts = vectorizer.fit_transform(X_all['url'])
    print "Dim after url:",url_counts.shape
    
    body_counts = sparse.hstack((body_counts,title_counts),format="csr")
    body_counts = sparse.hstack((body_counts,url_counts),format="csr")
    print "Final dim:",body_counts.shape
    
    
    
    feature_names = None
    #if hasattr(vectorizer, 'get_feature_names'):
    #	feature_names = np.asarray(vectorizer.get_feature_names())
    #X & X_test are converted to sparse matrix
    
    #bringt seltsamerweise nichts
    #X_alcat=pd.DataFrame(X_all['alchemy_category'])
    #X_alcat=X_alcat.fillna('NA')
    #X_alcat = one_hot_encoder(X_alcat, ['alchemy_category'], replace=True) 
    #print X_alcat
    #X_alcat=sparse.csr_matrix(pd.np.array(X_alcat))
    #body_counts = sparse.hstack((body_counts,X_alcat),format="csr")

    y = pd.np.array(y)
    if useSVD>1:
	#SVD of text data (LSA)
	tsvd=TruncatedSVD(n_components=useSVD, algorithm='randomized', n_iterations=5, random_state=42, tol=0.0)
	X_svd=tsvd.fit_transform(body_counts)
	X_svd=pd.DataFrame(np.asarray(X_svd))
	print "X_svd:",X_svd
	X_rest= X_all.drop(['body','url','alchemy_category','boilerplate','title'], axis=1)
	X_rest = X_rest.astype(float)
	#print X_rest   
	X_rest=X_rest.fillna(X_rest.mean())
	print "X_rest:",X_rest
	#X_rest=X_rest.append(X_svd,ignore_index=True)
	X_svd=np.concatenate((X_rest, X_svd), axis=1)
	#X_rest=X_svd
	print "X_svd:",X_svd.shape    
	X_svd_train = X_svd[len(X_test.index):]
	X_svd_test = X_svd[:len(X_test.index)]
	return(X_svd_train,y,X_svd_test,X_test.index)
    else:
	Xs = body_counts[len(X_test.index):]
	Xs_test = body_counts[:len(X_test.index)]
	#conversion to array necessary to work with integer indexing, .iloc does not work with this version
	return (Xs,y,Xs_test,X_test.index)
	
    

def one_hot_encoder(data, cols, replace=False):
    """ Takes a dataframe and a list of columns that need to be encoded.
    Returns a 3-tuple comprising the data, the vectorized data,
    and the fitted vectorizor.
    credits to https://gist.github.com/kljensen/5452382
    """
    vec = DictVectorizer()
    mkdict = lambda row: dict((col, row[col]) for col in cols)
    vecData = pd.DataFrame(vec.fit_transform(data[cols].apply(mkdict, axis=1)).toarray())
    vecData.columns = vec.get_feature_names()
    vecData.index = data.index
    if replace is True:
	data = data.drop(cols, axis=1)
	data = data.join(vecData)
    return data
    
    
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
	print("Density: %f" % density(lmodel.coef_))
	if feature_names is not None:
	  top10 = np.argsort(lmodel.coef_)[0,-10:][::-1]
	  #print model.coef_[top10b]
	  for i in xrange(top10.shape[0]):
	      print("Top %2d: coef: %0.3f %20s" % (i+1,lmodel.coef_[0,top10[i]],feature_names[top10[i]]))

def modelEvaluation(lmodel,lXs,ly):
    """
    MODEL EVALUATION
    """
    print "Model evaluation..."
    folds=5
    #parameters=np.logspace(-14, -7, num=8, base=2.0)#SDG
    #parameters=np.logspace(-7, 0, num=8, base=2.0)#LG
    #parameters=[250,500,1000,2000]#rf
    parameters=[2,3,4,5]#gbm
    #parameters=np.logspace(-7, -0, num=8, base=2.0)
    print "Parameter space:",parameters
    #feature selection within xvalidation
    oobpreds=np.zeros((lXs.shape[0],len(parameters)))
    for j,p in enumerate(parameters):
	if isinstance(lmodel,SGDClassifier):
	    lmodel.set_params(alpha=p)
	if isinstance(lmodel,LogisticRegression) or isinstance(lmodel,SVC):
	    lmodel.set_params(C=p)
	if isinstance(lmodel,RandomForestClassifier) :
	    lmodel.set_params(n_estimators=p)
	if isinstance(lmodel,GradientBoostingClassifier):
	    lmodel.set_params(max_depth=p)
        print lmodel.get_params()
        cv = StratifiedKFold(ly, n_folds=folds, indices=True)
	scores=np.zeros(folds)	
	for i, (train, test) in enumerate(cv):
	    #print("Extracting %d best features by a chi-squared test" % n_features)
	    #ch2 = SelectKBest(chi2, k=7500)
	    #Xtrain = ch2.fit_transform(lXs[train], ly[train])
	    #Xtest = ch2.transform(lXs[test])  
	    Xtrain = lXs[train]
	    Xtest = lXs[test]
	    lmodel.fit(Xtrain, ly[train])
	    oobpreds[test,j] = lmodel.predict_proba(Xtest)[:,1]
	    scores[i]=roc_auc_score(ly[test],oobpreds[test,j])
	    #print "AUC: %0.2f " % (scores[i])
	    #save oobpredictions
	print "Iteration:",j," parameter:",p,
	print " <AUC>: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2),
	print " AUC oob: %0.3f" %(roc_auc_score(ly,oobpreds[:,j]))
    scores=[roc_auc_score(ly,oobpreds[:,j]) for j in xrange(len(parameters))]
    plt.plot(parameters,scores,'ro')
    

def sigmoid(x):
  y = 1.0/(1.0 + exp(-x))
  return(y)
    

def rfFeatureImportance(forest):
    if not hasattr(forest,'feature_importances_'): return
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
    indices = np.argsort(importances)[::-1]
    print indices.shape
    print importances.shape
    # Print the feature ranking
    print("Feature ranking:")

    for f in range(len(indices)):
	print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    
    plt.bar(left=np.arange(len(indices)),height=importances[indices] , width=0.35, color='r')
    plt.ylabel('Importance')
    plt.title("Feature importances")

    
    
def ensembleBuilding(lXs,ly):
    """
    train ensemble
    """
    print "Ensemble training..."
    folds=10
    parameters=np.logspace(-14, -7, num=50, base=2.0)
    #parameters=nprnd.choice(parameters, 8)
    classifiers = {}
    for p in parameters:
        l1ratio=nprnd.ranf()
	dic ={'SDG_alpha'+str(p)+'_L1'+str(l1ratio): SGDClassifier(alpha=p, n_iter=50,penalty='elasticnet',l1_ratio=l1ratio,shuffle=True,random_state=round(42*l1ratio),loss='log')}
	classifiers.update(dic)
    #dic ={'NB': BernoulliNB(alpha=1.0)}
    #classifiers.update(dic)
    dic ={'LG1': LogisticRegression(penalty='l2', tol=0.0001, C=.25,random_state=42)}
    classifiers.update(dic)
    dic ={'SDG1': SGDClassifier(alpha=0.0005, n_iter=50,shuffle=True,random_state=42,loss='log',penalty='l2')}
    classifiers.update(dic)
    #dic ={'SDG2': SGDClassifier(alpha=0.0005, n_iter=50,shuffle=True,random_state=42,loss='log',penalty='l1')}
    #classifiers.update(dic)
    #dic ={'LG2': LogisticRegression(penalty='l1', tol=0.0001, C=1.0,random_state=42)}
    #classifiers.update(dic)
    #dic ={'KNN': KNeighborsClassifier(n_neighbors=5)}
    #classifiers.update(dic)  
    #dic ={'SDG3': SGDClassifier(alpha=.0001220703125, n_iter=50,penalty='elasticnet',l1_ratio=0.2,shuffle=True,random_state=42,loss='log')}
    #classifiers.update(dic)
    oobpreds=np.zeros((lXs.shape[0],len(classifiers)))
    for j,(key, lmodel) in enumerate(classifiers.iteritems()):
        #print lmodel.get_params()
        cv = StratifiedKFold(ly, n_folds=folds, indices=True)
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
	print " <AUC>: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2),
	print " AUC oob: %0.3f" %(roc_auc_score(ly,oobpreds[:,j]))
    scores=[roc_auc_score(ly,oobpreds[:,j]) for j in xrange(len(classifiers))]
    #simple averaging of blending
    oob_avg=np.mean(oobpreds,axis=1)
    print " AUC oob, simple mean: %0.3f" %(roc_auc_score(ly,oob_avg))
    
    #do another crossvalidation for weights
    blender=LogisticRegression(penalty='l2', tol=0.0001, C=1.0)
    cv = StratifiedKFold(ly, n_folds=folds, indices=True)
    blend_scores=np.zeros(folds)
    blend_oob=np.zeros((lXs.shape[0]))
    for i, (train, test) in enumerate(cv):
	Xtrain = oobpreds[train]
	Xtest = oobpreds[test]
	blender.fit(Xtrain, ly[train])
	blend_oob[test] = blender.predict_proba(Xtest)[:,1]
	blend_scores[i]=roc_auc_score(ly[test],blend_oob[test])
    print " <AUC>: %0.3f (+/- %0.3f)" % (blend_scores.mean(), blend_scores.std() * 2),
    print " AUC oob after blending: %0.3f" %(roc_auc_score(ly,blend_oob))
    print "Coefficients:",blender.coef_
    
    plt.plot(range(len(classifiers)),scores,'ro')
    return(classifiers,blender)
    

def ensemblePredictions(classifiers,blender,lXs_test,lidx,filename):
    """   
    Makes prediction
    """ 
    print "Make final ensemble prediction..."
    #make prediction for each classifiers
    preds=np.zeros((lXs_test.shape[0],len(classifiers)))
    for j,(key, lmodel) in enumerate(classifiers.iteritems()):
	preds[:,j]=lmodel.predict_proba(lXs_test)[:,1]
    #blend results
    finalpred=blender.predict_proba(preds)[:,1]   
    print "Saving predictions to: ",filename
    print "Final test dataframe:",lXs_test.shape
    pred_df = pd.DataFrame(finalpred, index=lidx, columns=['label'])
    pred_df.to_csv(filename)
    
    

def pyGridSearch(lmodel,lXs,ly):  
    """   
    Grid search with sklearn internal tool
    """ 
    print "Grid search..."
    parameters = {'C':[1000,10000,100], 'gamma':[0.001,0.0001]}
    #parameters = {'max_depth':[2,3, 4], 'learning_rate':[0.01, 0.001],'n_estimators':[2000,3000]}#gbm
    #parameters = {'max_depth':[3,4], 'learning_rate':[0.001,0.01],'n_estimators':[1000]}#gbm
    #parameters = {'n_estimators':[1000,1500], 'max_features':[5,8,10,12]}#rf
    clf_opt = grid_search.GridSearchCV(lmodel, parameters,cv=5,scoring='roc_auc',n_jobs=4,verbose=1)
    clf_opt.fit(lXs,ly)
    print type(clf_opt.grid_scores_)
    
    for params, mean_score, scores in clf_opt.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score.mean(), scores.std(), params))
    return(clf_opt.best_estimator_)
    
def buildModel(lmodel,lXs,ly,feature_names=None):
    """   
    Final model building part
    """ 
    print "Final Xval..."
    scores = cross_validation.cross_val_score(lmodel, lXs, ly, cv=5, scoring='roc_auc',n_jobs=4)
    print "AUC: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2)
    print "Build final model..."
    lmodel.fit(lXs,ly)
    #analyzeModel(lmodel,feature_names)
    return(lmodel)
 
if __name__=="__main__":
    """   
    MAIN PART
    """ 
    # Set a seed for consistant results
    t0 = time()
    np.random.seed(1234)
    #variables
    (Xs,y,Xs_test,data_indices) = prepareDatasets('tV',useSVD=20)
    print "Dim X (training):",Xs.shape
    print "Type X:",type(Xs)
    # Fit a model and predict
    #model = BernoulliNB(alpha=1.0)
    #model = SGDClassifier(alpha=.0001, n_iter=50,penalty='elasticnet',l1_ratio=0.2,shuffle=True,random_state=42,loss='log')
    #model = SGDClassifier(alpha=0.00048828125, n_iter=50,shuffle=True,random_state=42,loss='log',penalty='l2')#opt
    #model = LogisticRegression(penalty='l2', tol=0.0001, C=.5)#opt
    #model = RandomizedLogisticRegression(C=1, scaling=0.5, sample_fraction=0.75, n_resampling=200, selection_threshold=0.25, tol=0.001, fit_intercept=True, verbose=False, normalize=True, random_state=42)
    #model = KNeighborsClassifier(n_neighbors=10)
    model = SVC(C=1, cache_size=200, class_weight='auto', gamma=0.0, kernel='rbf', probability=True, shrinking=True,tol=0.001, verbose=False)
    #model = RandomForestClassifier(n_estimators=3000,max_depth=None,min_samples_leaf=5,n_jobs=1,criterion='entropy', max_features=10,oob_score=False,random_state=42)
    #model = ExtraTreesClassifier(n_estimators=500,max_depth=None,min_samples_leaf=5,n_jobs=1,criterion='entropy', max_features='auto',oob_score=False,random_state=42)
    #model = AdaBoostClassifier(n_estimators=500,learning_rate=0.1,random_state=42)
    #model = GradientBoostingClassifier(loss='deviance', learning_rate=0.01, n_estimators=2000, subsample=1.0, min_samples_split=2, min_samples_leaf=1, max_depth=3, init=None, random_state=42,verbose=False)
    #model = SVC(C=1, cache_size=200, class_weight='auto', gamma=0.0, kernel='rbf', probability=True, shrinking=True,tol=0.001, verbose=False)  
    #modelEvaluation(model,Xs,y)
    model=pyGridSearch(model,Xs,y)
    #(gclassifiers,gblender)=ensembleBuilding(Xs,y)
    #ensemblePredictions(gclassifiers,gblender,Xs_test,data_indices,'sub2808a.csv')
    #fit final model    
    model = buildModel(model,Xs,y)
    #rfFeatureImportance(model)
    makePredictions(model,Xs_test,data_indices,'../stumbled_upon/submissions/sub3008a.csv')	            
    print("Model building done in %fs" % (time() - t0))
    plt.show()
