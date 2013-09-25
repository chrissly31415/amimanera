#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""  Chrissly31415
August,September 2013

"""

from stumble import *


def createModels():
    ensemble=[]
    #logistic regression, sparse matric
    (Xs,y,Xs_test,data_indices) = prepareDatasets('tfidfV',useSVD=0,useJson=True,useHTMLtag=False,useAddFeatures=False,usePosTag=False,useAlcat=False,useGreedyFilter=False)   
    model = LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1.0, class_weight=None, random_state=None)
    xmodel = XModel("LogisticRegression_sparse",model,Xs,Xs_test)
    ensemble.append(xmodel)
    #rf, using SVD
    (X,y,X_test,data_indices) = prepareDatasets('tfidfV',useSVD=50,useJson=True,useHTMLtag=False,useAddFeatures=False,usePosTag=False,useAlcat=False,useGreedyFilter=True)
    model = RandomForestClassifier(n_estimators=100,max_depth=None,min_samples_leaf=12,n_jobs=1,criterion='entropy', max_features=4,oob_score=False,random_state=42)
    xmodel = XModel("RandomForest_dense",model,X,X_test)
    ensemble.append(xmodel)
    #extrarf, using SVD
    #(X,y,X_test,data_indices) = prepareDatasets('tfidfV',useSVD=50,useJson=True,useHTMLtag=False,useAddFeatures=False,usePosTag=False,useAlcat=False,useGreedyFilter=False)
    model  = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=50,verbose=False)
    xmodel = XModel("GradBoost_dense",model,X,X_test)
    ensemble.append(xmodel)
    
    #collect them
    for m in ensemble:
      m.summary()
    return(ensemble,y)

def createOOBdata(ensemble,ly,repeats=1):
    """
    Get cv oob predictions for classifiers
    """
    folds=8
    for m in ensemble:
	print "Computing oob predictions for:",m.name
	print m.classifier.get_params
	oobpreds=np.zeros((m.Xtrain.shape[0],repeats))
	for j in xrange(repeats):
	    #print lmodel.get_params()
	    cv = KFold(m.Xtrain.shape[0], n_folds=folds, indices=True,random_state=j,shuffle=True)
	    scores=np.zeros(folds)	
	    for i, (train, test) in enumerate(cv):
		if not m.sparse:
		    Xtrain = m.Xtrain.iloc[train]
		    Xtest = m.Xtrain.iloc[test]
		else:
		    Xtrain = m.Xtrain[train]
		    Xtest = m.Xtrain[test]
		#print Xtest['avglinksize'].head(3)
		m.classifier.fit(Xtrain, ly[train])
		oobpreds[test,j] = m.classifier.predict_proba(Xtest)[:,1]
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
	m.oobpreds=oob_avg
	#save oob + predictions to disc
	print "Train full modell and generate predictions..."
	m.classifier.fit(m.Xtrain,ly)
	m.preds = m.classifier.predict_proba(m.Xtest)[:,1]
	m.summary()
	
    return(ensemble)
      
  
def trainFullModels():
    pass


def trainEnsemble():
    pass


if __name__=="__main__":
    ensemble,y=createModels()
    ensemble=createOOBdata(ensemble,y)