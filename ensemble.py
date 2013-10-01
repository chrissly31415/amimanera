#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""  Chrissly31415
August,September 2013

"""

from stumble import *
from scipy.optimize import fmin

train_indices=[]
test_indices=[]

def createModels():
    global train_indices,test_indices
    ensemble=[]
    
    #logistic regression, sparse matric
    #(Xs,y,Xs_test,test_indices,train_indices) = prepareDatasets('tfidfV',useSVD=0,useJson=True,useHTMLtag=False,useAddFeatures=False,usePosTag=False,useAlcat=False,useGreedyFilter=False)  
    #model = LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1.0, class_weight=None, random_state=None)
    #xmodel = XModel("logreg1",model,Xs,Xs_test)
    #ensemble.append(xmodel)
    
    #logistic regression, sparse matric, count vetcorizer,5,5 character ngrams
    #(Xs,y,Xs_test,test_indices,train_indices) = prepareDatasets('cV',useSVD=0,useJson=True,useHTMLtag=False,useAddFeatures=False,usePosTag=False,useAlcat=False,useGreedyFilter=False)  
    #model = LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1.0, class_weight=None, random_state=None)
    #xmodel = XModel("logreg2_cv",model,Xs,Xs_test)
    #ensemble.append(xmodel)
    
    #pipeline with sdg
    #(Xs,y,Xs_test,test_indices,train_indices) = prepareDatasets('tfidfV',useSVD=0,useJson=True,useHTMLtag=False,useAddFeatures=False,usePosTag=False,useAlcat=False,useGreedyFilter=False)  
    #model = Pipeline([('filter', SelectPercentile(chi2, percentile=98)), ('model', SGDClassifier(alpha=0.00014, n_iter=50,shuffle=True,random_state=42,loss='log',penalty='elasticnet',l1_ratio=0.99))])
    #xmodel = XModel("sgd1",model,Xs,Xs_test)
    #ensemble.append(xmodel)
    
    #normal sdg
    #(Xs,y,Xs_test,test_indices,train_indices) = prepareDatasets('tfidfV',useSVD=0,useJson=True,useHTMLtag=False,useAddFeatures=False,usePosTag=False,useAlcat=False,useGreedyFilter=False)  
    #model = SGDClassifier(alpha=0.0005, n_iter=50,shuffle=True,random_state=42,loss='log',penalty='l2',n_jobs=4)
    #xmodel = XModel("sgd2",model,Xs,Xs_test)
    #ensemble.append(xmodel)
    
    
    #rf, using SVD
    #(X,y,X_test,test_indices,train_indices) = prepareDatasets('tfidfV',useSVD=50,useJson=True,useHTMLtag=True,useAddFeatures=True,usePosTag=True,useAlcat=False,useGreedyFilter=True)
    #model = RandomForestClassifier(n_estimators=500,max_depth=None,min_samples_leaf=12,n_jobs=1,criterion='entropy', max_features=4,oob_score=False,random_state=42)
    #xmodel = XModel("randomf1",model,X,X_test)
    #ensemble.append(xmodel)
    
    #rf, using SVD
    #(X,y,X_test,test_indices,train_indices) = prepareDatasets('tfidfV',useSVD=100,useJson=True,useHTMLtag=True,useAddFeatures=True,usePosTag=True,useAlcat=False,useGreedyFilter=False)
    #model = RandomForestClassifier(n_estimators=500,max_depth=None,min_samples_leaf=12,n_jobs=4,criterion='entropy', max_features='auto',oob_score=False,random_state=42)
    #xmodel = XModel("randomf2",model,X,X_test)
    #ensemble.append(xmodel)
    
    #xrf, using SVD
    #(X,y,X_test,test_indices,train_indices) = prepareDatasets('tfidfV',useSVD=50,useJson=True,useHTMLtag=False,useAddFeatures=False,usePosTag=True,useAlcat=False,useGreedyFilter=False)
    #model = ExtraTreesClassifier(n_estimators=500,max_depth=None,n_jobs=4,criterion='gini', max_features='auto',min_samples_leaf=10,oob_score=False,random_state=42)
    #xmodel = XModel("extrarf1",model,X,X_test)
    #ensemble.append(xmodel)
    
    #gradient boosting, using SVD
    #(X,y,X_test,test_indices,train_indices) = prepareDatasets('tfidfV',useSVD=50,useJson=True,useHTMLtag=True,useAddFeatures=True,usePosTag=True,useAlcat=False,useGreedyFilter=False)
    #model  = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=150,verbose=False)
    #xmodel = XModel("gradboost1",model,X,X_test)
    #ensemble.append(xmodel)
    
    """1,1character ngrams
    (Xs,y,Xs_test,test_indices,train_indices) = prepareDatasets('cV',useSVD=0,useJson=False,useHTMLtag=False,useAddFeatures=False,usePosTag=False,useAlcat=False,useGreedyFilter=False,char_ngram=1)  
    model = LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1.0, class_weight=None, random_state=42)
    xmodel = XModel("lr_char11",model,Xs,Xs_test)
    ensemble.append(xmodel)
    """
    
    
    """2,2 character ngrams
    """
    (Xs,y,Xs_test,test_indices,train_indices) = prepareDatasets('cV',useSVD=0,useJson=False,useHTMLtag=False,useAddFeatures=False,usePosTag=False,useAlcat=False,useGreedyFilter=False,char_ngram=2)  
    model = LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1.0, class_weight=None, random_state=42)
    xmodel = XModel("lr_char22",model,Xs,Xs_test)
    ensemble.append(xmodel)

    
    """3,3 character ngrams
    """
    (Xs,y,Xs_test,test_indices,train_indices) = prepareDatasets('cV',useSVD=0,useJson=False,useHTMLtag=False,useAddFeatures=False,usePosTag=False,useAlcat=False,useGreedyFilter=False,char_ngram=3)  
    model = LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1.0, class_weight=None, random_state=42)
    xmodel = XModel("lr_char33",model,Xs,Xs_test)
    ensemble.append(xmodel)

    
    """4,4 character ngrams
    """
    (Xs,y,Xs_test,test_indices,train_indices) = prepareDatasets('cV',useSVD=0,useJson=False,useHTMLtag=False,useAddFeatures=False,usePosTag=False,useAlcat=False,useGreedyFilter=False,char_ngram=4)  
    model = LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1.0, class_weight=None, random_state=42)
    xmodel = XModel("lr_char44",model,Xs,Xs_test)
    ensemble.append(xmodel)
  
    
    """5,5 character ngrams
    """
    (Xs,y,Xs_test,test_indices,train_indices) = prepareDatasets('cV',useSVD=0,useJson=False,useHTMLtag=False,useAddFeatures=False,usePosTag=False,useAlcat=False,useGreedyFilter=False,char_ngram=5)  
    model = LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1.0, class_weight=None, random_state=42)
    xmodel = XModel("lr_char55",model,Xs,Xs_test)
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
	print "Summary: <AUC,oob>: %0.3f (%d repeats)" %(roc_auc_score(ly,oob_avg),repeats)
	m.oobpreds=oob_avg
	#save oob + predictions to disc
	print "Train full modell and generate predictions..."
	m.classifier.fit(m.Xtrain,ly)
	m.preds = m.classifier.predict_proba(m.Xtest)[:,1]
	m.summary()
	#put data to data.frame and save
	m.oobpreds=pd.DataFrame(np.asarray(m.oobpreds),columns=["label"],index=train_indices)
	m.preds=pd.DataFrame(np.asarray(m.preds),columns=["label"],index=test_indices)
	allpred = pd.concat([m.preds, m.oobpreds])
	allpred.to_csv("../stumbled_upon/data/"+m.name+".csv")	
    return(ensemble)
   
def trainEnsemble(addMetaFeatures=False):
    test_indices = pd.read_csv('../stumbled_upon/data/test.tsv', sep="\t", na_values=['?'], index_col=1).index
    y = np.asarray(pd.read_csv('../stumbled_upon/data/train.tsv', sep="\t", na_values=['?'], index_col=1)['label'])
    #ensemble=["logreg1","sgd1","randomf1","randomf2","randomf3","gradboost1","gbmR"]
    ensemble=["logreg1","logreg2_cv","sgd1","sgd2","randomf1","randomf2","gradboost1","extrarf1","gbmR","lof"]
    ensemble=["logreg1","logreg2_cv","sgd2","randomf1","randomf2","gradboost1","gbmR"]#best so far 0.885
    #ensemble=["logreg1","logreg2_cv","sgd2","randomf1","randomf2","gradboost1","gbmR","lof"]
    for i,model in enumerate(ensemble):
	print "Loading model:",i," name:",model
	if i>0:
	    X = pd.read_csv("../stumbled_upon/data/"+model+".csv", sep=",", index_col=0)
	    X.columns=[model]
	    Xall=pd.concat([Xall,X], axis=1)
	else:
	    Xall = pd.read_csv("../stumbled_upon/data/"+model+".csv", sep=",", index_col=0)
	    Xall.columns=[model]
    
    if addMetaFeatures:
	pass
    
    Xtrain=Xall[len(test_indices):]
    print Xtrain
    Xtest=Xall[:len(test_indices)]
    print Xtest
    aucMinimize(ensemble,Xtrain,Xtest,y,test_indices,takeMean=False)
    #classicalBlend(ensemble,Xtrain,Xtest,y,test_indices)


def classicalBlend(ensemble,oobpreds,testset,ly,test_indices):
    #blending
    folds=8
    #do another crossvalidation for weights
    blender=LogisticRegression(penalty='l2', tol=0.0001, C=1.0)
    #blender=SGDClassifier(alpha=.005, n_iter=50,penalty='l2',shuffle=True,random_state=42,loss='log')
    #blender=AdaBoostClassifier(learning_rate=0.1,n_estimators=100,algorithm="SAMME.R")
    #blender=ExtraTreesRegressor(n_estimators=200,max_depth=None,n_jobs=1, max_features='auto',oob_score=False,random_state=42)
    #blender=ExtraTreesClassifier(n_estimators=50,max_depth=None,min_samples_leaf=10,n_jobs=1,criterion='entropy', max_features='auto',oob_score=False,random_state=42)
    cv = KFold(oobpreds.shape[0], n_folds=folds, indices=True,random_state=42)
    blend_scores=np.zeros(folds)
    blend_oob=np.zeros((oobpreds.shape[0]))
    for i, (train, test) in enumerate(cv):
	Xtrain = oobpreds.iloc[train]
	Xtest = oobpreds.iloc[test]
	blender.fit(Xtrain, ly[train])
	if hasattr(blender,'predict_proba'):
	    blend_oob[test] = blender.predict_proba(Xtest)[:,1]
	else:
	    blend_oob[test] = blender.predict(Xtest)
	blend_scores[i]=roc_auc_score(ly[test],blend_oob[test])
    print " <AUC>: %0.3f (+/- %0.3f)" % (blend_scores.mean(), blend_scores.std()),
    print " AUC oob after blending: %0.3f" %(roc_auc_score(ly,blend_oob))
    if hasattr(blender,'coef_'):
      print blender.coef_
      for i,model in enumerate(ensemble):
	print "%-16s   coef: %4.4f" %(model,blender.coef_[0][i])
      print "Sum: %4.4f"%(np.sum(blender.coef_))

    
    #plt.plot(range(len(ensemble)),scores,'ro')
    
    #Prediction



def aucMinimize(ensemble,Xtrain,Xtest,y,test_indices,takeMean=False):
    #http://www.kaggle.com/c/amazon-employee-access-challenge/forums/t/4928/combining-the-results-of-various-models?page=3
    def fopt(pars):
	fpr, tpr, thresholds = metrics.roc_curve(y, np.dot(Xtrain,pars))
	auc=metrics.auc(fpr, tpr)
	#print "auc:",auc
	return -auc
    n_models=len(ensemble)
    x0 = np.ones((n_models, 1)) / n_models
    xopt = fmin(fopt, x0)
    if takeMean==True:
	xopt=x0
	
    else:
	fpr, tpr, thresholds = metrics.roc_curve(y, np.dot(Xtrain,x0))
	auc=metrics.auc(fpr, tpr)
	print "->AUC,mean: %4.4f" %(auc)
	fpr, tpr, thresholds = metrics.roc_curve(y, np.dot(Xtrain,xopt))
	auc=metrics.auc(fpr, tpr)
	print "->AUC,opt: %4.4f" %(auc)
    
    for i,model in enumerate(ensemble):
	print "%-16s   coef: %4.4f" %(model,xopt[i])
    print "Sum: %4.4f"%(np.sum(xopt))
    
    preds=pd.DataFrame(np.dot(Xtest,xopt),columns=["label"],index=test_indices)
    preds.to_csv('../stumbled_upon/submissions/sub2809a.csv')
    print preds
  


if __name__=="__main__":
    ensemble,y=createModels()
    ensemble=createOOBdata(ensemble,y,10)
    #trainEnsemble()
    