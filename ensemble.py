#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""  Chrissly31415
August,September 2013

"""

from stumble import *
from scipy.optimize import fmin,fmin_cobyla

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
    
    #naive bayes
    #(X,y,X_test,test_indices,train_indices) = prepareDatasets('tfidfV',useSVD=50,useJson=True,useHTMLtag=True,useAddFeatures=True,usePosTag=True,useAlcat=True,useGreedyFilter=False)
    #model = Pipeline([('filter', SelectPercentile(f_classif, percentile=25)), ('model', BernoulliNB(alpha=0.1))])#opt dense 0.855
    #xmodel = XModel("naiveB1",model,X,X_test)
    #ensemble.append(xmodel)
    
    #rf, using SVD, using 41 variables selectd by rf feature selection ~Auc,cv=0.882
    #(X,y,X_test,test_indices,train_indices) = prepareDatasets('tfidfV',useSVD=50,useJson=True,useHTMLtag=True,useAddFeatures=True,usePosTag=True,useAlcat=False,useGreedyFilter=True)
    #model = RandomForestClassifier(n_estimators=500,max_depth=None,min_samples_leaf=10,n_jobs=4,criterion='entropy', max_features='auto',oob_score=False,random_state=42)
    #xmodel = XModel("randomf1",model,X,X_test)
    #ensemble.append(xmodel)
    
    #rf, using SVD 0.884
    #(X,y,X_test,test_indices,train_indices) = prepareDatasets('tfidfV',useSVD=100,useJson=True,useHTMLtag=True,useAddFeatures=True,usePosTag=True,useAlcat=False,useGreedyFilter=True)
    #model = RandomForestClassifier(n_estimators=500,max_depth=None,min_samples_leaf=10,n_jobs=4,criterion='entropy', max_features=15,oob_score=False)
    #xmodel = XModel("randomf2",model,X,X_test)
    #ensemble.append(xmodel)
    
    #rf, using brute force sparse to dense...takes VERY LONG ~0.870
    """
    (Xs,y,Xs_test,test_indices,train_indices) = prepareDatasets('tfidfV_small',useSVD=0,useJson=True,useHTMLtag=False,useAddFeatures=False,usePosTag=False,useAlcat=False,useGreedyFilter=False)
    X=pd.DataFrame(Xs.todense())
    X_test=pd.DataFrame(Xs_test.todense())
    model= RandomForestClassifier(n_estimators=500,max_depth=None,min_samples_leaf=5,n_jobs=4,criterion='entropy', max_features=100,oob_score=False,random_state=42)
    xmodel = XModel("randomf3",model,X,X_test)
    ensemble.append(xmodel)
    """
    
    #KNN fastest model 0.870
    #(X,y,X_test,test_indices,train_indices) = prepareDatasets('tfidfV',useSVD=50,useJson=True,useHTMLtag=True,useAddFeatures=True,usePosTag=True,useAlcat=True,useGreedyFilter=True)  
    #model= Pipeline([('filter', SelectPercentile(f_classif, percentile=15)), ('model', KNeighborsClassifier(n_neighbors=150))])
    #xmodel = XModel("knn1",model,X,X_test)
    #ensemble.append(xmodel)
    
    #xrf, using SVD 0.883
    (X,y,X_test,test_indices,train_indices) = prepareDatasets('tfidfV',useSVD=100,useJson=True,useHTMLtag=True,useAddFeatures=True,usePosTag=True,useAlcat=False,useGreedyFilter=True)
    model = ExtraTreesClassifier(n_estimators=600,max_depth=None,min_samples_leaf=10,n_jobs=4,criterion='entropy', max_features=20,oob_score=False)
    xmodel = XModel("extrarf2",model,X,X_test)
    ensemble.append(xmodel)
    
    #gradient boosting, using SVD 0.883, feature have been reduced to 60
    #(X,y,X_test,test_indices,train_indices) = prepareDatasets('tfidfV',useSVD=100,useJson=True,useHTMLtag=True,useAddFeatures=True,usePosTag=True,useAlcat=False,useGreedyFilter=True,loadTemp=True)
    #model  = GradientBoostingClassifier(loss='deviance', learning_rate=0.01, n_estimators=500, subsample=0.5, min_samples_split=6, min_samples_leaf=10, max_depth=5, init=None, random_state=123,verbose=False)
    #xmodel = XModel("gradboost2",model,X,X_test)
    #ensemble.append(xmodel)
    
    """1,1character ngrams  
    (Xs,y,Xs_test,test_indices,train_indices) = prepareDatasets('hV',useSVD=0,useJson=True,useHTMLtag=False,useAddFeatures=False,usePosTag=False,useAlcat=False,useGreedyFilter=False,char_ngram=1)  
    model = LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1.0, class_weight=None, random_state=42)
    xmodel = XModel("lr_char11_hV",model,Xs,Xs_test)
    ensemble.append(xmodel)
    """
       
    """2,2 character ngrams  
    (Xs,y,Xs_test,test_indices,train_indices) = prepareDatasets('hV',useSVD=0,useJson=True,useHTMLtag=False,useAddFeatures=False,usePosTag=False,useAlcat=False,useGreedyFilter=False,char_ngram=2)  
    model = LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1.0, class_weight=None, random_state=42)
    xmodel = XModel("lr_char22_hV",model,Xs,Xs_test)
    ensemble.append(xmodel)
    """
    
    """3,3 character ngrams  
    (Xs,y,Xs_test,test_indices,train_indices) = prepareDatasets('hV',useSVD=0,useJson=True,useHTMLtag=False,useAddFeatures=False,usePosTag=False,useAlcat=False,useGreedyFilter=False,char_ngram=3)  
    model = LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1.0, class_weight=None, random_state=42)
    xmodel = XModel("lr_char33_hV",model,Xs,Xs_test)
    ensemble.append(xmodel)
    """
    
    """4,4 character ngrams   
    (Xs,y,Xs_test,test_indices,train_indices) = prepareDatasets('hV',useSVD=0,useJson=True,useHTMLtag=False,useAddFeatures=False,usePosTag=False,useAlcat=False,useGreedyFilter=False,char_ngram=4)  
    model = LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1.0, class_weight=None, random_state=42)
    xmodel = XModel("lr_char44_hV",model,Xs,Xs_test)
    ensemble.append(xmodel)
    """
    
    """5,5 character ngrams
    (Xs,y,Xs_test,test_indices,train_indices) = prepareDatasets('hV',useSVD=0,useJson=True,useHTMLtag=False,useAddFeatures=False,usePosTag=False,useAlcat=False,useGreedyFilter=False,char_ngram=5)  
    model = LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1.0, class_weight=None, random_state=42)
    xmodel = XModel("lr_char55_hV",model,Xs,Xs_test)
    ensemble.append(xmodel)
    """
    
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
    #ensemble=["logreg1","logreg2_cv","sgd1","sgd2","randomf1","randomf2","gradboost1","extrarf1","gbmR","lof"]
    #ensemble=["logreg1","logreg2_cv","sgd2","randomf1","randomf2","gradboost1","gbmR"]#best so far 0.885
    #ensemble=["lgblend","logreg1","logreg2_cv","sgd1","sgd2","naiveB1","randomf1","randomf2","randomf3","extrarf1","extrarf2","gradboost1","gradboost2","gbmR","gbmR2","lr_char22_hV","lr_char33_hV","lr_char44_hV","lr_char55_hV","lr_char11","lr_char22","lr_char33","lr_char44","lr_char55"]
    ensemble=["naiveB1","extrarf2","sgd1","randomf2","gradboost2","lr_char33_hV","lr_char44_hV","lr_char55_hV","lr_char33","gbmR","lof"]
    #ensemble=["gradboost2","gbmR","randomf2","extrarf2"]
    #ensemble=["sgd1","naiveB1","randomf1","randomf2","gbmR","lr_char33_hV","lr_char44_hV","lr_char55_hV","lr_char33"]

    for i,model in enumerate(ensemble):
	print "Loading model:",i," name:",model
	if i>0:
	    X = pd.read_csv("../stumbled_upon/data/"+model+".csv", sep=",", index_col=0)
	    X.columns=[model]
	    Xall=pd.concat([Xall,X], axis=1)
	else:
	    Xall = pd.read_csv("../stumbled_upon/data/"+model+".csv", sep=",", index_col=0)
	    Xall.columns=[model]
    
    
    
    Xtrain=Xall[len(test_indices):]
    Xtest=Xall[:len(test_indices)]
    #if we add metafeature we should not use aucMinimize...
    if addMetaFeatures:
	#(Xs,y,Xs_test,test_indices,train_indices) = prepareDatasets('tfidfV',useSVD=2,useJson=True,useHTMLtag=True,useAddFeatures=True,usePosTag=True,useAlcat=True,useGreedyFilter=True)
	Xs = pd.read_csv('../stumbled_upon/data/Xens.csv', sep=",", index_col=0)
	Xs_test = pd.read_csv('../stumbled_upon/data/Xens_test.csv', sep=",", index_col=0)
	useCols=['linkwordscore','non_markup_alphanum_characters','frameTagRatio']
	Xs=Xs.loc[:,useCols]
	Xs_test=Xs_test.loc[:,useCols]
	Xtrain=pd.concat([Xtrain,Xs], axis=1)
	Xtest=pd.concat([Xtest,Xs_test], axis=1)
	pass
    
    
    print Xtrain
    print Xtest
    #aucMinimize(ensemble,Xtrain,Xtest,y,test_indices,takeMean=False)
    classicalBlend(ensemble,Xtrain,Xtest,y,test_indices)


def classicalBlend(ensemble,oobpreds,testset,ly,test_indices):
    #blending
    folds=8
    #do another crossvalidation for weights
    #blender=LogisticRegression(penalty='l2', tol=0.0001, C=1.0)
    #blender = Pipeline([('filter', SelectPercentile(f_regression, percentile=90)), ('model', LogisticRegression(penalty='l2', tol=0.0001, C=1.0))])
    #blender=SGDClassifier(alpha=.001, n_iter=50,penalty='l2',shuffle=True,random_state=42,loss='log')
    blender=AdaBoostClassifier(learning_rate=0.1,n_estimators=100)
    #blender=RandomForestClassifier(n_estimators=500,n_jobs=1, max_features='auto',oob_score=False,random_state=42)
    #blender=ExtraTreesClassifier(n_estimators=50,max_depth=None,min_samples_leaf=10,n_jobs=1,criterion='entropy', max_features='auto',oob_score=False,random_state=42)
    #blender=ExtraTreesRegressor(n_estimators=50,max_depth=None)
    cv = KFold(oobpreds.shape[0], n_folds=folds, indices=True,random_state=123)
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
      for i,model in enumerate(oobpreds.columns):
	fpr, tpr, thresholds = metrics.roc_curve(ly, oobpreds.iloc[:,i])
	auc=metrics.auc(fpr, tpr)
	print "%-32s  auc: %4.3f coef: %4.3f" %(model,auc,blender.coef_[0][i])
      print "Sum: %4.4f"%(np.sum(blender.coef_))
    #plt.plot(range(len(ensemble)),scores,'ro')
    
    #Prediction
    print "Make final ensemble prediction..."
    #make prediction for each classifiers
    preds=blender.fit(oobpreds,ly)
    #blend results
    preds=blender.predict_proba(testset)[:,1]   
    preds=pd.DataFrame(preds,columns=["label"],index=test_indices)
    preds.to_csv('../stumbled_upon/submissions/sub1310a.csv')
    print preds


def aucMinimize(ensemble,Xtrain,Xtest,y,test_indices,takeMean=False):
    #http://www.kaggle.com/c/amazon-employee-access-challenge/forums/t/4928/combining-the-results-of-various-models?page=3
    def fopt(params):
	# nxm  * m*1 ->n*1
	auc=roc_auc_score(y, np.dot(Xtrain,params))
	#print "auc:",auc
	return -auc
   
    constr=[lambda x,z=i: x[z] for i in range(len(ensemble))]

    n_models=len(ensemble)
    x0 = np.ones((n_models, 1)) / n_models
    #xopt = fmin(fopt, x0)
    xopt = fmin_cobyla(fopt, x0,constr,rhoend=1e-8)
    #normalize, not necessary for auc
    xopt=xopt/np.sum(xopt)
    
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
	fpr, tpr, thresholds = metrics.roc_curve(y, Xtrain.iloc[:,i])
	auc=metrics.auc(fpr, tpr)
	print "%-16s   auc: %4.3f  coef: %4.4f" %(model,auc,xopt[i])
    print "Sum: %4.4f"%(np.sum(xopt))
    #prediction
    preds=pd.DataFrame(np.dot(Xtest,xopt),columns=["label"],index=test_indices)
    preds.to_csv('../stumbled_upon/submissions/sub1310b.csv')
    print preds
    print preds.describe()
  


if __name__=="__main__":
    #ensemble,y=createModels()
    #ensemble=createOOBdata(ensemble,y,20)
    trainEnsemble(addMetaFeatures=True)
    