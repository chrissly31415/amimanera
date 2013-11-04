#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""  Chrissly31415
August,September 2013

"""

from stumble import *
import itertools
from scipy.optimize import fmin,fmin_cobyla
from random import randint

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
    
    #logistic regression, tagword features
    #(Xs,y,Xs_test,test_indices,train_indices) = prepareDatasets('test',useSVD=0,useJson=True,usePosTag=False,usewordtagSmoothing=False,usetagwordSmoothing=True)  
    #model = LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1.0, class_weight=None)
    #xmodel = XModel("logreg_tagword",model,Xs,Xs_test)
    #ensemble.append(xmodel)
    
    #logistic regression, wordtag features
    #(Xs,y,Xs_test,test_indices,train_indices) = prepareDatasets('test',useSVD=0,useJson=True,usePosTag=False,usewordtagSmoothing=True,usetagwordSmoothing=False)  
    #model = LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1.0, class_weight=None)
    #xmodel = XModel("logreg_wordtag",model,Xs,Xs_test)
    #ensemble.append(xmodel)
    
    #(Xs,y,Xs_test,test_indices,train_indices) = prepareDatasets('test',useSVD=0,useJson=True,usePosTag=True,usewordtagSmoothing=False,usetagwordSmoothing=False)
    #model = Pipeline([('filter', SelectPercentile(chi2, percentile=50)), ('model', LogisticRegression(penalty='l2', tol=0.0001, C=10.0))])
    #xmodel = XModel("logreg_postag",model,Xs,Xs_test)
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
    
    #rf, using SVD=2000
    #(X,y,X_test,test_indices,train_indices) = prepareDatasets('tfidfV',useSVD=1000,useJson=True,useHTMLtag=False,useAddFeatures=False,usePosTag=False,useAlcat=False,useGreedyFilter=False)#opt SVD=50
    #model = Pipeline([('filter', SelectPercentile(f_classif, percentile=5)), ('model', RandomForestClassifier(n_estimators=1000,max_depth=None,min_samples_leaf=5,n_jobs=4,criterion='gini', max_features='auto',oob_score=False))])
    #xmodel = XModel("randomf_1000SVD",model,X,X_test)
    #ensemble.append(xmodel)
    
    
    #KNN fastest model 0.870
    #(X,y,X_test,test_indices,train_indices) = prepareDatasets('tfidfV',useSVD=50,useJson=True,useHTMLtag=True,useAddFeatures=True,usePosTag=True,useAlcat=True,useGreedyFilter=True)  
    #model= Pipeline([('filter', SelectPercentile(f_classif, percentile=15)), ('model', KNeighborsClassifier(n_neighbors=150))])
    #xmodel = XModel("knn1",model,X,X_test)
    #ensemble.append(xmodel)
    
    #xrf, using SVD 0.883
    #(X,y,X_test,test_indices,train_indices) = prepareDatasets('tfidfV',useSVD=100,useJson=True,useHTMLtag=True,useAddFeatures=True,usePosTag=True,useAlcat=False,useGreedyFilter=True)
    #model = ExtraTreesClassifier(n_estimators=600,max_depth=None,min_samples_leaf=10,n_jobs=4,criterion='entropy', max_features=20,oob_score=False)
    #xmodel = XModel("extrarf2",model,X,X_test)
    #ensemble.append(xmodel)
    
    #gradient boosting, using SVD 0.883, feature have been reduced to 60
    #(X,y,X_test,test_indices,train_indices) = prepareDatasets('tfidfV',useSVD=100,useJson=True,useHTMLtag=True,useAddFeatures=True,usePosTag=True,useAlcat=False,useGreedyFilter=True,loadTemp=True)
    #model  = GradientBoostingClassifier(loss='deviance', learning_rate=0.01, n_estimators=500, subsample=0.5, min_samples_split=6, min_samples_leaf=10, max_depth=5, init=None, random_state=123,verbose=False)
    #xmodel = XModel("gradboost2",model,X,X_test)
    #ensemble.append(xmodel)
    
    #ADAboost
    (X,y,X_test,test_indices,train_indices) = prepareDatasets('hV',useSVD=10,useJson=False,useHTMLtag=False,useAddFeatures=False,usePosTag=False,useAlcat=False,useGreedyFilter=False,char_ngram=1,loadTemp=True)
    model = Pipeline([('filter', SelectPercentile(f_classif, percentile=80)), ('model', AdaBoostClassifier(n_estimators=200,learning_rate=0.1))])
    xmodel = XModel("ada",model,X,X_test)
    ensemble.append(xmodel)
    
    
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
   
def trainEnsemble(ensemble=None,useCols=None,addMetaFeatures=False):
    test_indices = pd.read_csv('../stumbled_upon/data/test.tsv', sep="\t", na_values=['?'], index_col=1).index
    y = np.asarray(pd.read_csv('../stumbled_upon/data/train.tsv', sep="\t", na_values=['?'], index_col=1)['label'])
    
    for i,model in enumerate(ensemble):
	print "Loading model:",i," name:",model
	if i>0:
	    X = pd.read_csv("../stumbled_upon/data/"+model+".csv", sep=",", index_col=0)
	    X.columns=[model]
	    Xall=pd.concat([Xall,X], axis=1)
	else:
	    Xall = pd.read_csv("../stumbled_upon/data/"+model+".csv", sep=",", index_col=0)
	    Xall.columns=[model]
    
    #normalize
    #Xall = (Xall - Xall.min()) / (Xall.max() - Xall.min())
    
    Xtrain=Xall[len(test_indices):]
    Xtest=Xall[:len(test_indices)]
    #if we add metafeature we should not use aucMinimize...
    if addMetaFeatures:
	multiply=True
	#(Xs,y,Xs_test,test_indices,train_indices) = prepareDatasets('tfidfV',useSVD=2,useJson=True,useHTMLtag=True,useAddFeatures=True,usePosTag=True,useAlcat=True,useGreedyFilter=True)
	#Xs = pd.read_csv('../stumbled_upon/data/Xens.csv', sep=",", index_col=0)
	#Xs_test = pd.read_csv('../stumbled_upon/data/Xens_test.csv', sep=",", index_col=0)
	Xs = pd.read_csv('../stumbled_upon/data/Xlarge.csv', sep=",", index_col=0)
	Xs_test = pd.read_csv('../stumbled_upon/data/Xlarge_test.csv', sep=",", index_col=0)
	
	
	#append lof
	tmp=pd.read_csv('../stumbled_upon/data/lof.csv', sep=",", index_col=0)

	Xs = pd.concat([Xs,tmp[len(test_indices):]], axis=1)
	Xs_test = pd.concat([Xs_test,tmp[:len(test_indices)]], axis=1)

	Xs=Xs.loc[:,useCols]
	Xs_test=Xs_test.loc[:,useCols]
	
	print "Original shape:",Xtrain.shape
	
	if multiply:
	    n=len(Xtrain.columns)
	    for i,col in enumerate(useCols):
		newcolnames=[]
		for name in Xtrain.columns[0:n]:
		    tmp=name+"X"+col
		    newcolnames.append(tmp)
		newcolnames= Xtrain.columns.append(pd.Index(newcolnames))
		#print type(Xs.ix[:,i])
		#Xtrain=pd.concat([Xtrain,Xs], axis=1)	    
		Xtrain=pd.concat([Xtrain,Xtrain.ix[:,0:n].mul(Xs.ix[:,i],axis=0)], axis=1)
		Xtrain.columns=newcolnames
		Xtest=pd.concat([Xtest,Xtest.ix[:,0:n].mul(Xs_test.ix[:,i],axis=0)], axis=1)
		Xtest.columns=newcolnames
		
	else:
	    Xtrain=pd.concat([Xtrain,Xs], axis=1)
	    Xtest=pd.concat([Xtest,Xs_test], axis=1)
	    
    
    #scale data
    X_all=pd.concat([Xtest,Xtrain])
    X_all=removeCorrelations(X_all,0.992)
    
    
    X_all = (X_all - X_all.min()) / (X_all.max() - X_all.min())
    Xtrain=X_all[len(test_indices):]
    Xtest=X_all[:len(test_indices)]
    #print Xtrain
    print "New shape",Xtrain.shape
    #print Xtrain
    #print Xtest
    #Xtrain,Xtest = aucMinimize(ensemble,Xtrain,Xtest,y,test_indices,takeMean=False,removeZeroModels=0.0001)
    #Xtrain,Xtest = aucMinimize(ensemble,Xtrain,Xtest,y,test_indices,takeMean=False,removeZeroModels=0.0001)
    #auc=aucMinimize(ensemble,Xtrain,Xtest,y,test_indices,takeMean=False)
    auc=classicalBlend(ensemble,Xtrain,Xtest,y,test_indices)
    return(auc)


def removeCorrelations(X_all,threshhold):
    #filter correlated data
    print "Removing correlated columns with threshhold:",threshhold
    c = X_all.corr().abs()
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


def classicalBlend(ensemble,oobpreds,testset,ly,test_indices):
    #blending
    folds=8
    #do another crossvalidation for weights
    blender=LogisticRegression(penalty='l2', tol=0.0001, C=1.0)
    #blender = Pipeline([('filter', SelectPercentile(f_regression, percentile=25)), ('model', LogisticRegression(penalty='l2', tol=0.0001, C=0.1))])
    #blender=SGDClassifier(alpha=.01, n_iter=50,penalty='l2',loss='log')
    #blender=AdaBoostClassifier(learning_rate=0.01,n_estimators=200)
    #blender=RandomForestClassifier(n_estimators=100,n_jobs=1, max_features='auto',oob_score=False)
    #blender=ExtraTreesClassifier(n_estimators=500,max_depth=None,min_samples_leaf=5,n_jobs=4,criterion='entropy', max_features='auto',oob_score=False)
    #blender=RandomForestClassifier(n_estimators=500,max_depth=None,min_samples_leaf=10,n_jobs=1,criterion='entropy', max_features=5,oob_score=False)
    #blender=ExtraTreesRegressor(n_estimators=500,max_depth=None)
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
    print " <AUC>: %0.4f (+/- %0.4f)" % (blend_scores.mean(), blend_scores.std()),
    oob_auc=roc_auc_score(ly,blend_oob)
    print " AUC oob after blending: %0.4f" %(oob_auc)
    if hasattr(blender,'coef_'):
      print blender.coef_
      for i,model in enumerate(oobpreds.columns):
	auc=roc_auc_score(ly, oobpreds.iloc[:,i])
	print "%-64s  auc: %4.3f coef: %4.3f" %(model,auc,blender.coef_[0][i])
      print "Sum: %4.4f"%(np.sum(blender.coef_))
    #plt.plot(range(len(ensemble)),scores,'ro')
    
    #Prediction
    print "Make final ensemble prediction..."
    #make prediction for each classifiers
    preds=blender.fit(oobpreds,ly)
    #blend results
    preds=blender.predict_proba(testset)[:,1]   
    preds=pd.DataFrame(preds,columns=["label"],index=test_indices)
    preds.to_csv('../stumbled_upon/submissions/sub0111a.csv')
    print preds
    return(oob_auc)


def aucMinimize(ensemble,Xtrain,Xtest,y,test_indices,takeMean=False,removeZeroModels=0.0):
    #adapted from http://www.kaggle.com/c/amazon-employee-access-challenge/forums/t/4928/combining-the-results-of-various-models?page=3
    oob_auc=0.0
    def fopt(params):
	# nxm  * m*1 ->n*1
	if np.isnan(np.sum(params)):
	    print "We have NaN here!!"
	    auc=0.0
	else:
	    auc=roc_auc_score(y, np.dot(Xtrain,params))
	#print "auc:",auc
	return -auc
   
    #we need constraints to avoid overfitting...
    #constr=[lambda x,z=i: x[z] for i in range(len(Xtrain.columns))]
    constr=[lambda x,z=i: x[z] for i in range(len(Xtrain.columns))]
    #constr2=[lambda x,z=i: 0.3-x[z] for i in range(len(Xtrain.columns))]
    #constr=constr+constr2

    n_models=len(Xtrain.columns)
    x0 = np.ones((n_models, 1)) / n_models
    #x0= np.random.random_sample((n_models,1))
    #xopt = fmin(fopt, x0)
    xopt = fmin_cobyla(fopt, x0,constr,rhoend=1e-12)
    #normalize, not necessary for auc
    xopt=xopt/np.sum(xopt)
    if np.isnan(np.sum(xopt)):
	    print "We have NaN here!!"
    
    
    if takeMean==True:
	xopt=x0
	auc=roc_auc_score(y, np.dot(Xtrain,xopt))
	print "->AUC,opt: %4.4f" %(auc)
	
    else:
	auc=roc_auc_score(y, np.dot(Xtrain,x0))
	print "->AUC,mean: %4.4f" %(auc)
	oob_auc=roc_auc_score(y, np.dot(Xtrain,xopt))
	print "->AUC,opt: %4.4f" %(oob_auc)
    
    zero_models=[]
    for i,model in enumerate(Xtrain.columns):
	auc = roc_auc_score(y, Xtrain.iloc[:,i])
	print "%-48s   auc: %4.3f  coef: %4.4f" %(model,auc,xopt[i])
	if xopt[i]<removeZeroModels:
	    zero_models.append(model)
    print "Sum: %4.4f"%(np.sum(xopt))
    if removeZeroModels>0.0:
	print "Dropping ",len(zero_models)," columns:",zero_models
	Xtrain=Xtrain.drop(zero_models,axis=1)
	Xtest=Xtest.drop(zero_models,axis=1)
	return (Xtrain,Xtest)
    
    #prediction
    preds=pd.DataFrame(np.dot(Xtest,xopt),columns=["label"],index=test_indices)
    preds = (preds - preds.min()) / (preds.max() - preds.min())
    preds.to_csv('../stumbled_upon/submissions/sub3110b.csv')
    print preds.describe()
    return(oob_auc)
  
def selectModels(): 
    #ensemble=["logreg_postag","lgblend","logreg1","logreg2_cv","logreg_wordtag","logreg_tagword","sgd1","sgd2","naiveB1","randomf1","randomf2","randomf3","randomf_1000SVD","extrarf1","extrarf2","gradboost1","gradboost2","gbmR","gbmR2","gbmR4","rfR","lr_char22_hV","lr_char33_hV","lr_char44_hV","lr_char55_hV","lr_char11","lr_char22","lr_char33","lr_char44","lr_char55"]
    ensemble=["logreg_postag","logreg_tagword","naiveB1","extrarf2","randomf2","gradboost2","lr_char33_hV","lr_char33","gbmR","gbmR4","rfR","lr_char55_hV","lr_char55"]
    useCols=[u'url_contains_foodstuff', u'CNJ', u'linkwordscore', u'non_markup_alphanum_characters', u'frameTagRatio', u'P', u'DET', u'logn_newline', u'char0', u'url_length', u'body_length', u'avglinksize', u'compression_ratio', u'ADJ', u'char1', u'char4', u'char8']
    #useCols=['lof','compression_ratio','image_ratio','logn_newline','avglinksize','wwwfacebook_ratio','url_contains_recipe','n_comment','spelling_errors_ratio','linkwordscore']#0.8880
    #ensemble=['logreg_postag', 'logreg_tagword', 'naiveB1', 'extrarf2', 'gradboost2', 'lr_char33', 'gbmR', 'gbmR4', 'rfR', 'lr_char55_hV']#TOP 0.8882
    #useCols=['lof', 'compression_ratio', 'image_ratio', 'logn_newline', 'avglinksize', 'wwwfacebook_ratio', 'linkwordscore']#TOP 0.8882
    #useCols=['lof','compression_ratio','image_ratio','logn_newline','avglinksize','spelling_errors_ratio','n_comment']
    #useCols=['lof','compression_ratio','n_comment','logn_newline','spelling_errors_ratio'] 
    
    randBinList = lambda n: [randint(0,1) for b in range(1,n+1)]
    auc_list=[]
    ens_list=[]
    cols_list=[]
    for i in range(5000):
	actlist=randBinList(len(ensemble))
	actensemble=[x for x in itertools.compress(ensemble,actlist)]
	
	actlist=randBinList(len(useCols))
	actCols=[x for x in itertools.compress(useCols,actlist)]
	
	#print actensemble
	auc=trainEnsemble(actensemble,actCols,addMetaFeatures=True)
	auc_list.append(auc)
	ens_list.append(actensemble)
	cols_list.append(actCols)
    maxauc=0.0
    topens=None
    topcols=None
    for ens,auc,col in zip(ens_list,auc_list,cols_list):
	print ens
	print "AUC: %4.4f" %(auc)
	if auc>maxauc:
	  maxauc=auc
	  topens=ens
	  topcols=col
    print "TOP ensemble:",topens
    print "TOP cols",topcols
    print "TOP auc: %4.4f" %(maxauc)

if __name__=="__main__":
    np.random.seed(123)
    #ensemble,y=createModels()
    #ensemble=createOOBdata(ensemble,y,10)
    #ensemble=["logreg1","sgd1","randomf1","randomf2","randomf3","gradboost1","gbmR"]
    #ensemble=["logreg1","logreg2_cv","sgd1","sgd2","randomf1","randomf2","gradboost1","extrarf1","gbmR","lof"]
    #ensemble=["logreg1","logreg2_cv","sgd2","randomf1","randomf2","gradboost1","gbmR"]#best so far 0.885
    #ensemble=["logreg_postag","rfR","gradboost2","gbmR","gbmR2","gbmR4","randomf2","randomf1","randomf3","randomf_1000SVD","lgblend","logreg1","logreg2_cv","logreg_wordtag","logreg_tagword","sgd1","sgd2","naiveB1","extrarf1","extrarf2","gradboost1","lr_char22_hV","lr_char33_hV","lr_char44_hV","lr_char55_hV","lr_char11","lr_char22","lr_char33","lr_char44","lr_char55","nnet_old"]
    #ensemble=["ada","knn1","logreg_postag","rfR","gradboost2","gbmR","gbmR2","randomf2","randomf1","randomf3","lgblend","logreg1","logreg2_cv","logreg_tagword","naiveB1","extrarf1","gradboost1","lr_char22_hV","lr_char33_hV","lr_char44_hV","lr_char11","lr_char22","lr_char33","lr_char44","lr_char55","gbmR4"]
    #ensemble=["logreg_postag","logreg1"]
    #ensemble=["logreg_postag","logreg_tagword","naiveB1","extrarf2","randomf2","gradboost2","lr_char33_hV","lr_char33","gbmR","gbmR4","rfR",'nnet_old']#AUC=0.8875
    #ensemble=["randomf2","gradboost2","rfR"]
    #ensemble=["logreg1","gradboost2","gbmR","randomf2","nnet_old","extrarf2","randomf_1000SVD",'naiveB1','knn1']
    #ensemble=["gbmR4","rfR"]
    #ensemble=[ 'gradboost2', 'naiveB1', 'extrarf2','lr_char33', 'gbmR', 'gbmR4', 'rfR','logreg_tagword','logreg_postag', 'lr_char55_hV','nnet_old']#TOP 0.8882
    ensemble=["randomf2",'nnet_old','logreg_postag', 'logreg_tagword', 'naiveB1', 'extrarf2', 'gradboost2', 'lr_char33', 'gbmR', 'gbmR4', 'rfR', 'lr_char55_hV','naiveB1']
    #ensemble=['nnet','gradboost2']
    #ensemble=["randomf1","randomf2","gradboost2","randomf_1000SVD"]
    useCols=['lof', 'compression_ratio', 'image_ratio', 'logn_newline', 'avglinksize', 'wwwfacebook_ratio', 'linkwordscore']#TOP 0.8882
    #useCols=[u'url_contains_foodstuff', u'CNJ', u'linkwordscore', u'non_markup_alphanum_characters', u'frameTagRatio', u'P', u'DET', u'logn_newline', u'char0', u'url_length', u'body_length', u'avglinksize', u'compression_ratio', u'ADJ', u'char1', u'char4', u'char8']
    #useCols=['lof','compression_ratio','image_ratio','logn_newline','avglinksize','wwwfacebook_ratio','url_contains_recipe','n_comment','spelling_errors_ratio','linkwordscore']#0.8880
    #ensemble=['logreg_postag', 'logreg_tagword', 'naiveB1', 'extrarf2', 'gradboost2', 'lr_char33', 'gbmR', 'gbmR4', 'rfR', 'lr_char55_hV']#TOP 0.8882
    trainEnsemble(ensemble,useCols,addMetaFeatures=True)
    #selectModels()
    