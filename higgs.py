#!/usr/bin/env python 
# coding: utf-8
import numpy as np
import pandas as pd
import sklearn as sl
import random
import math

from qsprLib import *
import inspect

from pandas.tools.plotting import scatter_matrix
from sklearn.datasets import load_digits


def prepareDatasets(nsamples=-1,onlyPRI=False,replaceNA=True,plotting=True,stats=True,transform=False):
    """
    Read in train and test data, create data matrix X and targets y   
    """
    X = pd.read_csv('../datamining-kaggle/higgs/training.csv', sep=",", na_values=['?'], index_col=0)
    X_test = pd.read_csv('../datamining-kaggle/higgs/test.csv', sep=",", na_values=['?'], index_col=0)
    
    
    if nsamples != -1: 
	rows = random.sample(X.index, nsamples)
	X = X.ix[rows]      
    
    weights = X['Weight']
    
    sSelector = X['Label']=='s'
    bSelector = X['Label']=='b'
    
    s = np.sum(weights[sSelector])  
    #b = np.sum(weights[bSelector])
    
    sumWeights = np.sum(weights)
    print "Sum weights: %8.2f"%(sumWeights)
    sumSWeights = np.sum(weights[sSelector])
    sumBWeights = np.sum(weights[bSelector])
    print "Sum (w_s): %8.2f n(s): %6d"%(sumSWeights,np.sum(sSelector==True))
    print "Sum (w_b): %8.2f n(b): %6d"%(sumBWeights,np.sum(bSelector==True))
    
    ntotal=250000
    wFactor = 1.* ntotal / X.shape[0]
    print "AMS,max: %4.3f (wfactor=%4.3f)" % (AMS(s, 0.0,wFactor),wFactor)
    
    y = X['Label'].str.replace(r's','1').str.replace(r'b','0')
    y = y.astype(float)
    	  
    X = X.drop(['Weight'], axis=1)
    X = X.drop(['Label'], axis=1)
   
    #modifications for ALL DATA
    X_all = pd.concat([X_test, X])
    
    if onlyPRI:
	cols = X_all.columns
	for col in cols:
	    if col.startswith('DER'):
		X_all = X_all.drop([col], axis=1)
		
    if replaceNA:
	X_all = X_all.replace(-999, np.NaN)
	X_all = X_all.fillna(X_all.mean())   
       
    transcols_PRI=['PRI_met','PRI_lep_pt','PRI_met_sumet','PRI_jet_all_pt','PRI_jet_leading_pt','PRI_jet_subleading_pt','PRI_tau_pt']
    transcols_DER=['DER_mass_transverse_met_lep','DER_mass_vis','DER_pt_h','DER_pt_ratio_lep_tau','DER_pt_tot','DER_sum_pt']
    if not onlyPRI:
	transcols_PRI=transcols_PRI+transcols_DER
    
    if transform:
	for col in transcols_PRI:
	    X_all[col]=X_all[col]-X_all[col].min()+1.0
	    X_all[col]=X_all[col].apply(np.log)
       
    if stats:
	for col in X_all.columns:
	    print col
	    print X_all[col].describe()
	    
	print "max observations:" 
	print X_all.apply(lambda x: x.idxmax())
	
	print "min observations:"
	print X_all.apply(lambda x: x.idxmin())
    
    
    if plotting:
	#X.boxplot(by='Label')
	#scatter_matrix(X, alpha=0.2, figsize=(6, 6))
	
	X_all.hist(color='b', alpha=0.5, bins=50)
    
    #print "PRESS A KEY..."
    #raw_input()
    #split data again
    X = X_all[len(X_test.index):]
    X_test = X_all[:len(X_test.index)]
    
    return (X,y,weights,X_test)

    
def AMS(s,b,factor):
    assert s >= 0
    assert b >= 0
    s = s * factor
    b = b * factor
    bReg = 10.
    return math.sqrt(2 * ((s + b + bReg) * 
                          math.log(1 + s / (b + bReg)) - s))


	
def amsXvalidation(lmodel,lX,ly,lw,nfolds=5,cutoff=0.5,useProba=True,useWeights=True,useRegressor=False):
    """
    Carries out crossvalidation using AMS metrics
    """
    ntotal = 250000
    vfunc = np.vectorize(binarizeProbs)
    #print sSelector
    #print bSelector
   #cv = StratifiedKFold(ly, nfolds)
    cv = KFold(lX.shape[0], n_folds=nfolds,shuffle=True)
    scores=np.zeros(nfolds)
    ams_scores=np.zeros(nfolds)
    scores_train=np.zeros(nfolds)
    ams_scores_train=np.zeros(nfolds)
    for i, (train, test) in enumerate(cv):	
	#train
	lytrain = np.asarray(ly.iloc[train])
	wtrain= np.asarray(lw.iloc[train])
	if useWeights is False:
	    lmodel.fit(lX.iloc[train],lytrain)
	else:
	    #scale wtrain
	    wsum = np.sum(wtrain)
	    wtrain_fit = useWeights*wtrain/wsum*Xtrain.shape[0]
	    #print "Weights have been normalized:",np.sum(wtrain_fit)
	    lmodel.fit(lX.iloc[train],lytrain,sample_weight=wtrain_fit)
	    #lmodel.fit(lX.iloc[train],lytrain)
	
	#training data
	sc_string='AUC'
	if useProba:
	    yinbag=lmodel.predict_proba(lX.iloc[train])
	    scores_train[i]=roc_auc_score(lytrain,yinbag[:,1])	    
	else: 
	    yinbag=lmodel.predict(lX.iloc[train])
	    if useRegressor:
		yinbag = vfunc(yinbag,cutoff)
	    scores_train[i]=f1_score(lytrain,yinbag)
	    sc_string='F1-SCORE'
	    
	#wfactor=1.* ntotal / train.shape[0]
	
	ams_scores_train[i]=AMS_metric(lytrain,yinbag,sample_weight=wtrain,needs_proba=useProba,use_proba=useProba,cutoff=cutoff)
	print "Training %8s=%6.3f AMS=%6.3f" % (sc_string,scores_train[i],ams_scores_train[i])
	
	#test
	truth=np.asarray(ly.iloc[test])
	weightsTest=np.asarray(lw.iloc[test])
	
	if useProba:
	    yoob=lmodel.predict_proba(lX.iloc[test])
	    scores[i]=roc_auc_score(truth,yoob[:,1]) 
	    sSelector = truth==1
	    bSelector = truth==0
	    plt.hist(yoob[:,1][sSelector],bins=50)
	    plt.hist(yoob[:,1][bSelector],bins=50)
	    plt.show()
	    
	else:
	    yoob=lmodel.predict(lX.iloc[test])
	    if useRegressor:
		yoob = vfunc(yoob,cutoff)
	    scores[i]=f1_score(truth,yoob)
	
	
	#compute AUC	#print wFactor	
	#compute AMS
	
	ams_scores[i]=AMS_metric(truth,yoob,sample_weight=weightsTest,needs_proba=useProba,use_proba=useProba,cutoff=cutoff)
	
	#if useProba:
	#    yoob_binary = vfunc(yoob,cutoff)
	#print classification_report(truth, yoob_binary,target_names=['s','b'])
	print "Iteration=%d %d/%d AUC=%6.3f AMS=%6.3f\n" % (i+1,train.shape[0], test.shape[0],scores[i],ams_scores[i])
	
	
    print "\n##SUMMARY##"
    print " <%-8s>: %0.3f (+/- %0.3f)" % (sc_string,scores.mean(), scores.std())
    print " <AMS>: %0.3f (+/- %0.3f)" % (ams_scores.mean(), ams_scores.std())
    print " <%-8s,train>: %0.3f (+/- %0.3f)" % (sc_string,scores_train.mean(), scores_train.std())
    print " <AMS,train>: %0.3f (+/- %0.3f)" % (ams_scores_train.mean(), ams_scores_train.std())

  
def AMS_metric(y_true,y_pred,**kwargs):
    """
    Higgs AMS metric
    """   
    if 'sample_weight' in kwargs:
	sample_weight=kwargs['sample_weight']
    else:
	print "We need sample weights for sensible evaluation!"
	return
	
    cutoff=0.5
    ntotal=250000
    tpr=0.0
    fpr=0.0 
    
    #print ("true",y_true)
    #print ("pred",y_pred)
    #plt.hist(y_pred,bins=50)
    #plt.show()
    #print "press a key..."
    #raw_input()
    
    #check if we are dealing with proba
    info="- using 0-1 classification"
    if kwargs['use_proba']:
	if 'cutoff' in kwargs:
	    cutoff=kwargs['cutoff']
	y_pred=y_pred[:,1]
	info="- using probabilities with cutoff=%4.2f"%(cutoff)
        
    for j,row in enumerate(y_pred):
	    if row>=cutoff:		
		if y_true[j]>=cutoff:
		    tpr=tpr+sample_weight[j]
		else:
		    fpr=fpr+sample_weight[j]

    sSelector = y_true==1
    wfactor=1.* ntotal / y_true.shape[0]
    smax = np.sum(sample_weight[sSelector])
    ams_max=AMS(smax, 0.0,wfactor)
    ams=AMS(tpr, fpr,wfactor)
    print 'AMS = %6.3f [AMS_max = %6.3f] %-32s'%(ams,ams_max,info)
    
    return ams   
    
def amsGridsearch(lmodel,lX,ly,lw,fitWithWeights=False,nfolds=5,useProba=False,cutoff=0.5):
    print 
    if not 'sample_weight' in inspect.getargspec(lmodel.fit).args:
	  print("WARNING: Fit function ignores sample_weight!")
	  
    fit_params = {'sample_weight': lw}
    fit_params['fitWithWeights']=fitWithWeights
    #https://github.com/scikit-learn/scikit-learn/issues/3223 + own modifications
    ams_scorer = make_scorer(score_func=AMS_metric,needs_proba=useProba,use_proba=useProba,cutoff=cutoff)
    
    #parameters = {'n_estimators':[150,300], 'max_features':[5,10]}#rf
    #parameters = {'max_depth':[3,5,10], 'learning_rate':[0.5,0.1,0.01,0.05,0.001],'n_estimators':[150,300,500],'subsample':[1.0]}#gbm
    parameters = {'max_depth':[8,6], 'learning_rate':[0.1,0.05],'n_estimators':[100,150],'subsample':[1.0],'loss':['deviance'],'min_samples_leaf':[50,100],'max_features':[5]}#gbm
    #parameters = {'max_depth':[10], 'learning_rate':[0.001],'n_estimators':[500],'subsample':[0.5],'loss':['deviance']}#gbm
    #parameters = {'max_depth':[15,20,25], 'learning_rate':[0.1,0.01],'n_estimators':[150,300],'subsample':[1.0,0.5]}#gbm
    #parameters = {'max_depth':[20,30], 'learning_rate':[0.1,0.05],'n_estimators':[300,500,1000],'subsample':[0.5],'loss':['exponential']}#gbm
    #parameters = {'max_depth':[15,20], 'learning_rate':[0.05,0.01,0.005],'n_estimators':[250,500],'subsample':[1.0,0.5]}#gbm
    #parameters = {'n_estimators':[200,500], 'learning_rate':[0.1,0.01,0.001]}#adaboost
    #parameters = {'filter__percentile':[20,15]}#naives bayes
    #parameters = {'filter__percentile': [15], 'model__alpha':[0.0001,0.001],'model__n_iter':[15,50,100],'model__penalty':['l1']}#SGD
    #parameters['model__n_neighbors']=[40,60]}#knn
    #parameters['model__alpha']=[1.0,0.8,0.5,0.1]#opt nb
    #parameters = {'n_neighbors':[10,30,40,50],'algorithm':['ball_tree'],'weights':['distance']}#knn
    clf_opt=grid_search.GridSearchCV(lmodel, parameters,n_jobs=4,verbose=1,scoring=ams_scorer,cv=nfolds,fit_params=fit_params,refit=True)
    
    clf_opt.fit(lX,ly)
    #dir(clf_opt)
    for params, mean_score, scores in clf_opt.grid_scores_:       
        print("%0.3f (+/- %0.3f) for %r" % (mean_score, scores.std(), params))
    
    #scores = cross_validation.cross_val_score(lmodel, lX, ly, fit_params=fit_params,scoring=ams_scorer,cv=nfolds)
    print "AMS: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std())
    #return(clf_opt.best_estimator_)
	
def makePredictions(lmodel,lXs_test,filename,useProba=True,cutoff=0.5):
    """
    Uses priorily fit model to make predictions
    """
    print "Final test dataframe:",lXs_test.shape
    
    print "Predicting..."
    if useProba:
	print "Using probabilities with cutoff:",cutoff
	probs = lmodel.predict_proba(lXs_test)[:,1]
	vfunc = np.vectorize(makeLabels)
	probs = vfunc(probs,cutoff)
    else:  
	probs = lmodel.predict(lXs_test)
    
    print "Class predictions..."

    print "Rank order..."
    idx_sorted = np.argsort(probs)
    ro = np.arange(lXs_test.shape[0])+1   
    #d = {'EventId': lXs_test.index[idx_sorted], 'RankOrder': ro,'Probs': probs[idx_sorted], 'class': probs[idx_sorted]}
    d = {'EventId': lXs_test.index[idx_sorted], 'RankOrder': ro, 'class': probs[idx_sorted]}
    
    print "Saving predictions to: ",filename
    pred_df = pd.DataFrame(data=d)
    pred_df.to_csv(filename,index=False)    
 
def makeLabels(a,cutoff):
    """
    make lables s=1 and b=0
    """
    if a>cutoff: return 's'
    else: return 'b'

    
def binarizeProbs(a,cutoff):   
    """
    turn probabilities to 1 and 0
    """
    if a>cutoff: return 1.0
    else: return 0.0


def analyzeLearningCurve(model,X,y,lw,folds=8):
    """
    make a learning curve according to http://scikit-learn.org/dev/auto_examples/plot_learning_curve.html
    """
    #digits = load_digits()
    #X, y = digits.data, digits.target
    #plt.hist(y,bins=40)
    #cv = cross_validation.ShuffleSplit(X.shape[0], n_iter=10, test_size=0.2, random_state=0)
    
    #cv = KFold(X.shape[0], n_folds=folds,shuffle=True)  
    cv = StratifiedKFold(y, n_folds=folds)
    #learn_score = make_scorer(roc_auc_score)
    #learn_score = make_scorer(score_func=AMS_metric,needs_proba=useProba,use_proba=useProba)
    learn_score=make_scorer(f1_score)
    plot_learning_curve(model, "learning curve", X, y, ylim=(0.1, 1.01), cv=cv, n_jobs=4,scoring=learn_score)

    
def buildAMSModel(lmodel,lXs,ly,lw=None,fitWithWeights=False,needs_proba=False,useProba=False,cutoff=0.5,feature_names=None):
    """   
    Final model building part
    """ 
    print "Xvalidation..."
    fit_params = {'sample_weight': lw}
    fit_params['fitWithWeights']=fitWithWeights
    ams_scorer = make_scorer(score_func=AMS_metric,needs_proba=needs_proba,use_proba=useProba,cutoff=cutoff)
    
    #how can we avoid that samples are used for fitting
    scores = cross_validation.cross_val_score(lmodel,lXs,ly,fit_params=fit_params, scoring=ams_scorer,cv=8,n_jobs=4)
    print "AMS: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std())
    print "Building model with all instances..."
    
    if fitWithWeights:
	    lmodel.fit(lXs,ly,sample_weight=lw)
    else:
	    lmodel.fit(lXs,ly)
    
    #analyzeModel(lmodel,feature_names)
    return(lmodel)
    

def checksubmission(filename):
    X = pd.read_csv(filename, sep=",", na_values=['?'], index_col=None)
    print X
    print X.describe()
    
    print "Unique IDs:",np.unique(X.EventId).shape[0]
    print "Unique ranks:",np.unique(X.RankOrder).shape[0]
    
    
if __name__=="__main__":
    """
    Main classTrue
    """
    #sample weights
    #http://scikit-learn.org/stable/auto_examples/svm/plot_weighted_samples.html
    #sample weights
    #https://github.com/scikit-learn/scikit-learn/pull/3224
    # Set a seed for consistant results
    # TODO make only class prediction and guess ranking... 
    # TODO http://www.rdkit.org/docs/Cookbook.html parallel stuff
    # TODO http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html#sklearn.metrics.make_scorer
    # TODO http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html
    t0 = time()
    np.random.seed(123)
    print "numpy:",np.__version__
    print "pandas:",pd.__version__
    print "sklearn:",sl.__version__
    #pd.set_option('display.height', 5)
    pd.set_option('display.max_columns', 14)
    pd.set_option('display.max_rows', 40)
    
    nsamples=-1
    nfolds=8
    onlyPRI=False
    replaceNA=False
    plotting=False
    stats=False
    transform=False
    useProba=False  #use probailities for prediction
    useWeights=True #use weights for training
    useRegressor=False
    cutoff=0.5
    subfile="/home/loschen/Desktop/datamining-kaggle/higgs/submissions/sub1806a.csv"
    Xtrain,ytrain,wtrain,Xtest=prepareDatasets(nsamples,onlyPRI,replaceNA,plotting,stats,transform)
    #print Xtrain.describe()
    #model = RidgeClassifier(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, class_weight=None, solver='auto')
    #model = LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1.0, fit_intercept=True, class_weight='auto')
    #model = SGDClassifier(alpha=0.001,n_iter=150,shuffle=True,loss='log',penalty='l1',n_jobs=4)#SW,correct implementation?
    #model = GaussianNB()
    #model = GradientBoostingClassifier(loss='exponential',n_estimators=150,learning_rate=.2,max_depth=6,verbose=2,subsample=1.0)
    #model = GradientBoostingRegressor(n_estimators=150,learning_rate=.1,max_depth=6,verbose=2,subsample=1.0)
    #model = GradientBoostingClassifier(loss='deviance', learning_rate=0.001, n_estimators=500, subsample=1.0, max_depth=10, max_features='auto',init=None,verbose=False)#opt fitting wo weights!!!
    #model = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=150, subsample=0.5, max_depth=25, max_features='auto',init=None,verbose=False)#opt2
    #model = GradientBoostingClassifier(loss='exponential',n_estimators=150, learning_rate=0.2, max_depth=5,subsample=1.0,verbose=2)
    #model = GradientBoostingClassifier(loss='deviance',n_estimators=150, learning_rate=0.05,max_depth=8,min_samples_leaf=50,max_features=5,verbose=False)#with no weights, cutoff=0.85 and useProba=True
    #model = pyGridSearch(model,Xtrain,ytrain)
    #model = Pipeline([('filter', SelectPercentile(f_classif, percentile=15)), ('model', GaussianNB())])
    #model = KNeighborsClassifier(n_neighbors=30)
    #model = AdaBoostClassifier(n_estimators=500,learning_rate=0.1)
    model=   RandomForestClassifier(n_estimators=250,max_depth=None,min_samples_leaf=10,n_jobs=1,criterion='entropy', max_features=5,oob_score=False)#SW-proba?
    #analyzeLearningCurve(model,Xtrain,ytrain,wtrain)
    #model = ExtraTreesClassifier(n_estimators=p,max_depth=None,min_samples_leaf=5,n_jobs=4,criterion='entropy', max_features=5,oob_score=False)#opt
    #model = Pipeline([('filter', SelectPercentile(f_classif, percentile=15)), ('model', model)])
    #model = amsGridsearch(model,Xtrain,ytrain,wtrain,fitWithWeights=useWeights,nfolds=nfolds,useProba=useProba,cutoff=cutoff)
    #amsXvalidation(model,Xtrain,ytrain,wtrain,nfolds=nfolds,cutoff=0.5,useProba=useProba,useWeights=useWeights,useRegressor=useRegressor)
    print model
    model=buildAMSModel(model,Xtrain,ytrain,wtrain,fitWithWeights=useWeights,needs_proba=useProba,useProba=useProba,cutoff=cutoff)
    makePredictions(model,Xtest,subfile,useProba=useProba,cutoff=cutoff)
    checksubmission(subfile)
    print("Model building done on %d samples in %fs" % (Xtrain.shape[0],time() - t0))
    plt.show()