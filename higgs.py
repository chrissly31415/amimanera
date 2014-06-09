#!/usr/bin/env python 
# coding: utf-8
import numpy as np
import pandas as pd
import sklearn as sl
import random
import math

from qsprLib import *

from pandas.tools.plotting import scatter_matrix
from sklearn.cross_validation import StratifiedKFold


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
    print "Sum (s): %8.2f n(s): %6d"%(sumSWeights,np.sum(sSelector==True))
    print "Sum (b): %8.2f n(b): %6d"%(sumBWeights,np.sum(bSelector==True))
    
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

                          
                          
def AMS_metric(truth,pred,weights,wfactor,cutoff):
    """
    AMS machinery
    """
    sSelector = truth==1
    bSelector = pred==0
    
    smax = np.sum(weights[sSelector])
    
    tpr=0.0
    fpr=0.0
    
    for j,row in enumerate(pred):
	    if row>=cutoff:		
		if truth[j]>=cutoff:
		    tpr=tpr+weights[j]
		else:
		    fpr=fpr+weights[j]
	    #print "Prob: %4.2f Pred: %4.2f  T: %4.2f" %(row,p[i],truth.iloc[i])
	
    print "Sum signals(normalized)   : %6.3f"%(tpr*wfactor)
    print "Sum background(normalized): %6.3f"%(fpr*wfactor)
    
    ams_max=AMS(smax, 0.0,wfactor)
    ams=AMS(tpr, fpr,wfactor)
    print 'AMS = %6.3f [AMS_max = %6.3f]'%(ams,ams_max)
    
    return ams
	

def def amsXvalidation2(lmodel,lX,ly,lw,nfolds=5,cutoff=0.5,proba=True,useWeights=False)
    
    
    ams_scorer = make_scorer(AMS_metric,lw)
    
    
    
	
	
def amsXvalidation(lmodel,lX,ly,lw,nfolds=5,cutoff=0.5,proba=True,useWeights=False):
    """
    Carries out crossvalidation using AMS metrics
    """
    ntotal = 250000
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
	    wtrain_fit = useWeights*wtrain/wsum
	    print "wsum:",np.sum(wtrain_fit)
	    lmodel.fit(lX.iloc[train],lytrain,sample_weight=wtrain_fit)
	
	#training data
	if proba:
	    yinbag=lmodel.predict_proba(lX.iloc[train])[:,1]
	    scores_train[i]=roc_auc_score(lytrain,yinbag)
	else: 
	    yinbag=lmodel.predict(lX.iloc[train])
	    
	wFactor_train=1.* ntotal / train.shape[0]	
	ams_scores_train[i]=AMS_metric(lytrain,yinbag,wtrain,wFactor_train,cutoff)
	print "Training AUC=%6.3f AMS=%6.3f" % (scores_train[i],ams_scores_train[i])
	
	#test
	truth=np.asarray(ly.iloc[test])
	weightsTest=np.asarray(lw.iloc[test])
	
	if proba:
	    yoob=lmodel.predict_proba(lX.iloc[test])[:,1]
	    scores[i]=roc_auc_score(truth,yoob)    
	else:
	    yoob=lmodel.predict(lX.iloc[test])
	
	#compute AUC	#print wFactor	
	#compute AMS	
	wFactor = 1.* ntotal / test.shape[0]
	ams_scores[i]=AMS_metric(wFactor,truth,yoob,weightsTest,cutoff)
	
	vfunc = np.vectorize(binarizeProbs)
	yoob_binary = vfunc(yoob,0.5)
	print classification_report(truth, yoob_binary,target_names=['s','b'])
	print "Iteration=%d %d/%d AUC=%6.3f AMS=%6.3f\n" % (i+1,train.shape[0], test.shape[0],scores[i],ams_scores[i])
	
	
    print "\n##SUMMARY##"
    print " <AUC>: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std())
    print " <AMS>: %0.3f (+/- %0.3f)" % (ams_scores.mean(), ams_scores.std())
    print " <AUC,train>: %0.3f (+/- %0.3f)" % (scores_train.mean(), scores_train.std())
    print " <AMS,train>: %0.3f (+/- %0.3f)" % (ams_scores_train.mean(), ams_scores_train.std())
    
	
def makePredictions(lmodel,lXs_test,filename):
    """
    Uses priorily fit model to make predictions
    """
    
    #rows = random.sample(lXs_test.index, 100)
    #lXs_test = lXs_test.ix[rows]
    
    print "Final test dataframe:",lXs_test.shape
    
    print "Predicting..."
    probs = lmodel.predict_proba(lXs_test)[:,1]    
    print "Class predictions..."
    vfunc = np.vectorize(makeLabels)
    preds = vfunc(probs,0.5)
    
    print "Rank order..."
    idx_sorted = np.argsort(probs)
    ro = np.arange(lXs_test.shape[0])   
    #d = {'EventId': lXs_test.index[idx_sorted], 'RankOrder': ro,'Probs': probs[idx_sorted], 'class': preds[idx_sorted]}
    d = {'EventId': lXs_test.index[idx_sorted], 'RankOrder': ro, 'class': preds[idx_sorted]}
    
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

 
if __name__=="__main__":
    """
    Main class
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
    Xtrain,ytrain,wtrain,Xtest=prepareDatasets(nsamples,onlyPRI,replaceNA,plotting,stats,transform)
    #param=np.logspace(-8, -1, num=8, base=2.0)#SDG
    param=[1.0]
    for i,p in enumerate(param):
	print "i: %d p: %f"%(i,p)
	#print Xtrain.describe()
	#model = RidgeClassifier(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, class_weight=None, solver='auto')
	#model = LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1.0, fit_intercept=True, class_weight='auto')
	#model = SGDClassifier(alpha=p,n_iter=150,shuffle=True,loss='log',penalty='l2',n_jobs=4)#SW,correct implementation?
	#model = BernoulliNB(alpha=.1, binarize=0.0, fit_prior=True, class_prior=None)#SW
	#model = GradientBoostingClassifier()
	#model = GradientBoostingClassifier(loss='deviance', learning_rate=0.01, n_estimators=1000, subsample=p, max_depth=20, max_features='auto',init=None,verbose=False)
	#model = GradientBoostingClassifier(loss='exponential',n_estimators=150, learning_rate=0.1, max_depth=3,subsample=0.8,verbose=None)
	#model = pyGridSearch(model,Xtrain,ytrain)
	#model =GaussianNB()
	#model = KNeighborsClassifier(n_neighbors=5)
	model=   RandomForestClassifier(n_estimators=500,max_depth=None,min_samples_leaf=5,n_jobs=4,criterion='entropy', max_features=5,oob_score=False)#SW
	
	#model = ExtraTreesClassifier(n_estimators=p,max_depth=None,min_samples_leaf=5,n_jobs=4,criterion='entropy', max_features=5,oob_score=False)#opt
	amsXvalidation2(model,Xtrain,ytrain,wtrain,nfolds=nfolds,cutoff=0.5,proba=True,useWeights=1.0)
	print model
	#buildModel(model,Xtrain,ytrain,wtrain)

    #makePredictions(model,Xtest,"submission.csv")
	
    print("Model building done on %d samples in %fs" % (Xtrain.shape[0],time() - t0))
    plt.show()