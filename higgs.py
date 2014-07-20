#!/usr/bin/python 
# coding: utf-8
import numpy as np
import pandas as pd
import sklearn as sl
import random
import math

from qsprLib import *
import inspect
import pickle

from pandas.tools.plotting import scatter_matrix
from xgboost_sklearn import *


def createFeatures(X_all):
    """
    Create some new features
    """
    def f(x):
        if x==-999.0:
            return 1.0
        else:
            return 0.0    

    print "Feature creation..."
    X_NA=pd.DataFrame(index=X_all.index)

    for colname in X_all.columns:
        if (X_all[colname]==-999.0).any():
            print colname
            X_NA[colname+'_mod']=X_all[colname].map(f)
            #X_all['hasNA']=X_all[colname].map(f2)

    X_NA['NA_sum'] = X_NA.sum(axis=1)    
   
    X_all = pd.concat([X_all, X_NA['NA_sum']],axis=1)
    print X_all.describe()
    print "End of feature creation..."    
    
    return (X_all)

def prepareDatasets(nsamples=-1,onlyPRI=False,replaceNA=True,plotting=True,stats=True,transform=False,createNAFeats=False,dropCorrelated=True,scale_data=False,clusterFeature=False):
    """
    Read in train and test data, create data matrix X and targets y   
    """
    X = pd.read_csv('../datamining-kaggle/higgs/training.csv', sep=",", na_values=['?'], index_col=0)
    X_test = pd.read_csv('../datamining-kaggle/higgs/test.csv', sep=",", na_values=['?'], index_col=0)
    
    
    if nsamples != -1: 
	rows = random.sample(X.index, nsamples)
	X = X.ix[rows]      
    
    weights = np.asarray(X['Weight'])
    
    sSelector = np.asarray(X['Label']=='s')
    bSelector = np.asarray(X['Label']=='b')
    
    s = np.sum(weights[sSelector])  
    #b = np.sum(weights[bSelector])
    
    sumWeights = np.sum(weights)
    print "Sum weights: %8.2f"%(sumWeights)
    sumSWeights = np.sum(weights[sSelector])
    sumBWeights = np.sum(weights[bSelector])
    print "Sum (w_s): %8.2f n(s): %6d"%(sumSWeights,np.sum(sSelector==True))
    print "Sum (w_b): %8.2f n(b): %6d"%(sumBWeights,np.sum(bSelector==True))
    
    print "Unique weights for signal    :",np.unique(weights[sSelector])
    print "Unique weights for background:",np.unique(weights[bSelector])
    
    ntotal=250000
    wFactor = 1.* ntotal / X.shape[0]
    print "AMS,max: %4.3f (wfactor=%4.3f)" % (AMS(s, 0.0,wFactor),wFactor)
    
    y = X['Label'].str.replace(r's','1').str.replace(r'b','0')
    y = np.asarray(y.astype(float))   	  
        
    X = X.drop(['Weight'], axis=1)
    X = X.drop(['Label'], axis=1)
    
    #modifications for ALL DATA
    X_all = pd.concat([X_test, X])    
    
    correllated=['PRI_jet_subleading_eta','PRI_jet_subleading_phi','PRI_jet_leading_eta','PRI_jet_leading_phi','DER_prodeta_jet_jet','DER_lep_eta_centrality']
    if dropCorrelated:
        #drop correllated features to PRI_jet_subleading_pt
        #drop correllated features to PRI_jet_leading_pt        
        #drop correlated features to DER_deltaeta_jet_jet
        for col in correllated:
            print "Dropping correlated features: ",col
            X_all = X_all.drop([col], axis=1)
    
    if ('DER' or 'PRI') in onlyPRI:
	cols = X_all.columns
	for col in cols:
	    if col.startswith(onlyPRI):
             print "Dropping column: ",col
             X_all = X_all.drop([col], axis=1)

    if createNAFeats:
        X_all=createFeatures(X_all)
		
    if replaceNA:
	X_all = X_all.replace(-999, np.NaN)
	X_all = X_all.fillna(X_all.mean())   
       
    transcols_PRI=['PRI_met','PRI_lep_pt','PRI_met_sumet','PRI_jet_all_pt','PRI_jet_leading_pt','PRI_jet_subleading_pt','PRI_tau_pt']
    transcols_DER=['DER_mass_transverse_met_lep','DER_mass_vis','DER_pt_h','DER_pt_ratio_lep_tau','DER_pt_tot','DER_sum_pt']
    if not onlyPRI:
	transcols_PRI=transcols_PRI+transcols_DER
    
    if transform:
        for col in transcols_PRI:
            if col in X_all.columns:
                print "log transformation of ",col
                X_all[col]=X_all[col]-X_all[col].min()+1.0
                X_all[col]=X_all[col].apply(np.log)
       
    if stats:
        for col in X_all.columns:
            print col
            print X_all[col].describe()
        print X_all.corr()
        #scatter_matrix(X_all, alpha=0.2, figsize=(6, 6), diagonal='hist')
        plt.show()
	    
	print "idx max observations:" 
	print X_all.apply(lambda x: x.idxmax())
	
	print "idx min observations:"
	print X_all.apply(lambda x: x.idxmin())
    
    
    if plotting:
        #print type(weights)
        #plt.hist(weights[sSelector],bins=50,color='b')
        #plt.hist(weights[bSelector],bins=50,color='r',alpha=0.3)
        plt.show()
        X[sSelector].hist(color='b', alpha=0.5, bins=50)
        X[bSelector].hist(color='r', alpha=0.5, bins=50)
    #split data again
    X = X_all[len(X_test.index):]
    X_test = X_all[:len(X_test.index)]
    
    if scale_data:
        X,X_test = scaleData(X,X_test)
    
    if clusterFeature:
	X,y,X_test,weights=clustering(X,y,X_test,weights,n_clusters=4,returnCluster=None,plotting=True)
    
    
    print "Dim train set:",X.shape    
    print "Dim test set :",X_test.shape
    return (X,y,X_test,weights)

    
def AMS(s,b,factor):
    assert s >= 0
    assert b >= 0
    s = s * factor
    b = b * factor
    bReg = 10.
    return math.sqrt(2 * ((s + b + bReg) * 
                          math.log(1 + s / (b + bReg)) - s))


def modTrainWeights(wtrain,lytrain,scale_wt=None,verbose=False):
    """
    Modify training weights
    """
    sSelector = lytrain==1
    wsum_s = np.sum(wtrain[sSelector])
    wsum_b = np.sum(wtrain[lytrain==0])
    if scale_wt=='auto':             
	scale_wt=0.33*wsum_b/wsum_s
    elif scale_wt is None:
	scale_wt=1.0
    wtrain_fit = np.copy(wtrain)         
    wtrain_fit[sSelector] = wtrain[sSelector]*scale_wt
    #wtrain_fit = wtrain*scale_wt
    #plt.hist(wtrain_fit[sSelector],bins=500,color='g')
    #plt.hist(wtrain_fit,bins=500,color='b',alpha=0.3)
    #plt.show()
    if verbose: print "wsum,s: %4.2f wsum,s,mod: %4.2f wsum,b: %4.2f orig ratio: %4.2f scale_factor: %8.3f\n"%(wsum_s,np.sum(wtrain_fit[sSelector]),wsum_b,wsum_b/wsum_s,scale_wt)
    return wtrain_fit
  

def amsXvalidation(lmodel,lX,ly,lw,nfolds=5,cutoff=0.5,useProba=True,useWeights=True,useRegressor=False,scale_wt=None,buildModel=False):
    """
    Carries out crossvalidation using AMS metrics
    """
    #ntotal = 250000
    vfunc = np.vectorize(binarizeProbs)
    
    lX = np.asarray(lX)
    ly = np.asarray(ly)
    cv = StratifiedKFold(ly, nfolds)
    #cv = StratifiedShuffleSplit(ly, nfolds, test_size=0.5)
    #cv = KFold(lX.shape[0], n_folds=nfolds,shuffle=True)
    scores=np.zeros(nfolds)
    ams_scores=np.zeros(nfolds)
    scores_train=np.zeros(nfolds)
    ams_scores_train=np.zeros(nfolds)
    for i, (train, test) in enumerate(cv):	
	#train
	lytrain = ly[train]
	wtrain= lw[train]
	if useWeights is False:
	    print "Ignoring weights for fit."
	    lmodel.fit(lX[train],lytrain)
	else:
	 #scale wtrain
         wtrain_fit=modTrainWeights(wtrain,lytrain,scale_wt)
         lmodel.fit(lX[train],lytrain,sample_weight=wtrain_fit)
	
	#training data
	sc_string='AUC'
	if useProba:
	    yinbag=lmodel.predict_proba(lX[train])
	    scores_train[i]=roc_auc_score(lytrain,yinbag[:,1])	   
	else: 
         yinbag=lmodel.predict(lX[train])
         if useRegressor:
             yinbag = vfunc(yinbag,cutoff)
	    
         scores_train[i]=precision_score(lytrain,yinbag)
         sc_string='PRECISION'

	ams_scores_train[i]=ams_score(lytrain,yinbag,sample_weight=wtrain,use_proba=useProba,cutoff=cutoff)
	print "Training %8s=%6.3f AMS=%6.3f" % (sc_string,scores_train[i],ams_scores_train[i])
	
	#test
	truth=ly[test]
	weightsTest=lw[test]
	
	if useProba:
	    yoob=lmodel.predict_proba(lX[test])
	    scores[i]=roc_auc_score(truth,yoob[:,1]) 
	    sSelector = truth==1
	    bSelector = truth==0
	    wsum=np.sum(yoob >= cutoff)
	    wsum_truth=np.sum(sSelector)
	    print "Cutoff: %6.3f Nr. signals(pred): %4d Ratio(pred): %6.3f signals(truth): %4d ratio(truth): %6.3f" %(cutoff,wsum,wsum/(float(yoob.shape[0])),wsum_truth,wsum_truth/(float(yoob.shape[0])) )
	    #plt.hist(yoob[:,1],bins=50,color='b')
	    #plt.hist(yoob[:,1][bSelector],bins=50,alpha=0.3,color='g')
	    plt.show()
	    
	    
	else:
         yoob=lmodel.predict(lX[test])
         if useRegressor:
             yoob = vfunc(yoob,cutoff)
          #scores[i]=f1_score(truth,yoob)
         scores[i]=precision_score(truth,yoob)

	ams_scores[i]=ams_score(truth,yoob,sample_weight=weightsTest,use_proba=useProba,cutoff=cutoff)
	print "Iteration=%d %d/%d %-8s=%6.3f AMS=%6.3f\n" % (i+1,train.shape[0], test.shape[0],sc_string,scores[i],ams_scores[i])
	
	
    print "\n##XV SUMMARY##"
    print " <%-8s>: %0.3f (+/- %0.3f)" % (sc_string,scores.mean(), scores.std())
    print " <AMS>: %0.3f (+/- %0.3f)" % (ams_scores.mean(), ams_scores.std())
    print " <%-8s,train>: %0.3f (+/- %0.3f)" % (sc_string,scores_train.mean(), scores_train.std())
    print " <AMS,train>: %0.3f (+/- %0.3f)" % (ams_scores_train.mean(), ams_scores_train.std())
    
    if buildModel:
	print "\n##Building final model##"
	if useWeights:
	    w_fit=modTrainWeights(lw,ly,scale_wt)
	    lmodel.fit(lX,ly,sample_weight=w_fit)
	else:
	    lmodel.fit(lX,ly)
	return model
    else:
	return None

  
def ams_score(y_true,y_pred,**kwargs):
    """
    Higgs AMS metric
    """  
    #use scoring weights if available
    if 'scoring_weight' in kwargs:
	sample_weight=kwargs['scoring_weight']
    elif 'sample_weight' in kwargs:
	sample_weight=kwargs['sample_weight']
    else:
	print "We need sample weights for sensible evaluation!"
	return
	
    cutoff=0.5
    ntotal=250000
    tpr=0.0
    fpr=0.0 

    #check if we are dealing with proba
    info="- using 0-1 classification"
    if kwargs['use_proba']  and kwargs['cutoff'] is not None:
	if 'cutoff' in kwargs:
	    cutoff=kwargs['cutoff']
	#print len(y_pred.shape)
	if len(y_pred.shape)>1:
	    y_pred=y_pred[:,1]
	info="- using probabilities with cutoff=%4.2f"%(cutoff)
        
    for j,row in enumerate(y_pred):
	    if row>=cutoff:		
		if y_true[j]>=cutoff:
		    tpr=tpr+sample_weight[j]
		else:
		    fpr=fpr+sample_weight[j]

    sSelector = y_true==1
    #print "Unique weights for signal:",np.unique(sample_weight[sSelector])[0:5]
    
    wfactor=1.* ntotal / y_true.shape[0]
    smax = np.sum(sample_weight[sSelector])
    ams_max=AMS(smax, 0.0,wfactor)
    ams=AMS(tpr, fpr,wfactor)
    print 'AMS = %6.3f [AMS_max = %6.3f] %-32s'%(ams,ams_max,info)
    
    return ams   
    
def amsGridsearch(lmodel,lX,ly,lw,fitWithWeights=False,nfolds=5,useProba=False,cutoff=0.5,scale_wt='auto'):
    print 
    if not 'sample_weight' in inspect.getargspec(lmodel.fit).args:
	  print("WARNING: Fit function ignores sample_weight!")
	  
    fit_params = {'scoring_weight': lw}
    if scale_wt is None:
	fit_params['sample_weight']=lw
    else:
	wtrain_fit=modTrainWeights(lw,ly,scale_wt)
	fit_params['sample_weight']=wtrain_fit
	
    fit_params['fitWithWeights']=fitWithWeights
    
    #https://github.com/scikit-learn/scikit-learn/issues/3223 + own modifications
    ams_scorer = make_scorer(score_func=ams_score,use_proba=useProba,cutoff=cutoff)
    
    #parameters = {'n_estimators':[150,300], 'max_features':[5,10]}#rf
    parameters = {'n_estimators':[250], 'max_features':[6,8,10],'min_samples_leaf':[5,10]}#xrf+xrf
    #parameters = {'max_depth':[3,5,10], 'learning_rate':[0.5,0.1,0.01,0.05,0.001],'n_estimators':[150,300,500],'subsample':[1.0]}#gbm
    
    #parameters = {'max_depth':[6,5], 'learning_rate':[0.1,0.09,0.08],'n_estimators':[150],'subsample':[1.0],'loss':['deviance'],'min_samples_leaf':[20],'max_features':[6,8,10]}#gbm
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
    return(clf_opt.best_estimator_)
	
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
    #learn_score = make_scorer(score_func=ams_score,use_proba=useProba)
    learn_score=make_scorer(f1_score)
    plot_learning_curve(model, "learning curve", X, y, ylim=(0.1, 1.01), cv=cv, n_jobs=4,scoring=learn_score)

    
def buildAMSModel(lmodel,lXs,ly,lw=None,fitWithWeights=False,nfolds=8,useProba=False,cutoff=0.5,feature_names=None,scale_wt=None,saveModel=True):
    """   
    Final model building part
    """ 
  
    fit_params = {'scoring_weight': lw}
    if scale_wt is None:
	fit_params['sample_weight']=lw
    else:
	fit_params['sample_weight']=modTrainWeights(lw,ly,scale_wt)
	
    fit_params['fitWithWeights']=fitWithWeights
    ams_scorer = make_scorer(score_func=ams_score,use_proba=useProba,cutoff=cutoff)
    
    #how can we avoid that samples are used for fitting
    print "Xvalidation..."
    scores = cross_validation.cross_val_score(lmodel,lXs,ly,fit_params=fit_params, scoring=ams_scorer,cv=nfolds,n_jobs=4)
    print "AMS: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std())
    print "Building model with all instances..."
    
    if fitWithWeights:
	    lmodel.fit(lXs,ly,sample_weight=fit_params['sample_weight'])
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


def pcAnalysis(X,Xtest,y,w=None,ncomp=2,transform=False):
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
    else:
        print "PC analysis"
        #X_all = pd.concat([Xtest, X])
        X_all = X
        #Uwaga! this is transformation is necessary otherwise PCA gives rubbish!!
        ytrain = np.asarray(y)        
        X_r = pca.fit_transform(np.asarray(X_all))  
        
        if w is None:
            plt.scatter(X_r[ytrain == 0,0], X_r[ytrain == 0,1], c='r', label="background",alpha=0.1)
            plt.scatter(X_r[ytrain == 1,0], X_r[ytrain == 1,1], c='g',label="signal",alpha=0.1)
        else:
            plt.scatter(X_r[ytrain == 0,0], X_r[ytrain == 0,1], c='r', label="background",s=w[ytrain==0]*25.0,alpha=0.1)
            plt.scatter(X_r[ytrain == 1,0], X_r[ytrain == 1,1], c='g',label="signal",s=w[ytrain==1]*1000.0,alpha=0.1)

        print(pca.explained_variance_ratio_) 
        plt.legend()
        #plt.xlim(-3500,2000)
        #plt.ylim(-1000,2000)
        plt.draw()
        
        #clustering        

def clustering(Xtrain,ytrain,Xtest,wtrain=None,n_clusters=3,returnCluster=0,plotting=False):
        """
        Cluster data set
        """
        if returnCluster is not None and returnCluster+1>n_clusters:
            print "Error: returnCluster can not exceed number of clusters!"
                    
        X_all = pd.concat([Xtest, Xtrain])

        centroids,label,inertia = k_means(X_all,n_clusters=n_clusters,verbose=False,n_jobs=2)
        
        pca = PCA(n_components=2)
        X_r = pca.fit_transform(np.asarray(X_all))        
        
        #plt.hist(label,bins=40)
    
        print range(n_clusters)
        cluster_names=['cluster0','cluster1','cluster2','cluster3','cluster4','cluster5']
        for c, i, target_name in zip("rgbkcy", range(n_clusters), cluster_names):
            if plotting:
                plt.scatter(X_r[label == i, 0], X_r[label == i, 1], c=c, label=target_name,alpha=0.1)
            print "Cluster %4d n: %4d"%(i,np.sum(label==i))
        
        #plt.legend()        
        
        #alternatively we could append cluster label to dataset...                
        X_new=pd.DataFrame(data=label,columns=['cluster_type'],index=X_all.index)                
     
        X_sub=pd.concat([X_all, X_new],axis=1)
        
        #print X_sub.describe()
        #X_sub=X_all.iloc[label==returnCluster,:]        
        
        #print X_sub
                
        #Xtrain_sub = X_sub[len(Xtest.index):]
        #Xtest_sub = X_sub[:len(Xtest.index)] 
        Xtrain = X_sub[len(Xtest.index):]
        Xtest = X_sub[:len(Xtest.index)]
                                     
        if returnCluster is not None:                             
            #print Xtrain['cluster_type']==returnCluster
            clusterSelect=np.asarray(Xtrain['cluster_type']==returnCluster)                                        
                                            
            Xtrain_sub = Xtrain.loc[clusterSelect,:]
            #print type(wtrain)        
            wtrain_sub = wtrain[clusterSelect]
            ytrain_sub = ytrain.loc[clusterSelect]
            
            Xtest_sub = Xtest.loc[Xtest['cluster_type']==returnCluster,:]
            
            Xtest_sub=Xtest_sub.drop(['cluster_type'],axis=1)       
            Xtrain_sub=Xtrain_sub.drop(['cluster_type'],axis=1)    
            
            print "Dim of return train set:",Xtrain_sub.shape  
            print "Dim of return test set:",Xtest_sub.shape               
            
            #print Xtrain_sub
            #print wtrain_sub
            #print ytrain_sub        
            return (Xtrain_sub,ytrain_sub,Xtest_sub,wtrain_sub)
        else:    
            return (Xtrain,ytrain,Xtest,wtrain)

    
    
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
    # TODO scale sample weights????http://pastebin.com/VcL3dWiK
    # TODO predict weights???
    # weights as np.array
    # group by jet
    # group by different NA
    #we should optimise precision instead of recall
    # sample_weights are small for signals? look at TPR vs. FPR wenn weights are scaled for signals 
    # we need training weights and scoring weights
    t0 = time()
    
    print "numpy:",np.__version__
    print "pandas:",pd.__version__
    print "sklearn:",sl.__version__
    #pd.set_option('display.height', 5)
    pd.set_option('display.max_columns', 14)
    pd.set_option('display.max_rows', 40)
    
    np.random.seed(123)
    nsamples=100000
    onlyPRI='' #'PRI' or 'DER'
    createNAFeats=False #brings something?
    dropCorrelated=False
    scale_data=False #bringt nichts by NB
    replaceNA=False
    plotting=False
    stats=False
    transform=False
    useProba=True  #use probailities for prediction
    useWeights=False #use weights for training
    scale_wt=None
    useRegressor=False
    cutoff=0.75
    clusterFeature=False
    subfile="/home/loschen/Desktop/datamining-kaggle/higgs/submissions/sub1007a.csv"
    Xtrain,ytrain,Xtest,wtrain=prepareDatasets(nsamples,onlyPRI,replaceNA,plotting,stats,transform,createNAFeats,dropCorrelated,scale_data,clusterFeature)
    nfolds=4#StratifiedShuffleSplit(ytrain, 8, test_size=0.5)
    
    #pcAnalysis(Xtrain,Xtest,ytrain,wtrain,ncomp=2,transform=False)       
    #RF cluster1 AMS=2.600 (77544)
    #RF cluster2 AMS=4.331 (72543)
    #RF cluster3 AMS=3.742 (99913 samples) ~ weighted average AMS = 3.56
    #NB cluster1 AMS=1.850 (77544 samples)
    #NB cluster2 AMS=1.834 (72543 samples)
    #NB cluster3 AMS=2.467 (9950,100,200,913 samples)
    #NB all AMS=1.162
    #NB all AMS=1.315 (variable transformation)
    #NB all AMS=1.315 (transformation+proba=0.5)
    #NB all AMS=1.374 (transformation+proba=0.15)
    #NB all AMS=1.473 (transformation+proba=0.05)
    #NB all AMS=1.641 (transformation+proba=0.01)
    #NB all AMS=1.745 (transformation+proba=0.01,replaceNA)
    #NB all AMS=1.822 (transformation+proba=0.15,replaceNA,dropcorrelated)
    #print Xtrain.describe()
    #model = RidgeClassifier(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, class_weight=None, solver='auto')#AMS~1.85
    #model = LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1.0, fit_intercept=True, class_weight='auto')
    #model = SGDClassifier(alpha=0.1,n_iter=150,shuffle=True,loss='log',penalty='l1',n_jobs=4)#SW,correct implementation?
    #model = GaussianNB()#AMS~1.2
    #model = GradientBoostingClassifier(loss='exponential',n_estimators=150,learning_rate=.2,max_depth=6,verbose=2,subsample=1.0)   
    #model = GradientBoostingClassifier(loss='deviance', learning_rate=0.001, n_estimators=500, subsample=1.0, max_depth=10, max_features='auto',init=None,verbose=False)#opt fitting wo weights!!!
    #model = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=150, subsample=0.5, max_depth=25, max_features='auto',init=None,verbose=False)#opt2
    
    #model = GradientBoostingRegressor(n_estimators=150,learning_rate=.1,max_depth=6,verbose=1,subsample=1.0)
    #model = GradientBoostingClassifier(loss='deviance',n_estimators=150, learning_rate=0.05,max_depth=6,min_samples_leaf=100,max_features='auto',verbose=0)#with no weights, cutoff=0.85 and useProba=True
    #model = pyGridSearch(model,Xtrain,ytrain)
    #model = Pipeline([('filter', SelectPercentile(f_classif, percentile=15)), ('model', GaussianNB())])
    #model = KNeighborsClassifier(n_neighbors=5,weights='distance',algorithm='ball_tree')#AMS~2.245
    #model = AdaBoostClassifier(n_estimators=150,learning_rate=0.1)
    
    #model = SVC(C=1.0,gamma=0.0)
    
    #model = ExtraTreesClassifier(n_estimators=250,max_depth=None,min_samples_leaf=5,n_jobs=1,criterion='entropy', max_features=10,oob_score=False)##scale_wt 600 cutoff 0.85
    #model = AdaBoostClassifier(n_estimators=100,learning_rate=0.1)    
    #analyzeLearningCurve(model,Xtrain,ytrain,wtrain)
    #odel = ExtraTreesClassifier(n_estimators=250,max_depth=None,min_samples_leaf=5,n_jobs=4,criterion='entropy', max_features=5,oob_score=False)#opt
    #model = Pipeline([('filter', SelectPercentile(f_classif, percentile=15)), ('model', model)])
    #model = amsGridsearch(model,Xtrain,ytrain,wtrain,fitWithWeights=useWeights,nfolds=nfolds,useProba=useProba,cutoff=cutoff)
    #model = GradientBoostingClassifier(loss='deviance',n_estimators=150, learning_rate=0.1, max_depth=6,subsample=1.0,verbose=False) #opt weight =500 AMS=3.548
    #model = GradientBoostingClassifier(loss='deviance',n_estimators=200, learning_rate=0.08, max_depth=7,subsample=1.0,max_features=10,min_samples_leaf=20,verbose=False) #opt weight =500 AMS=3.548
    #model = XgboostClassifier(n_estimators=120,learning_rate=0.1,max_depth=6,n_jobs=4,cutoff=0.5,NA=-999.9)
    model =  RandomForestClassifier(n_estimators=250,max_depth=None,min_samples_leaf=5,n_jobs=4,criterion='entropy', max_features=5,oob_score=False)#SW-proba=False ams=3.42
    #amsXvalidation(model,Xtrain,ytrain,wtrain,nfolds=nfolds,cutoff=p,useProba=useProba,useWeights=useWeights,useRegressor=useRegressor,scale_wt=c,buildModel=True)
    #iterativeFeatureSelection(model,Xtrain,Xtest,ytrain,1,1)    
    amsXvalidation(model,Xtrain,ytrain,wtrain,nfolds=nfolds,cutoff=cutoff,useProba=useProba,useWeights=useWeights,useRegressor=useRegressor,scale_wt=scale_wt,buildModel=False)
    #clist=[0.25,0.50,0.75,0.85]
    #for c in clist:
	#  cutoff=c
	 # print "c",c
	  #amsXvalidation(model,Xtrain,ytrain,wtrain,nfolds=nfolds,cutoff=cutoff,useProba=useProba,useWeights=useWeights,useRegressor=useRegressor,scale_wt=scale_wt,buildModel=False)
	  #model=buildAMSModel(model,Xtrain,ytrain,wtrain,nfolds=nfolds,fitWithWeights=useWeights,useProba=useProba,cutoff=cutoff,scale_wt=scale_wt)
	  #model = amsGridsearch(model,Xtrain,ytrain,wtrain,fitWithWeights=useWeights,nfolds=nfolds,useProba=useProba,cutoff=cutoff,scale_wt=scale_wt)
    print model
    makePredictions(model,Xtest,subfile,useProba=useProba,cutoff=cutoff)
    checksubmission(subfile)
    print("Model building done on %d samples in %fs" % (Xtrain.shape[0],time() - t0))
    plt.show()