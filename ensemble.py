#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""  Chrissly31415
August,September 2013

"""

from higgs import *
from FullModel import *
import itertools
from scipy.optimize import fmin,fmin_cobyla
from random import randint
import sys

train_indices=[]
test_indices=[]

def createModels():
    global train_indices,test_indices
    ensemble=[]
    
    #GBM1 AMS~3.66 OK
    #X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=False,dropCorrelated=False,scale_data=False,clusterFeature=False)
    #model = GradientBoostingClassifier(loss='deviance',n_estimators=200, learning_rate=0.08, max_depth=7,subsample=1.0,max_features=10,min_samples_leaf=20,verbose=False) #opt weight =500 AMS=3.548
    #xmodel = XModel("gbm1",model,X,Xtest,w,cutoff=0.85,scale_wt=200)
    #ensemble.append(xmodel)
    
    #XRF1 AMS~3.4 OK
    #X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=False,dropCorrelated=False,scale_data=False,clusterFeature=False)
    #model = ExtraTreesClassifier(n_estimators=250,max_depth=None,min_samples_leaf=5,n_jobs=4,criterion='entropy', max_features=10,oob_score=False)##scale_wt 600 cutoff 0.85
    #xmodel = XModel("xrf1",model,X,Xtest,w,cutoff=0.85,scale_wt=600)
    #ensemble.append(xmodel)
    
    #RF1 AMS~3.5 OK
    #X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=False,dropCorrelated=False,scale_data=False,clusterFeature=False)
    #model =  RandomForestClassifier(n_estimators=300,max_depth=None,min_samples_leaf=10,n_jobs=4,criterion='entropy', max_features=5,oob_score=False)
    #xmodel = XModel("rf1",model,X,Xtest,w,cutoff=0.85,scale_wt=600)
    #ensemble.append(xmodel)
    
    #RF2 AMS~3.5 no weights for fit!!! OK
    #X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=False,dropCorrelated=False,scale_data=False,clusterFeature=False)
    #model =  RandomForestClassifier(n_estimators=250,max_depth=None,min_samples_leaf=5,n_jobs=4,criterion='entropy', max_features=5,oob_score=False)
    #xmodel = XModel("rf2",model,X,Xtest,w,cutoff=0.75,scale_wt=None)
    #ensemble.append(xmodel)
    
    #GBM2 loss exponential AMS ~3.5 ->predict_proba... OK
    #X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=False,dropCorrelated=False,scale_data=False,clusterFeature=False)
    #model = GradientBoostingClassifier(loss='exponential',n_estimators=200, learning_rate=0.2, max_depth=6,subsample=1.0,min_samples_leaf=5,verbose=False)
    #xmodel = XModel("gbm2",model,X,Xtest,w,cutoff=None,scale_wt=35)
    #ensemble.append(xmodel)
    
    #TODO RF2MOD modfiy weights only slighty from unity
    #TODO modify sqrt(negwts) or pow(posweihgts,n*0.5) for training with n as parameter
    #TODO maximizeAMS from minimizeAUC!
    #TODO create test set within xval loop, i.e. bagging approach for gradient boosting
    
    #XGBOOST AMS ~3.58 (PUB.LD)
    #X,y,Xtest,w=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=False,dropCorrelated=False,scale_data=False,clusterFeature=False)
    #model = XgboostClassifier(n_estimators=120,learning_rate=0.1,max_depth=6,n_jobs=4,cutoff=0.5,NA=-999.9)
    #xmodel = XModel("xgboost1",model,X,Xtest,w,cutoff=0.7,scale_wt=1)
    #ensemble.append(xmodel)
    
    #ADAboost
    #(X,y,X_test,test_indices,train_indices) = prepareDatasets('hV',useSVD=10,useJson=False,useHTMLtag=False,useAddFeatures=False,usePosTag=False,useAlcat=False,useGreedyFilter=False,char_ngram=1,loadTemp=True)
    #model = Pipeline([('filter', SelectPercentile(f_classif, percentile=80)), ('model', AdaBoostClassifier(n_estimators=200,learning_rate=0.1))])
    
    
    #collect them
    for m in ensemble:
	m.summary()
    return(ensemble,y)

def createOOBdata(ensemble,ly,repeats=5,bagging=False):
    """
    Get cv oob predictions for classifiers
    """
    vfunc = np.vectorize(binarizeProbs)
    folds=4
    
    for m in ensemble:
        use_proba = m.cutoff is not None
	print "Computing oob predictions for:",m.name
	print m.classifier.get_params
	oobpreds=np.zeros((m.Xtrain.shape[0],repeats))
	ybag=np.zeros((m.Xtest.shape[0],repeats))
	
	for j in xrange(repeats):
	    #print lmodel.get_params()
	    cv = KFold(m.Xtrain.shape[0], n_folds=folds,random_state=j,shuffle=True)

	    scores=np.zeros(folds)
	    ams_scores=np.zeros(folds)
	    for i, (train, valid) in enumerate(cv):
		Xtrain = m.Xtrain.iloc[train]
		Xvalid = m.Xtrain.iloc[valid]
		ytrain = ly[train]
		wtrain = m.sample_weight[train]
		if m.scale_wt is not None:
		    wtrain_fit=modTrainWeights(wtrain,ytrain,m.scale_wt)
		else:
		    wtrain_fit=None
		m.classifier.fit(Xtrain,ytrain,sample_weight=wtrain_fit)
		if use_proba:
		    oobpreds[valid,j] = m.classifier.predict_proba(Xvalid)[:,1]
		    scores[i]=roc_auc_score(ly[valid],oobpreds[valid,j])
		    #makes bagging prediction, use partial model to fit test datamining
		    ybag[:,j] = m.classifier.predict_proba(m.Xtest)[:,1]
		    
		    
		else:
		    oobpreds[valid,j] = m.classifier.predict(Xvalid)
		    ybag[:,j] = m.classifier.predict(m.Xtest)
		
		ams_scores[i]=ams_score(ly[valid],oobpreds[valid,j],sample_weight=m.sample_weight[valid],use_proba=use_proba,cutoff=m.cutoff)
		#print "*AMS: %0.2f " % (ams_scores[i])
		
	    
	    #oobscore=roc_auc_score(ly,oobpreds[:,j])
	    ams_oobscore=ams_score(ly,oobpreds[:,j],sample_weight=m.sample_weight,use_proba=use_proba,cutoff=m.cutoff)
	    
	    print "Iteration:",j,
	    #print " <AUC>: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std()),	    
	    #print " AUC,oob: %0.3f" %(oobscore),
	    print " <AMS>: %0.3f (+/- %0.3f)" % (ams_scores.mean(), ams_scores.std()),
	    print " AMS,oob: %0.3f" %(ams_oobscore)
	    
	#collect OOB scores    
	#scores=[roc_auc_score(ly,oobpreds[:,j]) for j in xrange(repeats)]
		
	#simple averaging of blending
	oob_avg=np.mean(oobpreds,axis=1)
	#print "Summary: <AUC,oob>: %0.3f (%d repeats)" %(roc_auc_score(ly,oob_avg),repeats)
	print "Summary: <AMS,oob>: %0.3f (%d repeats)" %(ams_score(ly,oob_avg,sample_weight=m.sample_weight,use_proba=use_proba,cutoff=m.cutoff),repeats)
	
	#save oob + predictions to disc
	print "Train full modell and generate predictions..."
	
	w_fit=None
	if m.scale_wt is not None:
	    w_fit=modTrainWeights(m.sample_weight,ly,m.scale_wt)
	    	
	m.classifier.fit(m.Xtrain,ly,w_fit)
	
	if use_proba:
	    #probabilities
	    m.oobpreds=oob_avg
	    m.preds = m.classifier.predict_proba(m.Xtest)[:,1]	  
	    #collect bagging predictions
	    ybag_avg=np.mean(ybag,axis=1)
	    
	    #classes
	    yoob = vfunc(np.asarray(m.oobpreds),m.cutoff)
	    if bagging:
	        ypred = vfunc(np.asarray(ybag_avg),m.cutoff)
	    else:
		ypred = vfunc(np.asarray(m.preds),m.cutoff)
		
	else:
	    m.oobpreds=vfunc(oob_avg,0.5)
	    m.preds = m.classifier.predict(m.Xtest)
	    ybag_avg=np.mean(ybag,axis=1)	    
	    yoob = m.oobpreds
	    
	    if bagging:
	        ypred = vfunc(np.asarray(ybag_avg),0.5)
	    else:
		ypred = m.preds
	    #collect bagging predictions 
	#this is also necessary for 0-1 classifier
	
	m.summary()	
	#put data to data.frame and save	
	m.oobpreds=pd.DataFrame(np.asarray(m.oobpreds),columns=["proba"])
	tmp=pd.DataFrame(yoob,columns=["label"])
	m.oobpreds=pd.concat([tmp, m.oobpreds],axis=1)
			
	m.preds=pd.DataFrame(np.asarray(m.preds),columns=["proba"])
	tmp=pd.DataFrame(ypred,columns=["label"])	
	m.preds=pd.concat([tmp, m.preds],axis=1)
	#save final model

	allpred = pd.concat([m.preds, m.oobpreds])
	#print allpred
	#submission data is first, train data is last!
	filename="/home/loschen/Desktop/datamining-kaggle/higgs/data/"+m.name+".csv"
	print "Saving oob + predictions as csv to:",filename
	allpred.to_csv(filename,index=False)
	
	#XModel.saveModel(m,"/home/loschen/Desktop/datamining-kaggle/higgs/data/"+m.name+".pkl")
	
    return(ensemble)
   
def trainEnsemble(ensemble=None,useCols=None,addMetaFeatures=False):
    test_indices = pd.read_csv('../datamining-kaggle/higgs/test.csv', sep=",", na_values=['?'], index_col=0).index
    y = pd.read_csv('../datamining-kaggle/higgs/training.csv', sep=",", na_values=['?'])
    y = y['Label'].str.replace(r's','1').str.replace(r'b','0')
    y = np.asarray(y.astype(float))
    
    for i,model in enumerate(ensemble):
	print "Loading model:",i," name:",model
	if i>0:
	    X = pd.read_csv("/home/loschen/Desktop/datamining-kaggle/higgs/data/"+model+".csv", sep=",")[['proba']]
	    X.columns=[model]
	    Xall=pd.concat([Xall,X], axis=1)
	else:
	    Xall = pd.read_csv("/home/loschen/Desktop/datamining-kaggle/higgs/data/"+model+".csv", sep=",")[['proba']]
	    Xall.columns=[model]
    
    Xtrain=Xall[len(test_indices):]
    Xtest=Xall[:len(test_indices)]
    
    #if we add metafeature we should not use aucMinimize...
    if addMetaFeatures:
	multiply=False
	(Xs,y,Xs_test,w)=prepareDatasets(nsamples=-1,onlyPRI='',replaceNA=False,plotting=False,stats=False,transform=False,createNAFeats=False,dropCorrelated=False,scale_data=False,clusterFeature=False)

	Xs=Xs.loc[:,useCols]
	Xs.index=Xtrain.index
	Xs_test=Xs_test.loc[:,useCols]
	Xs_test.index=Xtest.index
		
	if multiply:
	    pass
	    #n=len(Xtrain.columns)
	    #for i,col in enumerate(useCols):
		#newcolnames=[]
		#for name in Xtrain.columns[0:n]:
		    #tmp=name+"X"+col
		    #newcolnames.append(tmp)
		#newcolnames= Xtrain.columns.append(pd.Index(newcolnames))
		##print type(Xs.ix[:,i])
		##Xtrain=pd.concat([Xtrain,Xs], axis=1)	    
		#Xtrain=pd.concat([Xtrain,Xtrain.ix[:,0:n].mul(Xs.ix[:,i],axis=0)], axis=1)
		#Xtrain.columns=newcolnames
		#Xtest=pd.concat([Xtest,Xtest.ix[:,0:n].mul(Xs_test.ix[:,i],axis=0)], axis=1)
		#Xtest.columns=newcolnames
		
	else:
	    Xtrain=pd.concat([Xtrain,Xs], axis=1)
	    Xtest=pd.concat([Xtest,Xs_test], axis=1)
	    
	    print "Dim train",Xtrain.shape
	    print "Dim test",Xtest.shape
    
    #scale data
    X_all=pd.concat([Xtest,Xtrain])   
    X_all=removeCorrelations(X_all,0.99)
    #X_all = (X_all - X_all.min()) / (X_all.max() - X_all.min())
    Xtrain=X_all[len(test_indices):]
    Xtest=X_all[:len(test_indices)]
    #print Xtrain
    #print "New shape",Xtrain.shape
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
    print c
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
    weights=np.asarray(pd.read_csv('../datamining-kaggle/higgs/training.csv', sep=",", na_values=['?'], index_col=0)['Weight'])
    #blending
    folds=4
    cutoff_all=0.85
    scale_wt=200
    
    print "Blending, using general cutoff %4.3f, "%(cutoff_all),
    if scale_wt is not None:
	print "scale_weights %5.1f:"%(scale_wt)
  
    #blender=LogisticRegression(penalty='l1', tol=0.0001, C=1.0)
    #blender = Pipeline([('filter', SelectPercentile(f_regression, percentile=25)), ('model', LogisticRegression(penalty='l2', tol=0.0001, C=0.1))])
    #blender=SGDClassifier(alpha=.01, n_iter=50,penalty='l2',loss='log')
    #blender=AdaBoostClassifier(learning_rate=0.01,n_estimators=50)
    blender=RandomForestClassifier(n_estimators=50,n_jobs=4, max_features='auto',oob_score=False)
    #blender=ExtraTreesClassifier(n_estimators=500,max_depth=None,min_samples_leaf=5,n_jobs=4,criterion='entropy', max_features='auto',oob_score=False)
    #blender=RandomForestClassifier(n_estimators=500,max_depth=None,min_samples_leaf=10,n_jobs=1,criterion='entropy', max_features=5,oob_score=False)
    #blender=ExtraTreesRegressor(n_estimators=500,max_depth=None)
    cv = KFold(oobpreds.shape[0], n_folds=folds,random_state=123)
    blend_scores=np.zeros(folds)
    ams_scores=np.zeros(folds)
    blend_oob=np.zeros((oobpreds.shape[0]))
    for i, (train, test) in enumerate(cv):
	Xtrain = oobpreds.iloc[train]
	Xtest = oobpreds.iloc[test]
	wfit=None
	if scale_wt is not None:
	  wfit = modTrainWeights(weights[train],ly[train],scale_wt)
	  blender.fit(Xtrain, ly[train],sample_weight=wfit)
	else:
	  blender.fit(Xtrain, ly[train])
	
	
	if hasattr(blender,'predict_proba'):
	    blend_oob[test] = blender.predict_proba(Xtest)[:,1]
	    use_proba=True
	else:
	    blend_oob[test] = blender.predict(Xtest)
	    use_proba=False
	blend_scores[i]=roc_auc_score(ly[test],blend_oob[test])
	ams_scores[i]=ams_score(ly[test],blend_oob[test],sample_weight=weights[test],use_proba=use_proba,cutoff=cutoff_all,verbose=True)
	
    print " <AUC>: %0.4f (+/- %0.4f)" % (blend_scores.mean(), blend_scores.std()),
    oob_auc=roc_auc_score(ly,blend_oob)
    print " AUC oob after blending: %0.4f" %(oob_auc)
    
    print " <AMS>: %0.4f (+/- %0.4f)" % (ams_scores.mean(), ams_scores.std()),
    oob_ams=ams_score(ly,blend_oob,sample_weight=weights,use_proba=use_proba,cutoff=cutoff_all,verbose=False)
    print " AMS oob after blending: %0.4f" %(oob_ams)
    
    if hasattr(blender,'coef_'):
      for i,model in enumerate(oobpreds.columns):
	ams=ams_score(ly,oobpreds.iloc[:,i],sample_weight=weights,use_proba=use_proba,cutoff=cutoff_all,verbose=False)
	auc=roc_auc_score(ly, oobpreds.iloc[:,i])
	print "%-16s ams: %5.3f auc: %4.3f coef: %4.3f" %(model,ams,auc,blender.coef_[0][i])
      print "Sum: %4.4f"%(np.sum(blender.coef_))
    #plt.plot(range(len(ensemble)),scores,'ro')
    
    #Prediction
    print "Make final ensemble prediction..."
    #make prediction for each classifiers
 
    if scale_wt is not None:
	wfit = modTrainWeights(weights,ly,scale_wt)
	preds=blender.fit(oobpreds,ly,sample_weight=wfit)
    else:
	preds=blender.fit(oobpreds,ly)
    #blend results
    preds=blender.predict_proba(testset)[:,1]   
    
    subfile='/home/loschen/Desktop/datamining-kaggle/higgs/submissions/subXXXa.csv'
    #preds=pd.DataFrame(preds,columns=["label"],index=test_indices)
    testset.index=test_indices
    makePredictions(preds,testset,subfile,useProba=True,cutoff=cutoff_all)
    #checksubmission(subfile)
    #preds.to_csv('/home/loschen/Desktop/datamining-kaggle/higgs/submissions/subXXXa.csv')
    #print preds
    return(oob_ams)


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
    preds.to_csv('/home/loschen/Desktop/datamining-kaggle/higgs/submissions/subXXXa.csv')
    print preds.describe()
    return(oob_auc)
  
def selectModels(): 
    #ensemble=["logreg_postag","lgblend","logreg1","logreg2_cv","logreg_wordtag","logreg_tagword","sgd1","sgd2","naiveB1","randomf1","randomf2","randomf3","randomf_1000SVD","extrarf1","extrarf2","gradboost1","gradboost2","gbmR","gbmR2","gbmR4","rfR","lr_char22_hV","lr_char33_hV","lr_char44_hV","lr_char55_hV","lr_char11","lr_char22","lr_char33","lr_char44","lr_char55"]
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
    #ensemble=createOOBdata(ensemble,y,5,bagging=False)
    models=["gbm1","rf1","rf2","xrf1","gbm2"]
    #models=["gbm1"]
    useCols=['DER_mass_MMC']
    trainEnsemble(models,useCols,addMetaFeatures=True)
    #selectModels()
    