#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""  
Ensemble helper tools

Chrissly31415
October,September 2014

"""

from FullModel import *
import itertools
from scipy.optimize import fmin,fmin_cobyla
from random import randint
import sys
from sklearn.externals.joblib import Parallel, delayed, logger
from sklearn.base import clone
from crowd import *


def createModels():
    ensemble=[]
   
    #KNN1 ~ 0.576
    #garbage=["<.*?>", "http", "www","img","border","style","px","margin","left", "right","font","solid","This translation tool is for your convenience only.*?Note: The accuracy and accessibility of the resulting translation is not guaranteed"]
    #garbage2=['http','www','img','border','0','1','2','3','4','5','6','7','8','9','a','the']
    #stop_words = text.ENGLISH_STOP_WORDS.union(garbage).union(garbage2)
    #(Xtrain, ytrain, Xtest,idx)  = prepareDataset(seed=42,nsamples=-1,doSeparateTFID=['product_title','query'],doSVDseparate=200,stop_words=stop_words,standardize=True)
    #model = KNeighborsClassifier(n_neighbors=5)
    #xmodel = XModel("knn1_r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,class_names=None)
    #ensemble.append(xmodel)
    
    #SVM1
    #garbage=["<.*?>", "http", "www","img","border","style","px","margin","left", "right","font","solid","This translation tool is for your convenience only.*?Note: The accuracy and accessibility of the resulting translation is not guaranteed"]
    #garbage2=['http','www','img','border','0','1','2','3','4','5','6','7','8','9','a','the']
    #stop_words = text.ENGLISH_STOP_WORDS.union(garbage).union(garbage2)
    #(Xtrain, ytrain, Xtest,idx)  = prepareDataset(seed=42,nsamples=-1,doSeparateTFID=['product_title','query'],doSVDseparate=200,stop_words=stop_words,standardize=True)
    #model = SVC(C=32,gamma=0.001)
    #xmodel = XModel("svm1_r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,class_names=None)
    #ensemble.append(xmodel)

    #SVM2
    stop_words = text.ENGLISH_STOP_WORDS
    (Xtrain, ytrain, Xtest,idx)  = prepareDataset(seed=42,nsamples=-1,doTFID=True,concat=True,doSVD=400,stop_words=stop_words,standardize=True)
    model = SVC(C=10,gamma=0.0)
    xmodel = XModel("svm2_r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,class_names=None)
    ensemble.append(xmodel)
    
    
    #some info
    for m in ensemble:
	m.summary()
    return(ensemble)

    
def finalizeModel(m,binarizeProbas=False,use_proba=True):
	"""
	Make predictions and save them
	"""
	print "Make predictions and save them..."
	if binarizeProbas:
	    #oob class predictions,binarize data
	    vfunc = np.vectorize(binarizeProbs)
	    #oob from crossvalidation
	    yoob = vfunc(np.asarray(m.oob_preds),m.cutoff)
	    #probas final prediction
	    m.preds = m.classifier.predict_proba(m.Xtest)[:,1]	  	    
	    #classes for final test set
	    ypred = vfunc(np.asarray(m.preds),m.cutoff)

	else:
	    #if 0-1 outcome, only classes or regression
	    #oob from crossvalidation
	    yoob = m.oob_preds
	    #final prediction
	    if use_proba:
	      m.preds = m.classifier.predict_proba(m.Xtest)
	    else:
	      m.preds = m.classifier.predict(m.Xtest)
	    ypred = m.preds
	    
	m.summary()	
	
	#put data to data.frame and save
	#OOB DATA
	m.oob_preds=pd.DataFrame(np.asarray(m.oob_preds),columns=['prediction'])
		
	#TESTSET prediction	
	m.preds=pd.DataFrame(np.asarray(m.preds),columns=['prediction'])
	
	#save final model
	allpred = pd.concat([m.preds, m.oob_preds])
	#submission data is first, train data is last!
	filename="/home/loschen/Desktop/datamining-kaggle/crowdflower/data/"+m.name+".csv"
	print "Saving oob + predictions as csv to:",filename
	allpred.to_csv(filename,index=False)
	
	#XModel.saveModel(m,"/home/loschen/Desktop/datamining-kaggle/higgs/data/"+m.name+".pkl")
	XModel.saveCoreData(m,"/home/loschen/Desktop/datamining-kaggle/crowdflower/data/"+m.name+".pkl")
	return(m)
    

def createOOBdata_parallel(ensemble,repeats=2,nfolds=5,n_jobs=1,score_func='log_loss',verbose=False,calibrate=False,binarize=False,returnModel=False,use_proba=True):
    """
    parallel oob creation
    """
    global funcdict

    for m in ensemble:
	print "Computing oob predictions for:",m.name
	print m.classifier.get_params
	if m.class_names is not None:
	    n_classes = len(m.class_names)
	else:
	    n_classes = 1
	print "n_classes",n_classes
	
	oob_preds=np.zeros((m.ytrain.shape[0],n_classes,repeats))
	ly=m.ytrain
	oobscore=np.zeros(repeats)
	maescore=np.zeros(repeats)
	
	#outer loop
	for j in xrange(repeats):
	    cv = StratifiedKFold(ly, n_folds=nfolds,shuffle=True,random_state=None)
	    #cv = StratifiedShuffleSplit(ly, n_iter=nfolds, test_size=0.25,random_state=j)
	    
	    scores=np.zeros(len(cv))
	    scores2=np.zeros(len(cv))
	    
	    #parallel stuff
	    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
			    pre_dispatch='2*n_jobs')
	    
	    #parallel run, returns a list of oob predictions
	    oob_pred = parallel(delayed(fit_and_score)(clone(m.classifier), m.Xtrain, ly, train, test,use_proba=use_proba,returnModel=False)
			  for train, test in cv)
	  
	    print oob_pred
	    if returnModel:
	      for in_bagmodel in inbag_models:
		  print in_bagmodel
	    
	    for i,(train,test) in enumerate(cv):
	        oob_pred[i] = oob_pred[i].reshape(oob_pred[i].shape[0],n_classes)
		oob_preds[test,:,j] = oob_pred[i]
		
		scores[i]=funcdict[score_func](ly[test],oob_preds[test,:,j])
		#print "Fold %d - score:%0.3f " % (i,scores[i])
		#scores_mae[i]=funcdict['mae'](ly[test],oob_preds[test,j])

	    oobscore[j]=funcdict[score_func](ly,oob_preds[:,:,j])
	    #maescore[j]=funcdict['mae'](ly,oob_preds[:,j])
	    
	    print "Iteration:",j,
	    print " <score>: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std()),
	    print " score,oob: %0.3f" %(oobscore[j])
	    #print " ## <mae>: %0.3f (+/- %0.3f)" % (scores_mae.mean(), scores_mae.std()),
	    #print " score3,oob: %0.3f" %(maescore[j])
	    
	#simple averaging of blending
	oob_avg=np.mean(oob_preds,axis=2)
	#probabilities
	m.oob_preds=oob_avg	

	if binarize:
	    #if 0-1 outcome, only classes
	    vfunc = np.vectorize(binarizeProbs)
	    m.oob_preds=vfunc(oob_avg,0.5)
	   
	score_oob = funcdict[score_func](ly,m.oob_preds)
	# = funcdict['mae'](ly,m.oob_preds)
	print "Summary: <score,oob>: %6.3f +- %6.3f   score,oob-total: %0.3f (after %d repeats)\n" %(oobscore.mean(),oobscore.std(),score_oob,repeats)
	
	#Train model with test sets
	print "Train full modell...",	
	if m.sample_weight is not None:
	    print "... with sample weights"
	    m.classifier.fit(m.Xtrain,ly,m.sample_weight)
	else:
	    m.classifier.fit(m.Xtrain,ly)
	
	m=finalizeModel(m,use_proba=use_proba)
	
    return(ensemble)
    
def fit_and_score(xmodel,X,y,train,valid,sample_weight=None,scale_wt=None,use_proba=False,returnModel=False):
    """
    Score function for parallel oob creation
    """
    if isinstance(X,pd.DataFrame): 
	Xtrain = X.iloc[train]
	Xvalid = X.iloc[valid]
    else:
	Xtrain = X[train]
	Xvalid = X[valid]
	
    ytrain = y[train]
    
    if sample_weight is not None: 
	wtrain = sample_weight[train]
	xmodel.fit(Xtrain,ytrain,sample_weight=wtrain)
    else:
	xmodel.fit(Xtrain,ytrain)
    
    if use_proba:
	  #saving out-of-bag predictions
	  local_pred = xmodel.predict_proba(Xvalid)
	  #prediction for test set
    #classification/regression   
    else:
	local_pred = xmodel.predict(Xvalid)
    #score=funcdict['rmse'](y[test],local_pred)
    #rndint = randint(0,10000000)
    #print "score %6.3f - %10d "%(score,rndint)
    #outpd = pd.DataFrame({'truth':y[test].flatten(),'pred':local_pred.flatten()})
    #outpd.index = Xvalid.index
    #outpd = pd.concat([Xvalid[['TMAP']], outpd],axis=1)
    #outpd.to_csv("pred"+str(rndint)+".csv")
    #raw_input()
    if returnModel:
       return local_pred, xmodel
    else:
       return local_pred

def trainEnsemble_multiclass(ensemble,mode='linear',score_func='log_loss',useCols=None,addMetaFeatures=False,use_proba=True,dropCorrelated=False,skipCV=False,subfile=""):
    """
    Train the ensemble
    """
    basedir="/home/loschen/Desktop/datamining-kaggle/crowdflower/data/"

    for i,model in enumerate(ensemble):
	
	print "Loading model:",i," name:",model
	xmodel = XModel.loadModel(basedir+model)
	class_names = xmodel.class_names
	if class_names is None:
	  class_names = ['Class']
	print "OOB data:",xmodel.oob_preds.shape
	print "pred data:",xmodel.preds.shape
	print "y train:",xmodel.ytrain.shape
	
	if i>0:
	    xmodel.oob_preds.columns = [model+"_"+n for n in class_names]
	    Xtrain = pd.concat([Xtrain,xmodel.oob_preds], axis=1)
	    Xtest = pd.concat([Xtest,xmodel.preds], axis=1)
	    
	else:
	    Xtrain = xmodel.oob_preds
	    Xtest = xmodel.preds
	    y = xmodel.ytrain
	    Xtrain.columns = [model+"_"+n for n in class_names]

    print Xtrain.columns
    print Xtrain.shape
    #print Xtrain.describe()
    print Xtest.shape
    #print Xtest.describe()
   
    if mode is 'classical':
	results=classicalBlend(ensemble,Xtrain,Xtest,y,skipCV=skipCV,subfile=subfile)
    elif mode is 'mean':
	results=linearBlend_multiclass(ensemble,Xtrain,Xtest,y,score_func=score_func,takeMean=True,subfile=subfile)
    elif mode is 'voting':
        results=voting_multiclass(ensemble,Xtrain,Xtest,y,subfile=subfile)
    else:
	results=linearBlend_multiclass(ensemble,Xtrain,Xtest,y,score_func=score_func,takeMean=False,subfile=subfile)
    return(results)

def voting_multiclass(ensemble,Xtrain,Xtest,y,n_classes=9,subfile=""):
    """
    Voting for multi classifiction result
    """
    print "Majority voting for predictions"
    voter = np.reshape(Xtrain.values,(Xtrain.shape[0],-1,n_classes)).swapaxes(0,1)

    for model in voter:
	max_idx=model.argmax(axis=1)
	for row,idx in zip(model,max_idx):
	    row[:]=0.0
	    row[idx]=1.0
    
    voter = voter.mean(axis=0)
    print voter
    print voter.shape

def classicalBlend(ensemble,oobpreds,testset,ly,use_proba=True,score_func='log_loss',subfile="",cv=8,skipCV=False):
    """
    Blending using sklearn classifier
    """
     
    #blender=LogisticRegression(penalty='l2', tol=0.0001, C=100)
    #blender = Pipeline([('filter', SelectPercentile(f_regression, percentile=25)), ('model', LogisticRegression(penalty='l2', tol=0.0001, C=0.1))])
    #blender=SGDClassifier(alpha=0.1, n_iter=50,penalty='l2',loss='log',n_jobs=folds)
    #blender=AdaBoostClassifier(learning_rate=0.01,n_estimators=50)
    #blender=RandomForestClassifier(n_estimators=500,n_jobs=4, max_features='auto',oob_score=False,min_samples_leaf=10,max_depth=None)
    #blender = CalibratedClassifierCV(blender, method='isotonic', cv=3)
    #blender=ExtraTreesClassifier(n_estimators=500,max_depth=None,min_samples_leaf=5,n_jobs=4,criterion='entropy', max_features='auto',oob_score=False)
    #blender=RandomForestClassifier(n_estimators=500,max_depth=None,min_samples_leaf=10,n_jobs=1,criterion='entropy', max_features=5,oob_score=False)
    blender = XgboostClassifier(n_estimators=300,learning_rate=0.055,max_depth=3,subsample=.5,n_jobs=8,objective='multi:softprob',eval_metric='mlogloss',booster='gbtree',silent=1)
    #blender = BaggingClassifier(base_estimator=blender,n_estimators=40,n_jobs=1,verbose=2,random_state=None,max_samples=0.96,max_features=0.94,bootstrap=False)
    #blender = XgboostClassifier(n_estimators=200,learning_rate=0.05,max_depth=3,subsample=.5,n_jobs=8,objective='multi:softprob',eval_metric='mlogloss',booster='gbtree',silent=1)
    if not skipCV:
	#blender = CalibratedClassifierCV(baseblender, method='sigmoid', cv=3)
	cv = StratifiedKFold(ly, n_folds=8,shuffle=True)
	#parameters = {'n_estimators':[200,250,300],'max_depth':[3],'learning_rate':[0.055,0.05,0.45],'subsample':[0.5]}#XGB
	#blender=makeGridSearch(blender,oobpreds,ly,n_jobs=2,refit=True,cv=cv,scoring='log_loss',random_iter=-1,parameters=parameters)
	
	blend_scores=np.zeros(len(cv))
	n_classes = 9 #oobpreds.shape[1]/len(ensemble)
	blend_oob=np.zeros((oobpreds.shape[0],n_classes))
	print blender
	for i, (train, test) in enumerate(cv):
	    Xtrain = oobpreds.iloc[train]
	    Xtest = oobpreds.iloc[test]
	    blender.fit(Xtrain, ly[train])	
	    if use_proba:
		blend_oob[test] = blender.predict_proba(Xtest)
	    else:
		print "Warning: Using predict, no proba!"

		blend_oob[test] = blender.predict(Xtest)
	    blend_scores[i]=funcdict[score_func](ly[test],blend_oob[test])
	    print "Fold: %3d <%s>: %0.6f" % (i,score_func,blend_scores[i])
	
	print " <"+score_func+">: %0.6f (+/- %0.4f)" % (blend_scores.mean(), blend_scores.std()),
	oob_auc=funcdict[score_func](ly,blend_oob)
	print " "+score_func+": %0.6f" %(oob_auc)
	
	if hasattr(blender,'coef_'):
	  print "%-16s %10s %10s" %("model",score_func,"coef")
	  idx = 0
	  for i,model in enumerate(ensemble):
	    idx_start = n_classes*i
	    idx_end = n_classes*(i+1)
	    coldata=np.asarray(oobpreds.iloc[:,idx_start:idx_end])
	    score=funcdict[score_func](ly, coldata)
	    print "%-16s %10.3f%10.3f" %(model,score,blender.coef_[0][i])
	  print "sum coef: %4.4f"%(np.sum(blender.coef_))
	#plt.plot(range(len(ensemble)),scores,'ro')
    
    if len(subfile)>1:
	#Prediction
	print "Make final ensemble prediction..."
	#make prediction for each classifiers   
	preds=blender.fit(oobpreds,ly)
	#blend results
	preds=blender.predict_proba(testset)
	#print preds
	makePredictions(blender,testset,filename=subfile)
	
	if not skipCV: plt.hist(blend_oob,bins=50,label='oob')
	plt.hist(preds,bins=50,alpha=0.3,label='pred')
	plt.legend()
	plt.show()
	
    return(blend_scores.mean())



def multiclass_mult(Xtrain,params,n_classes):
    """
    Multiplication rule for multiclass models
    """
    ypred = np.zeros((len(params),Xtrain.shape[0],n_classes))
    for i,p in enumerate(params):
		idx_start = n_classes*i
		idx_end = n_classes*(i+1)
		ypred[i] = Xtrain.iloc[:,idx_start:idx_end]*p
    ypred = np.mean(ypred,axis=0)
    return ypred

def blend_mult(Xtrain,params,n_classes=None):
    if n_classes <2:
	return np.dot(Xtrain,params)
    else: 
	return multiclass_mult(Xtrain,params,n_classes)

def linearBlend_multiclass(ensemble,Xtrain,Xtest,y,score_func='log_loss',greater_is_better=True,use_proba=False,normalize=True,removeZeroModels=-1,takeMean=False,alpha=None,subfile="",plotting=True):
    """
    Blending for multiclass systems
    """
    def fopt(params):
	# nxm  * m*1 ->n*1
	if np.isnan(np.sum(params)):
	    print "We have NaN here!!"
	    score=0.0
	else:
	    #ypred=np.dot(Xtrain,params)	
	    ypred = blend_mult(Xtrain,params,n_classes)
	    #print ypred
	    if not use_proba: ypred = np.round(ypred)
	    #print ypred
	    score=funcdict[score_func](y,ypred)
	    print "score: %8.3f"%(score)
	    #regularization
	    if alpha is not None:
	      penalty=alpha*np.sum(np.square(params))
	      print "orig score:%8.3f"%(score),
	      score=score-penalty
	      print " - Regularization - alpha: %8.3f penalty: %8.3f regularized score: %8.3f"%(alpha,penalty,score)
	    if greater_is_better: score = -1*score
	return score

    y = np.asarray(y)
    n_models=len(ensemble)
    n_classes = Xtrain.shape[1]/len(ensemble)
    #print "Xtrain.shape:",Xtrain.shape[1]
    print "n_classes",n_classes
    
    lowerbound=0.0
    upperbound=0.5
    constr=None
    constr=[lambda x,z=i: x[z]-lowerbound for i in range(n_models)]
    constr2=[lambda x,z=i: upperbound-x[z] for i in range(n_models)]
    constr=constr+constr2
    print n_models
    x0 = np.ones((n_models, 1)) / n_models
    print x0

    #x0= np.random.random_sample((n_models,1))
    
    xopt = fmin_cobyla(fopt, x0,constr,rhoend=1e-7,maxfun=5000)
    
    if takeMean:
	print "Taking the mean..."
	xopt=x0
    
    #normalize coefficient
    if normalize: 
	xopt=xopt/np.sum(xopt)
	print "Normalized coefficients:",xopt
    
    if np.isnan(np.sum(xopt)):
	    print "We have NaN here!!"
    
    ypred=blend_mult(Xtrain,xopt,n_classes)
    if not use_proba: ypred = np.round(ypred)
    
    ymean= blend_mult(Xtrain,x0,n_classes)
    
    oob_score=funcdict[score_func](y,ypred)
    print "->score,opt: %4.4f" %(oob_score)
    score=funcdict[score_func](y,ymean)
    print "->score,mean: %4.4f" %(score)
       
    zero_models=[]
    print "%4s %-48s %6s %6s" %("nr","model","score","coeff")
    for i,model in enumerate(ensemble):
	idx_start = n_classes*i
	idx_end = n_classes*(i+1)
	coldata=np.asarray(Xtrain.iloc[:,idx_start:idx_end])
	score = funcdict[score_func](y,coldata)	
	print "%4d %-48s %6.3f %6.3f" %(i+1,model,score,xopt[i])
	if xopt[i]<removeZeroModels:
	    zero_models.append(model)
    print "##sum coefficients: %4.4f"%(np.sum(xopt))
    
    if removeZeroModels>0.0:
	print "Dropping ",len(zero_models)," columns:",zero_models
	Xtrain=Xtrain.drop(zero_models,axis=1)
	Xtest=Xtest.drop(zero_models,axis=1)
	return (Xtrain,Xtest)
    
    #prediction flatten makes a n-dim row vector from a nx1 column vector...
    preds = blend_mult(Xtest,xopt,n_classes).flatten()
    if not use_proba: preds = np.round(preds).astype(int)
    #if calibration:
	#print "Try to calibrate..."
	#calibrator = IsotonicRegression(ymin=1E-15,ymax=1.0-1E-15,out_of_bounds='clip')
	
    if plotting:
	plt.hist(ypred,bins=50,alpha=0.3,label='oob')
	plt.hist(preds,bins=50,alpha=0.3,label='pred')
	plt.legend()
	plt.show()
    
    if subfile is not None:
	makePredictions(None,Xtest=preds,filename=subfile)
    else:
	return oob_score
      
  
def selectModels(ensemble,startensemble=[],niter=10,mode='amsMaximize',useCols=None): 
    """
    Random mode for best model selection
    """
    randBinList = lambda n: [randint(0,1) for b in range(1,n+1)]
    auc_list=[0.0]
    ens_list=[]
    cols_list=[]
    for i in range(niter):
	print "iteration %5d/%5d, current max_score: %6.3f"%(i+1,niter,max(auc_list))
	actlist=randBinList(len(ensemble))
	actensemble=[x for x in itertools.compress(ensemble,actlist)]
	actensemble=startensemble+actensemble
	print actensemble
	#print actensemble
	auc=trainEnsemble(actensemble,mode=mode,useCols=useCols,addMetaFeatures=False,dropCorrelated=False)
	auc_list.append(auc)
	ens_list.append(actensemble)
	#cols_list.append(actCols)
    maxauc=0.0
    topens=None
    topcols=None
    for ens,auc in zip(ens_list,auc_list):	
	print "SCORE: %4.4f" %(auc),
	print ens
	if auc>maxauc:
	  maxauc=auc
	  topens=ens
	  #topcols=col
    print "\nTOP ensemble:",topens
    print "TOP score: %4.4f" %(maxauc)

def selectModelsGreedy(ensemble,startensemble=[],niter=2,mode='classical',useCols=None,dropCorrelated=False,greater_is_better=False):    
    """
    Select best models in a greedy forward selection
    """
    topensemble=startensemble
    score_list=[]
    ens_list=[]
    if greater_is_better:
	bestscore=0.0
    else:
	bestscore=1E15
    for i in range(niter):
	if greater_is_better:
	    maxscore=0.0
	else:
	    maxscore=1E15
	topidx=-1
	for j in range(len(ensemble)):
	    if ensemble[j] not in topensemble:
		actensemble=topensemble+[ensemble[j]]
	    else:
		continue
	    
	    #score=trainEnsemble(actensemble,mode=mode,useCols=useCols,addMetaFeatures=False,dropCorrelated=dropCorrelated)
	    score=trainEnsemble_multiclass(actensemble,mode=mode,useCols=None,use_proba=True)
	    print "##(Current top score: %4.4f | overall best score: %4.4f) actual score: %4.4f  - " %(maxscore,bestscore,score),
	    print actensemble
	    if greater_is_better:
		if score>maxscore:
		    maxscore=score
		    topidx=j
	    else:
		if score<maxscore:
		    maxscore=score
		    topidx=j
	    
	#pick best set
	#if not maxscore+>bestscore:
	#    print "Not gain in score anymore, leaving..."
	#    break
	topensemble.append(ensemble[topidx])
	print "TOP score: %4.4f" %(maxscore),
	print " - actual ensemble:",topensemble
	score_list.append(maxscore)
	ens_list.append(list(topensemble))
	if greater_is_better:
	  if maxscore>bestscore:
	      bestscore=maxscore
	else:
	  if maxscore<bestscore:
	      bestscore=maxscore
    
    for ens,score in zip(ens_list,score_list):	
	print "SCORE: %4.4f" %(score),
	print ens
	
    plt.plot(score_list)
    plt.show()
    return topensemble

 
def blendSubmissions(fileList,coefList):
    """
    Simple blend dataframes from fileList
    """
    pass
    

if __name__=="__main__":
    #ensemble=createModels()
    #ensemble=createOOBdata_parallel(ensemble,repeats=1,nfolds=5,n_jobs=5,use_proba=False,score_func='quadratic_weighted_kappa') #oob data averaging leads to significant variance reduction
    #ensemble=createOOBdata_parallel(ensemble,repeats=1,nfolds=8,n_jobs=8,score_func='accuracy_score',use_proba=False)#OneVSOne
    all_models=['knn1_r1','svm1_r1','svm2_r1']
    models = ['svm2_r1','svm1_r1']


    useCols=None
    trainEnsemble_multiclass(models,mode='mean',score_func='quadratic_weighted_kappa',useCols=None,addMetaFeatures=False,use_proba=False,dropCorrelated=False,subfile='/home/loschen/Desktop/datamining-kaggle/crowdflower/submissions/sub12062015a.csv')
    #selectModelsGreedy(models,startensemble=['dnn10_r1','bagxgb5_r1','rf1_r1','dnn3_r1','dnn1_r1','xgboost2_r1'],niter=10,mode='classical',greater_is_better=False)
   