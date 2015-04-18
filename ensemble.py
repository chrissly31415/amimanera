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
from otto import *


def createModels():
    ensemble=[]
   
    #XGBOOST1 CV~0.45
    #(Xtrain,ytrain,Xtest,labels) = prepareDataset()
    #model = XgboostClassifier(n_estimators=400,learning_rate=0.05,max_depth=10,subsample=.5,n_jobs=1,objective='multi:softprob',eval_metric='mlogloss',booster='gbtree',silent=1)
    #xmodel = XModel("xgboost1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,class_names=sorted(list(set(labels))))
    #ensemble.append(xmodel)

    #RF1 ~ 0.491
    #(Xtrain,ytrain,Xtest,labels) = prepareDataset()
    #basemodel = RandomForestClassifier(n_estimators=500,max_depth=None,min_samples_leaf=1,n_jobs=1,criterion='entropy', max_features=20,oob_score=False)
    #model = CalibratedClassifierCV(basemodel, method='isotonic', cv=3)
    #xmodel = XModel("rf1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,class_names=sorted(list(set(labels))))
    #ensemble.append(xmodel)

    #DNN1 CV~0.485
    #(Xtrain,ytrain,Xtest,labels) = prepareDataset(standardize=True,log_transform=True)
    """
    model = NeuralNet(
    layers=[  # three layers: one hidden layer
        ('input', layers.InputLayer),      
        ('hidden1', layers.DenseLayer),
        ('dropout1', layers.DropoutLayer),
        ('hidden2', layers.DenseLayer),
        ('dropout2', layers.DropoutLayer),
        ('hidden3', layers.DenseLayer),
        ('dropout3', layers.DropoutLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape=(None,Xtrain.shape[1]),  # 96x96 input pixels per batch
    
    hidden1_num_units=300,  # number of units in hidden layer
    hidden1_nonlinearity=nonlinearities.rectify,
    dropout1_p=0.25,
    
    hidden2_num_units=300,
    hidden2_nonlinearity=nonlinearities.rectify,
    dropout2_p=0.1,
    
    hidden3_num_units=300,
    hidden3_nonlinearity=nonlinearities.rectify,
    dropout3_p=0.5,
    
    output_nonlinearity=nonlinearities.softmax,  # output layer uses identity function
    output_num_units=9,  # 30 target values

    eval_size=0.0,
    #regularization=l2,
    batch_iterator_train=BatchIterator(batch_size=1024),
    batch_iterator_test=BatchIterator(batch_size=1024),
    
    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=theano.shared(float32(0.005)),
    update_momentum=theano.shared(float32(0.9)),

    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.002, stop=0.00001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        #EarlyStopping(patience=200),
        ],


    regression=False,  # flag to indicate we're dealing with regression problem
    max_epochs=1000,  # we want to train this many epochs
    verbose=1,
    )
    """
    #xmodel = XModel("dnn1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,class_names=sorted(list(set(labels))))
    #ensemble.append(xmodel)

    #DNN2 CV~0.
    #(Xtrain,ytrain,Xtest,labels) = prepareDataset(standardize=True,log_transform=False)
    """
    model = NeuralNet(
    layers=[  # three layers: one hidden layer
        ('input', layers.InputLayer),      
        ('hidden1', layers.DenseLayer),
        ('dropout1', layers.DropoutLayer),
        ('hidden2', layers.DenseLayer),
        ('dropout2', layers.DropoutLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape=(None,Xtrain.shape[1]),  # 96x96 input pixels per batch
    
    hidden1_num_units=300,  # number of units in hidden layer
    hidden1_nonlinearity=nonlinearities.rectify,
    dropout1_p=0.2,
    
    hidden2_num_units=300,
    hidden2_nonlinearity=nonlinearities.rectify,
    dropout2_p=0.3,
   
    
    output_nonlinearity=nonlinearities.softmax,  # output layer uses identity function
    output_num_units=9,  

    eval_size=0.0,
    #objective=L2Regularization,
    #objective_alpha=0.001,
    batch_iterator_train=BatchIterator(batch_size=2048),
    batch_iterator_test=BatchIterator(batch_size=2048),
    
    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=theano.shared(float32(0.004)),
    update_momentum=theano.shared(float32(0.9)),

    #on_epoch_finished=[
    #    AdjustVariable('update_learning_rate', start=0.002, stop=0.00001),
    #    AdjustVariable('update_momentum', start=0.9, stop=0.999),
    #    #EarlyStopping(patience=200),
    #    ],


    regression=False,  # flag to indicate we're dealing with regression problem
    max_epochs=1000,  # we want to train this many epochs
    verbose=1,
    )
    """
    #xmodel = XModel("dnn1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,class_names=sorted(list(set(labels))))
    #ensemble.append(xmodel)

    #LOGREG1
    #start_set = ['feat_11', 'feat_60', 'feat_34', 'feat_14', 'feat_90', 'feat_15', 'feat_62', 'feat_42', 'feat_39', 'feat_36', 'feat_75', 'feat_68', 'feat_9', 'feat_43', 'feat_40', 'feat_76', 'feat_86', 'feat_26', 'feat_35', 'feat_59', 'feat_47', 'feat_17', 'feat_48', 'feat_69', 'feat_50', 'feat_91', 'feat_92', 'feat_56', 'feat_53', 'feat_25', 'feat_84', 'feat_57', 'feat_78', 'feat_58', 'feat_41', 'feat_32', 'feat_67', 'feat_72', 'feat_77', 'feat_64', 'feat_20', 'feat_71', 'feat_83', 'feat_19', 'feat_23', 'feat_88', 'feat_33', 'feat_73', 'feat_93', 'feat_3', 'feat_81', 'feat_13']
    #(Xtrain,ytrain,Xtest,labels) = prepareDataset(nsamples=-1,standardize=True,polynomialFeatures=2,featureFilter=start_set,addFeatures=False)
    #model = LogisticRegression(C=0.1,class_weight=None,penalty='L1')
    #xmodel = XModel("logreg1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,class_names=sorted(list(set(labels))))
    #ensemble.append(xmodel)
    
    #XGBOOST1 CV~0.45
    #(Xtrain,ytrain,Xtest,labels) = prepareDataset()
    #model = XgboostClassifier(n_estimators=400,learning_rate=0.05,max_depth=10,subsample=.5,n_jobs=1,objective='multi:softprob',eval_metric='mlogloss',booster='gbtree',silent=1)
    #xmodel = XModel("xgboost1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,class_names=sorted(list(set(labels))))
    #ensemble.append(xmodel)
    
    #XGBOOST2 CV~0.46
    #all_features=[u'feat_1', u'feat_2', u'feat_3', u'feat_4', u'feat_5', u'feat_6', u'feat_7', u'feat_8', u'feat_9', u'feat_10', u'feat_11', u'feat_12', u'feat_13', u'feat_14', u'feat_15', u'feat_16', u'feat_17', u'feat_18', u'feat_19', u'feat_20', u'feat_21', u'feat_22', u'feat_23', u'feat_24', u'feat_25', u'feat_26', u'feat_27', u'feat_28', u'feat_29', u'feat_30', u'feat_31', u'feat_32', u'feat_33', u'feat_34', u'feat_35', u'feat_36', u'feat_37', u'feat_38', u'feat_39', u'feat_40', u'feat_41', u'feat_42', u'feat_43', u'feat_44', u'feat_45', u'feat_46', u'feat_47', u'feat_48', u'feat_49', u'feat_50', u'feat_51', u'feat_52', u'feat_53', u'feat_54', u'feat_55', u'feat_56', u'feat_57', u'feat_58', u'feat_59', u'feat_60', u'feat_61', u'feat_62', u'feat_63', u'feat_64', u'feat_65', u'feat_66', u'feat_67', u'feat_68', u'feat_69', u'feat_70', u'feat_71', u'feat_72', u'feat_73', u'feat_74', u'feat_75', u'feat_76', u'feat_77', u'feat_78', u'feat_79', u'feat_80', u'feat_81', u'feat_82', u'feat_83', u'feat_84', u'feat_85', u'feat_86', u'feat_87', u'feat_88', u'feat_89', u'feat_90', u'feat_91', u'feat_92', u'feat_93']
    #addedFeatures_short=[u'arg_max', u'row_sd']
    #(Xtrain,ytrain,Xtest,labels) = prepareDataset(addFeatures=True,final_filter=all_features+addedFeatures_short)
    #model = XgboostClassifier(n_estimators=200,learning_rate=0.1,max_depth=10,subsample=.75,n_jobs=4,objective='multi:softprob',eval_metric='mlogloss',booster='gbtree',silent=1,eval_size=0.0)
    #xmodel = XModel("xgboost1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,class_names=sorted(list(set(labels))))
    #ensemble.append(xmodel)
    
    
    #GBC1 ~0.52
    (Xtrain,ytrain,Xtest,labels) = prepareDataset(nsamples=-1,addFeatures=True)
    model = GradientBoostingClassifier(loss='deviance',n_estimators=300, learning_rate=0.03, max_depth=10,subsample=.5,verbose=1)
    xmodel = XModel("gbc1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,class_names=sorted(list(set(labels))))
    ensemble.append(xmodel)
    
    #SVM1 ~ 0.45!!->0.64!!!!FAULTY???
    #start_set = ['feat_11', 'feat_60', 'feat_34', 'feat_14', 'feat_90', 'feat_15', 'feat_62', 'feat_42', 'feat_39', 'feat_36', 'feat_75', 'feat_68', 'feat_9', 'feat_43', 'feat_40', 'feat_76', 'feat_86', 'feat_26', 'feat_35', 'feat_59', 'feat_47', 'feat_17', 'feat_48', 'feat_69', 'feat_50', 'feat_91', 'feat_92', 'feat_56', 'feat_53', 'feat_25', 'feat_84', 'feat_57', 'feat_78', 'feat_58', 'feat_41', 'feat_32', 'feat_67', 'feat_72', 'feat_77', 'feat_64', 'feat_20', 'feat_71', 'feat_83', 'feat_19', 'feat_23', 'feat_88', 'feat_33', 'feat_73', 'feat_93', 'feat_3', 'feat_81', 'feat_13']
    #(Xtrain,ytrain,Xtest,labels) = prepareDataset(nsamples=45000,featureFilter=start_set,addFeatures=False)
    #model = SVC(kernel='rbf',C=10.0, gamma=0.0, verbose = 0, probability=True)
    #xmodel = XModel("svm1_r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,class_names=sorted(list(set(labels))))
    #ensemble.append(xmodel)    
    
    #bagxgb1
    #(Xtrain,ytrain,Xtest,labels) = prepareDataset(nsamples=-1,addFeatures=False)
    #basemodel = XgboostClassifier(n_estimators=400,learning_rate=0.05,max_depth=10,subsample=.5,n_jobs=1,objective='multi:softprob',eval_metric='mlogloss',booster='gbtree',silent=1)
    #model = BaggingClassifier(base_estimator=basemodel,n_estimators=20,n_jobs=1,verbose=1)
    #xmodel = XModel("bagxgb1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,class_names=sorted(list(set(labels))))
    #ensemble.append(xmodel)
    
    
    #some info
    for m in ensemble:
	m.summary()
    return(ensemble)

    
def finalizeModel(m,binarizeProbas=False):
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
	    m.preds = m.classifier.predict_proba(m.Xtest)	    
	    ypred = m.preds
	    
	m.summary()	
	
	#put data to data.frame and save
	#OOB DATA
	m.oob_preds=pd.DataFrame(np.asarray(m.oob_preds))
		
	#TESTSET prediction	
	m.preds=pd.DataFrame(np.asarray(m.preds))
	#save final model
	#TESTSETscores
	#OOBDATA
	allpred = pd.concat([m.preds, m.oob_preds])
	#print allpred
	#submission data is first, train data is last!
	filename="/home/loschen/Desktop/datamining-kaggle/otto/data/"+m.name+".csv"
	print "Saving oob + predictions as csv to:",filename
	allpred.to_csv(filename,index=False)
	
	#XModel.saveModel(m,"/home/loschen/Desktop/datamining-kaggle/higgs/data/"+m.name+".pkl")
	XModel.saveCoreData(m,"/home/loschen/Desktop/datamining-kaggle/otto/data/"+m.name+".pkl")
	return(m)
    

def createOOBdata_parallel(ensemble,repeats=2,nfolds=8,n_jobs=1,score_func='log_loss',verbose=False,calibrate=False,binarize=False):
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
	    
	    #parallel run
	    oob_pred = parallel(delayed(fit_and_score)(clone(m.classifier), m.Xtrain, ly, train, test,use_proba=True)
			  for train, test in cv)
	    
	    for i,(train,test) in enumerate(cv):
		oob_preds[test,:,j] = oob_pred[i]
		
		scores[i]=funcdict[score_func](ly[test],oob_preds[test,:,j])
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
	
	m=finalizeModel(m)
	
    return(ensemble)
    
def fit_and_score(xmodel,X,y,train,test,sample_weight=None,scale_wt=None,use_proba=False):
    """
    Score function for parallel oob creation
    """
    if isinstance(X,pd.DataFrame): 
	Xtrain = X.iloc[train]
	Xvalid = X.iloc[test]
    else:
	Xtrain = X[train]
	Xvalid = X[test]
	
    ytrain = y[train]
    
    if sample_weight is not None: 
	wtrain = sample_weight[train]
	xmodel.fit(Xtrain,ytrain,sample_weight=wtrain)
    else:
	xmodel.fit(Xtrain,ytrain)
    
    if use_proba:
	#saving out-of-bag predictions
	  local_pred = xmodel.predict_proba(Xvalid)
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
       
    return local_pred

def trainEnsemble_multiclass(ensemble,mode='linear',useCols=None,addMetaFeatures=False,use_proba=True,dropCorrelated=False,subfile=""):
    """
    Train the ensemble
    """
    basedir="/home/loschen/Desktop/datamining-kaggle/otto/data/"

    for i,model in enumerate(ensemble):
	
	print "Loading model:",i," name:",model
	xmodel = XModel.loadModel(basedir+model)
	class_names = xmodel.class_names
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
	results=classicalBlend(ensemble,Xtrain,Xtest,y,subfile=subfile)
    elif mode is 'mean':
	results=linearBlend_multiclass(ensemble,Xtrain,Xtest,y,takeMean=True,subfile=subfile)
    else:
	results=linearBlend_multiclass(ensemble,Xtrain,Xtest,y,takeMean=False,subfile=subfile)
    return(results)

def trainEnsemble(ensemble,mode='classical',useCols=None,addMetaFeatures=False,use_proba=True,dropCorrelated=False,subfile=""):
    """
    Train the ensemble
    """
    basedir="/home/loschen/Desktop/datamining-kaggle/otto/data/"
    xensemble=[]
    for i,model in enumerate(ensemble):
	print "Loading model:",i," name:",model
	xmodel = XModel.loadModel(basedir+model)
	print "OOB data     :",xmodel.oob_preds.shape
	print "y data       :",xmodel.preds.shape
	if i>0:
	    Xtrain = pd.concat([Xtrain,xmodel.oob_preds], axis=1)
	    Xtest = pd.concat([Xtest,xmodel.preds], axis=1)
	    
	else:
	    #print type(xmodel.oob_preds)
	    #print xmodel.oob_preds
	    #Xall = pd.read_csv("/home/loschen/Desktop/datamining-kaggle/higgs/data/"+model+".csv", sep=",")[col]
	    Xtrain = xmodel.oob_preds
	    Xtest = xmodel.preds
	xensemble.append(xmodel)

    y = xensemble[0].ytrain
    Xtrain.columns=ensemble
    #print Xtrain.head(10)
    #print Xtrain.describe()
    
    Xtest.columns=ensemble
    #print Xtest.head(10)
    #print Xtest.describe()
    
    #scale data
    X_all=pd.concat([Xtest,Xtrain])   
    if dropCorrelated: X_all=removeCorrelations(X_all,0.995)
    #X_all = (X_all - X_all.min()) / (X_all.max() - X_all.min())
    
    Xtrain=X_all[Xtest.shape[0]:]
    Xtest=X_all[:Xtest.shape[0]]
    
    if mode is 'classical':
	results=classicalBlend(ensemble,Xtrain,Xtest,y,subfile=subfile)
    elif mode is 'mean':
	results=linearBlend(ensemble,Xtrain,Xtest,y,takeMean=True,subfile=subfile)
    else:
	results=linearBlend(ensemble,Xtrain,Xtest,y,takeMean=False,subfile=subfile)
    return(results)
    
    
def voting(ensemble,Xtrain,Xtest,y,test_indices,subfile):
    """
    Voting for simple classifiction result
    """
    
    vfunc = np.vectorize(binarizeProbs)
    
    print "Majority voting for predictions"
    #weights=np.asarray(pd.read_csv('../datamining-kaggle/higgs/training.csv', sep=",", na_values=['?'], index_col=0)['Weight'])
    
    for i,col in enumerate(Xtrain.columns):
	cutoff = computeCutoff(Xtrain[col].values,False)
	#label=col+"_"+str(i)
	Xtrain[col]=vfunc(Xtrain[col].values,cutoff)
	#oob_avg=np.mean(Xtmp,axis=1)
	score=ams_score(y,Xtrain[col].values,sample_weight=weights,use_proba=False,cutoff=cutoff,verbose=False)
	print "%4d AMS,oob data: %0.4f colum: %20s" % (i,score,col)
	cutoff = computeCutoff(Xtest[col].values,False)
	Xtest[col]=vfunc(Xtest[col].values,cutoff)
	#plt.hist(np.asarray(Xtrain[col]),bins=50,label=col+'_oob')
	#plt.hist(np.asarray(Xtest[col]),bins=50,label=col+'pred',alpha=0.2)
	#plt.legend()
	#plt.show()
	#del Xtrain[col]
    
    
    oob_avg=np.asarray(np.mean(Xtrain,axis=1))
    score=ams_score(y,oob_avg,sample_weight=weights,use_proba=False,cutoff=0.5,verbose=True)
    print " AMS,oob all: %0.4f" % (score)
    
    preds=np.asarray(np.mean(Xtest,axis=1))
    
    plt.hist(np.asarray(oob_avg),bins=30,label='oob')
    plt.hist(preds,bins=50,label='pred',alpha=0.2)
    plt.legend()
    plt.show()
    
    if len(subfile)>1:
	Xtest.index=test_indices
	makePredictions(preds,Xtest,subfile,useProba=True,cutoff=0.5)
	checksubmission(subfile)
    

def classicalBlend(ensemble,oobpreds,testset,ly,use_proba=True,score_func='log_loss',subfile=""):
    """
    Blending using sklearn classifier
    """
    folds=8  
    #blender=LogisticRegression(penalty='l2', tol=0.0001, C=100)
    #blender = Pipeline([('filter', SelectPercentile(f_regression, percentile=25)), ('model', LogisticRegression(penalty='l2', tol=0.0001, C=0.1))])
    #blender=SGDClassifier(alpha=0.1, n_iter=50,penalty='l2',loss='log',n_jobs=folds)
    #blender=AdaBoostClassifier(learning_rate=0.01,n_estimators=50)
    blender=RandomForestClassifier(n_estimators=500,n_jobs=4, max_features='auto',oob_score=False,min_samples_leaf=10,max_depth=None)
    blender = CalibratedClassifierCV(blender, method='isotonic', cv=3)
    #blender=ExtraTreesClassifier(n_estimators=500,max_depth=None,min_samples_leaf=5,n_jobs=4,criterion='entropy', max_features='auto',oob_score=False)
    #blender=RandomForestClassifier(n_estimators=500,max_depth=None,min_samples_leaf=10,n_jobs=1,criterion='entropy', max_features=5,oob_score=False)
    #blender=ExtraTreesRegressor(n_estimators=500,max_depth=None)
    #cv = KFold(oobpreds.shape[0], n_folds=folds,random_state=123)
    #cv = StratifiedShuffleSplit(ly, n_iter=folds, test_size=0.5)
    cv = StratifiedKFold(ly, folds,shuffle=True)
    blend_scores=np.zeros(folds)
    n_classes = oobpreds.shape[1]/len(ensemble)
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
	plt.hist(blend_oob,bins=50,label='oob')
	plt.hist(preds,bins=50,alpha=0.3,label='pred')
	plt.legend()
	plt.show()	
	makePredictions(blender,testset,filename=subfile)
	
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

def linearBlend_multiclass(ensemble,Xtrain,Xtest,y,score_func='log_loss',normalize=True,removeZeroModels=-1,takeMean=False,alpha=None,subfile="",plotting=False):
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
	    ypred = multiclass_mult(Xtrain,params,n_classes)
	    score=funcdict[score_func](y,ypred)
	    #regularization
	    if alpha is not None:
	      penalty=alpha*np.sum(np.square(params))
	      print "orig score:%8.3f"%(score),
	      score=score-penalty
	      print " - Regularization - alpha: %8.3f penalty: %8.3f regularized score: %8.3f"%(alpha,penalty,score) 
	return score

    y = np.asarray(y)
    n_models=len(ensemble)
    n_classes = Xtrain.shape[1]/len(ensemble)
    
    lowerbound=0.0
    upperbound=0.4
    constr=None
    constr=[lambda x,z=i: x[z]-lowerbound for i in range(n_models)]
    constr2=[lambda x,z=i: upperbound-x[z] for i in range(n_models)]
    constr=constr+constr2
    print n_models
    x0 = np.ones((n_models, 1)) / n_models

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
    
    ypred=multiclass_mult(Xtrain,xopt,n_classes)
    
    ymean= multiclass_mult(Xtrain,x0,n_classes)
    
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
    preds = multiclass_mult(Xtest,xopt,n_classes)

    if plotting:
	plt.hist(ypred,bins=50,alpha=0.3,label='oob')
	plt.hist(preds,bins=50,alpha=0.3,label='pred')
	plt.legend()
	plt.show()
    
    
    makePredictions(None,preds,filename=subfile)


def linearBlend(ensemble,Xtrain,Xtest,y,weights=None,score_func='rmse',test_indices=None,normalize=True,takeMean=False,removeZeroModels=0.0,alpha=None,subfile="",plotting=False):

    def fopt(params):
	# nxm  * m*1 ->n*1
	if np.isnan(np.sum(params)):
	    print "We have NaN here!!"
	    score=0.0
	else:
	    ypred=np.dot(Xtrain,params).flatten()
	    score=funcdict[score_func](ypred,y)
	    #score=funcdict['mae'](ypred,y.values)
	    
	    #regularization
	    if alpha is not None:
	      penalty=alpha*np.sum(np.square(params))
	      print "orig score:%8.3f"%(score),
	      score=score-penalty
	      print " - Regularization - alpha: %8.3f penalty: %8.3f regularized score: %8.3f"%(alpha,penalty,score) 
	return score

    y = np.asarray(y)
    lowerbound=0.0
    upperbound=0.3
    constr=[lambda x,z=i: x[z]-lowerbound for i in range(len(Xtrain.columns))]
    constr2=[lambda x,z=i: upperbound-x[z] for i in range(len(Xtrain.columns))]
    constr=constr+constr2
    
    n_models=len(Xtrain.columns)
    x0 = np.ones((n_models, 1)) / n_models
    #x0= np.random.random_sample((n_models,1))
    
    xopt = fmin_cobyla(fopt, x0,constr,rhoend=1e-7,maxfun=5000)
    
    if takeMean:
	print "Taking the mean..."
	xopt=x0
    
    #normalize coefficient
    if normalize: xopt=xopt/np.sum(xopt)
    
    if np.isnan(np.sum(xopt)):
	    print "We have NaN here!!"
    
    ypred=np.dot(Xtrain,xopt).flatten()
    #ypred = ypred/np.max(ypred)#normalize?
    
    ymean= np.dot(Xtrain,x0).flatten()
    #ymean = ymean/np.max(ymean)#normalize?
    
    oob_score=funcdict[score_func](y,ypred)
    print "->score,opt: %4.4f" %(oob_score)
    score=funcdict[score_func](y,ymean)
    print "->score,mean: %4.4f" %(score)
    
    
    zero_models=[]
    print "%4s %-48s %6s %6s" %("nr","model","score","coeff")
    for i,model in enumerate(Xtrain.columns):
	coldata=np.asarray(Xtrain.iloc[:,i])
	score = funcdict[score_func](y,coldata)	
	print "%4d %-48s %6.3f %6.3f" %(i+1,model,score,xopt[i])
	if xopt[i]<removeZeroModels:
	    zero_models.append(model)
    if not normalize: print "##sum coefficients: %4.4f"%(np.sum(xopt))
    
    if removeZeroModels>0.0:
	print "Dropping ",len(zero_models)," columns:",zero_models
	Xtrain=Xtrain.drop(zero_models,axis=1)
	Xtest=Xtest.drop(zero_models,axis=1)
	return (Xtrain,Xtest)
    
    #prediction flatten makes a n-dim row vector from a nx1 column vector...
    preds=np.dot(Xtest,xopt).flatten()

    if plotting:
	plt.hist(ypred,bins=50,alpha=0.3,label='oob')
	plt.hist(preds,bins=50,alpha=0.3,label='pred')
	plt.legend()
	plt.show()
    
    #return dataframes with blending results
    Xtrain['blend']=ypred
    Xtest['blend']=preds
    Xtrain['ytrain']=y

    return(Xtrain,Xtest)
  
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

def selectModelsGreedy(ensemble,startensemble=[],niter=2,mode='classical',useCols=None,dropCorrelated=False):    
    """
    Select best models in a greedy forward selection
    """
    topensemble=startensemble
    score_list=[]
    ens_list=[]
    bestscore=0.0
    for i in range(niter):
	maxscore=0.0
	topidx=-1
	for j in range(len(ensemble)):
	    if ensemble[j] not in topensemble:
		actensemble=topensemble+[ensemble[j]]
	    else:
		continue
	    
	    auc=trainEnsemble(actensemble,mode=mode,useCols=useCols,addMetaFeatures=False,dropCorrelated=dropCorrelated)
	    print "##(Current top score: %4.4f | overall best score: %4.4f) actual score: %4.4f  - " %(maxscore,bestscore,auc),
	    print actensemble
	    if auc>maxscore:
		maxscore=auc
		topidx=j
	#pick best set
	if not maxscore+0.05>bestscore:
	    print "Not gain in score anymore, leaving..."
	    break
	topensemble.append(ensemble[topidx])
	print "TOP score: %4.4trainEnsemblef" %(maxscore),
	print " - actual ensemble:",topensemble
	score_list.append(maxscore)
	ens_list.append(list(topensemble))
	if maxscore>bestscore:
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
    np.random.seed(123)
    ensemble=createModels()
    ensemble=createOOBdata_parallel(ensemble,repeats=1,nfolds=8,n_jobs=8) #oob data averaging leads to significant variance reduction
    
    models=['bagxgb1_r1','xgboost1_r3','dnn1_r3','rf1_r3','logreg1_r3','onehot1_r3','gbc1_r1']
    #models=['bagxgb1_r1']
    #useCols=['A']
    useCols=None
    #trainEnsemble_multiclass(models,mode='classical',useCols=None,addMetaFeatures=False,use_proba=True,dropCorrelated=False,subfile='/home/loschen/Desktop/datamining-kaggle/otto/submissions/submission14042015c.csv')
   
    #trainEnsemble(model,mode='classical',useCols=useCols,addMetaFeatures=False,use_proba=True,dropCorrelated=False,subfile='/home/loschen/Desktop/datamining-kaggle/higgs/submissions/sub1509b.csv')
    