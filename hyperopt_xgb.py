#!/usr/bin/python 
# coding: utf-8


#https://github.com/hyperopt/hyperopt/wiki/FMin
#http://fastml.com/optimizing-hyperparams-with-hyperopt/
#https://github.com/zygmuntz/kaggle-burn-cpu/blob/master/driver.py
#https://github.com/hyperopt/hyperopt-sklearn

from hyperopt import fmin, tpe, hp
from qsprLib import *
from otto import *

"""
space = hp.choice('classifier_type', [
    {
        'type': 'naive_bayes',
    },
    {
        'type': 'svm',
        'C': hp.lognormal('svm_C', 0, 1),
        'kernel': hp.choice('svm_kernel', [
            {'ktype': 'linear'},
            {'ktype': 'RBF', 'width': hp.lognormal('svm_rbf_width', 0, 1)},
            ]),
    },
    {
        'type': 'dtree',
        'criterion': hp.choice('dtree_criterion', ['gini', 'entropy']),
        'max_depth': hp.choice('dtree_max_depth',
            [None, hp.qlognormal('dtree_max_depth_int', 3, 1, 1)]),
        'min_samples_split': hp.qlognormal('dtree_min_samples_split', 2, 1, 1),
    },
    ])

space2 = ( 
	hp.qloguniform( 'l1_dim', log( 10 ), log( 1000 ), 1 ), 
	hp.qloguniform( 'l2_dim', log( 10 ), log( 1000 ), 1 ),
	hp.loguniform( 'learning_rate', log( 1e-5 ), log( 1e-2 )),
	hp.uniform( 'momentum', 0.5, 0.99 ),
	hp.uniform( 'l1_dropout', 0.1, 0.9 ),
	hp.uniform( 'decay_factor', 1 + 1e-3, 1 + 1e-1 )
)


space = ( 
	hp.qloguniform( 'n_hidden', log( 10 ), log( 1000 ), 1 ),
	hp.uniform( 'alpha', 0, 1 ),
	hp.loguniform( 'rbf_width', log( 1e-5 ), log( 100 )),
	hp.choice( 'activation_func', [ 'tanh', 'sine', 'tribas', 'inv_tribas', 'sigmoid', 'hardlim', 'softlim', 'gaussian', 'multiquadric','inv_multiquadric' ] )
)
"""


space_xgb = ( 
	hp.loguniform( 'learning_rate', np.log(0.3), np.log(0.02) ),
	hp.quniform( 'max_depth', 6, 12,1 ),
	hp.uniform( 'subsample', 0.5, 1.0 ),
	hp.uniform( 'colsample_bytree', 0.5, 1.0 ),
	#hp.uniform( 'gamma', 0.1, 3.0 ),
	#hp.quniform( 'min_child_weight', 1, 10,1 ),
	hp.uniform( 'max_features', 0.9, 1.0 ),
	hp.uniform( 'max_samples', 0.9, 1.0 ),
	hp.choice( 'bootstrap', [ False] )
)

	
def func_xgb(params):
	global counter
	
	counter += 1
	print "Iteration:        %d"%(counter)
	s = time()
	
	#learning_rate, max_depth, subsample,colsample_bytree,gamma,min_child_weight = params
	learning_rate, max_depth, subsample,colsample_bytree,max_features,max_samples,bootstrap = params
	print "learning_rate:    %6.4f"% (learning_rate)
	print "max_depth:        %6.4f" %(max_depth)
	print "subsample:        %6.4f"%(subsample)
	print "colsample_bytree: %6.4f"%(colsample_bytree)
	#print "gamma: 		 %6.4f"%(gamma)
	#print "min_child_weight: %6.4f"%(min_child_weight)
	print "max_features: 	 %6.4f"%(max_features)
	print "max_samples: 	 %6.4f"%(max_samples)
	print "bootstrap: 	 %6d"%(bootstrap)
	
	basemodel = XgboostClassifier(n_estimators=200,learning_rate=learning_rate,max_depth=max_depth,subsample=subsample,colsample_bytree=colsample_bytree,n_jobs=4,objective='multi:softprob',eval_metric='mlogloss',booster='gbtree',silent=1)
	model = BaggingClassifier(base_estimator=basemodel,n_estimators=4,n_jobs=1,verbose=2,random_state=None,max_samples=max_samples,max_features=max_features,bootstrap=bootstrap)
	score = buildModel(model,X,y,cv=StratifiedShuffleSplit(y,2,test_size=0.125),scoring=scoring_func,n_jobs=2,trainFull=False,verbose=True)
	
	#Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.15, random_state=42)
	#model.fit( Xtrain, ytrain )
	#p = model.predict_proba( Xtest )

	#score = multiclass_log_loss(ytest, p)
	print ">>score: %6.3f (+/- %6.3f)"%(-1*score.mean(),score.std())
	print "elapsed: {}s \n".format( int( round( time() - s )))
	return -1*score.mean()

scoring_func = make_scorer(multiclass_log_loss, greater_is_better=False, needs_proba=True)
counter=0
(X,y,Xtest,labels) = prepareDataset(nsamples='shuffle',addFeatures=True)
best = fmin(fn=func_xgb,space=space_xgb,algo=tpe.suggest,max_evals=30)
print best

