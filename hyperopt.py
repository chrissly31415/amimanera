#!/usr/bin/python 
# coding: utf-8
#https://github.com/hyperopt/hyperopt/wiki/FMin
#http://fastml.com/optimizing-hyperparams-with-hyperopt/
#https://github.com/zygmuntz/kaggle-burn-cpu/blob/master/driver.py
#https://github.com/hyperopt/hyperopt-sklearn

from hyperopt import fmin, tpe, hp
from qsprLib import *
from otto import *
import math

space_xgb = ( 
	hp.loguniform( 'learning_rate', np.log(0.01), np.log(0.2) ),
	hp.quniform( 'max_depth', 8, 12,1 ),
	hp.uniform( 'subsample', 0.5, 1.0 ),
	hp.uniform( 'colsample_bytree', 0.5, 1.0 ),
	#hp.uniform( 'max_features', 0.9, 1.0 ),
	#hp.uniform( 'max_samples', 0.9, 1.0 ),
	#hp.choice( 'bootstrap', [ False] ),
	hp.uniform( 'gamma', 0.1, 3.0 ),
	hp.quniform( 'min_child_weight', 1, 10,1 )
)

space_nn  = ( 
	hp.quniform( 'hidden1_num_units', 600, 1000,50 ),
	hp.quniform( 'hidden2_num_units', 500,800,50 ),
	hp.quniform( 'hidden3_num_units', 250,500,50 ),
	hp.uniform( 'dropout0_p', 0.05, 0.2),
	#hp.uniform( 'dropout1_p', 0.1, 0.5 ),
	#hp.uniform( 'dropout2_p', 0.1, 0.5 ),
	#hp.uniform( 'dropout3_p', 0.1, 0.5 ),
	hp.quniform( 'max_epochs', 50,150,50),
	hp.loguniform( 'learning_rate', np.log( 0.01 ), np.log( 0.03 )),
	hp.loguniform( 'L2_alpha', np.log( 1E-7 ), np.log( 1E-4 )),
	hp.uniform( 'max_features', 0.9, 1.0 )
)


def func_nn(params):
      global counter
      
      counter += 1
      print "Iteration:        %d"%(counter)
      s = time()
      
      #hidden1_num_units,hidden2_num_units,hidden3_num_units,dropout0_p,dropout1_p,dropout2_p,dropout3_p,max_epochs,learning_rate,L2_alpha = params
      #dropout0_p,dropout1_p,dropout2_p,dropout3_p,max_epochs,learning_rate,L2_alpha = params
      hidden1_num_units,hidden2_num_units,hidden3_num_units,dropout0_p,max_epochs,learning_rate,L2_alpha,max_features = params
      dropout1_p=0.5
      dropout2_p=0.25
      dropout3_p=0.25
      
      print "hidden1_num_units:    %6d"% (hidden1_num_units)
      print "hidden2_num_units:    %6d"% (hidden2_num_units)
      print "hidden3_num_units:    %6d"% (hidden3_num_units)
      print "dropout0_p:          %6.2f"%(dropout0_p)
      print "dropout1_p:          %6.2f"%(dropout1_p)
      print "dropout2_p:          %6.2f"%(dropout2_p)
      print "dropout3_p:          %6.2f"%(dropout3_p)
      print "max_epochs:          %6d"% (max_epochs)
      print "learning_rate:       %6.2e"%(learning_rate)
      print "L2_alpha:            %6.2e"%(L2_alpha)
      print "max_features: 	 %6.4f (%6d)"%(max_features,math.floor(98*max_features))
      

      basemodel = NeuralNet(layers=[('input', layers.InputLayer),#0.464
      ('dropout0', layers.DropoutLayer),
      ('hidden1', layers.DenseLayer),
      ('dropout1', layers.DropoutLayer),
      ('hidden2', layers.DenseLayer),
      ('dropout2', layers.DropoutLayer), 
      ('hidden3', layers.DenseLayer),
      ('dropout3', layers.DropoutLayer), 
      ('output', layers.DenseLayer)],

      input_shape=(None, math.floor(98*max_features)),
      dropout0_p=dropout0_p,

      hidden1_num_units=hidden1_num_units,
      hidden1_nonlinearity=nonlinearities.rectify,
      dropout1_p=dropout1_p,

      hidden2_num_units=hidden2_num_units,
      hidden2_nonlinearity=nonlinearities.rectify,
      dropout2_p=dropout2_p,

      hidden3_num_units=hidden3_num_units,
      hidden3_nonlinearity=nonlinearities.rectify,
      dropout3_p=dropout3_p,

      output_num_units=9,
      output_nonlinearity=nonlinearities.softmax,

      objective=L2Regularization,
      objective_alpha=float32(1E-6),

      update=adagrad,
      update_learning_rate=theano.shared(float32(learning_rate)),

      eval_size=0.0,
      verbose=0,
      max_epochs=100,

      on_epoch_finished=[
	      AdjustVariable('update_learning_rate', start=learning_rate, stop=0.001),
	      #EarlyStopping(patience=20),
	      ],
      )
      model = BaggingClassifier(base_estimator=basemodel,n_estimators=3,n_jobs=1,verbose=0,random_state=None,max_samples=1.0,max_features=max_features,bootstrap=False)
      score = buildModel(model,X.values,y,cv=StratifiedShuffleSplit(y,3,test_size=0.15),scoring=scoring_func,n_jobs=1,trainFull=False,verbose=True)
      
      print ">>score: %6.3f (+/- %6.3f)"%(-1*score.mean(),score.std())
      print "elapsed: {}s \n".format( int( round( time() - s )))
      return -1*score.mean()
	
def func_xgb(params):
      global counter
      
      counter += 1
      print "Iteration:        %d"%(counter)
      s = time()
      
      learning_rate, max_depth, subsample,colsample_bytree,gamma,min_child_weight = params
      #learning_rate, max_depth, subsample,colsample_bytree,max_features,max_samples,bootstrap = params
      print "learning_rate:    %6.4f"% (learning_rate)
      print "max_depth:        %6.4f" %(max_depth)
      print "subsample:        %6.4f"%(subsample)
      print "colsample_bytree: %6.4f"%(colsample_bytree)
      print "gamma: 		 %6.4f"%(gamma)
      print "min_child_weight: %6.4f"%(min_child_weight)
      #print "max_features: 	 %6.4f"%(max_features)
      #print "max_samples: 	 %6.4f"%(max_samples)
      #print "bootstrap: 	 %6d"%(bootstrap)
      
      model = XgboostClassifier(n_estimators=500,learning_rate=learning_rate,max_depth=max_depth,subsample=subsample,colsample_bytree=colsample_bytree,gamma=gamma,min_child_weight=min_child_weight,n_jobs=4,objective='multi:softprob',eval_metric='mlogloss',booster='gbtree',silent=1)
      #model = BaggingClassifier(base_estimator=basemodel,n_estimators=3,n_jobs=1,verbose=1,random_state=None,max_samples=max_samples,max_features=max_features,bootstrap=bootstrap)
      score = buildModel(model,X,y,cv=StratifiedShuffleSplit(y,2,test_size=0.15),scoring=scoring_func,n_jobs=2,trainFull=False,verbose=True)
      
      #Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.15, random_state=42)
      #model.fit( Xtrain, ytrain )
      #p = model.predict_proba( Xtest )

      #score = multiclass_log_loss(ytest, p)
      print ">>score: %6.3f (+/- %6.3f)"%(-1*score.mean(),score.std())
      print "elapsed: {}s \n".format( int( round( time() - s )))
      return -1*score.mean()



scoring_func = make_scorer(multiclass_log_loss, greater_is_better=False, needs_proba=True)
counter=0
#(X,y,Xtest,labels) = prepareDataset(nsamples='shuffle',addFeatures=True)
all_features=[u'feat_1', u'feat_2', u'feat_3', u'feat_4', u'feat_5', u'feat_6', u'feat_7', u'feat_8', u'feat_9', u'feat_10', u'feat_11', u'feat_12', u'feat_13', u'feat_14', u'feat_15', u'feat_16', u'feat_17', u'feat_18', u'feat_19', u'feat_20', u'feat_21', u'feat_22', u'feat_23', u'feat_24', u'feat_25', u'feat_26', u'feat_27', u'feat_28', u'feat_29', u'feat_30', u'feat_31', u'feat_32', u'feat_33', u'feat_34', u'feat_35', u'feat_36', u'feat_37', u'feat_38', u'feat_39', u'feat_40', u'feat_41', u'feat_42', u'feat_43', u'feat_44', u'feat_45', u'feat_46', u'feat_47', u'feat_48', u'feat_49', u'feat_50', u'feat_51', u'feat_52', u'feat_53', u'feat_54', u'feat_55', u'feat_56', u'feat_57', u'feat_58', u'feat_59', u'feat_60', u'feat_61', u'feat_62', u'feat_63', u'feat_64', u'feat_65', u'feat_66', u'feat_67', u'feat_68', u'feat_69', u'feat_70', u'feat_71', u'feat_72', u'feat_73', u'feat_74', u'feat_75', u'feat_76', u'feat_77', u'feat_78', u'feat_79', u'feat_80', u'feat_81', u'feat_82', u'feat_83', u'feat_84', u'feat_85', u'feat_86', u'feat_87', u'feat_88', u'feat_89', u'feat_90', u'feat_91', u'feat_92', u'feat_93']
(X,y,Xtest,labels) = prepareDataset(nsamples='shuffle',standardize=True,log_transform=True,addFeatures=True,doSVD=None,final_filter=None)
best = fmin(fn=func_xgb,space=space_xgb,algo=tpe.suggest,max_evals=50,rseed=123)
#best = fmin(fn=func_nn,space=space_nn,algo=tpe.suggest,max_evals=30,rseed=123)
print best

