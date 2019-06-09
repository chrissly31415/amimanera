#!/usr/bin/python 
# coding: utf-8
#https://github.com/hyperopt/hyperopt/wiki/FMin
#http://fastml.com/optimizing-hyperparams-with-hyperopt/
#https://github.com/zygmuntz/kaggle-burn-cpu/blob/master/driver.py
#https://github.com/hyperopt/hyperopt-sklearn

from hyperopt import fmin, tpe, hp
from qsprLib import *
from home_depot import *
import math

space_xgb = ( 
	#hp.loguniform( 'learning_rate', np.log(0.01), np.log(0.2) ),
	hp.uniform( 'learning_rate',0.01 ,0.05),
	hp.quniform( 'max_depth', 7, 15,1 ),
	hp.uniform( 'subsample', 0.5, 1.0 ),
	hp.uniform( 'colsample_bytree', 0.5, 1.0 ),
	#hp.uniform( 'max_features', 0.9, 1.0 ),
	#hp.uniform( 'max_samples', 0.9, 1.0 ),
	#hp.choice( 'bootstrap', [ False] ),
	hp.uniform( 'gamma', 0.1, 3.0 ),
	hp.quniform( 'min_child_weight', 1, 10,1 )
)

space_nn  = ( 
	hp.quniform( 'hidden1_num_units', 400,600,50),
	hp.quniform( 'hidden2_num_units', 400,600,50 ),
	hp.quniform( 'hidden3_num_units', 250,600,50 ),
	#hp.uniform( 'dropout0_p', 0.0, 0.0),
	hp.uniform( 'dropout1_p', 0.0, 0.25 ),
	hp.uniform( 'dropout2_p', 0.0, 0.25 ),
	hp.uniform( 'dropout3_p', 0.0, 0.25 ),
	hp.quniform( 'max_epochs', 50,150,25),
	hp.loguniform( 'learning_rate', np.log( 1E-4 ), np.log( 1E-2 )),
	hp.loguniform( 'L2_alpha', np.log( 1E-6 ), np.log( 1E-2 )),
	#hp.uniform( 'leakiness', 0.1, 0.3 ),
	#hp.uniform( 'max_features', 0.9, 1.0 ),
	#hp.uniform( 'max_samples', 0.9, 1.0 )
	
)

def func_nn(params):
      global counter
      global X
      global ta

      counter += 1
      print("Iteration:        %d"%(counter))
      s = time()

      #hidden1_num_units,hidden2_num_units,hidden3_num_units,dropout0_p,dropout1_p,dropout2_p,dropout3_p,max_epochs,learning_rate,L2_alpha = params
      #dropout0_p,dropout1_p,dropout2_p,dropout3_p,max_epochs,learning_rate,L2_alpha = params
      #hidden1_num_units,hidden2_num_units,hidden3_num_units,dropout0_p,max_epochs,learning_rate,L2_alpha,max_features = params
      #hidden1_num_units,hidden2_num_units,dropout0_p,dropout1_p,dropout2_p,max_epochs,learning_rate,max_features = params
      #hidden1_num_units,hidden2_num_units,hidden3_num_units,dropout1_p,dropout2_p,dropout3_p,max_epochs,learning_rate,L2_alpha,max_features,max_samples = params
      hidden1_num_units,hidden2_num_units,hidden3_num_units,dropout1_p,dropout2_p,dropout3_p,max_epochs,learning_rate,L2_alpha = params
      dropout0_p=0.0
      #dropout2_p=0.25
      #dropout3_p=0.25

      print("hidden1_num_units:    %6d"% (hidden1_num_units))
      print("hidden2_num_units:    %6d"% (hidden2_num_units))
      print("hidden3_num_units:    %6d"% (hidden3_num_units))
      #print "dropout0_p:          %6.2f"%(dropout0_p)
      print("dropout1_p:          %6.2f"%(dropout1_p))
      print("dropout2_p:          %6.2f"%(dropout2_p))
      print("dropout3_p:          %6.2f"%(dropout3_p))
      print("max_epochs:          %6d"% (max_epochs))
      print("learning_rate:       %6.2e"%(learning_rate))
      print("L2_alpha:            %6.2e"%(L2_alpha))
      #print "leakiness:            %6.2e"%(leakiness)
      input_shape =X.shape[1]
      #input_shape = int(math.floor(X.shape[1]*max_features))
      #print "max_features: 	 %6.4f (%6d)"%(max_features,input_shape)
      #print "max_samples: 	 %6.4f"%(max_samples)

      print(input_shape)
      #input_shape = 271


      basemodel = NeuralNet(layers=[('input', layers.InputLayer),
	#('dropout0', layers.DropoutLayer),
	('hidden1', layers.DenseLayer),
	('dropout1', layers.DropoutLayer),
	('hidden2', layers.DenseLayer),
	('dropout2', layers.DropoutLayer),
	('hidden3', layers.DenseLayer),
	('dropout3', layers.DropoutLayer),
	('output', layers.DenseLayer)],

	input_shape=(None, input_shape),
	#dropout0_p=dropout0_p,

	hidden1_num_units=hidden1_num_units,
	hidden1_nonlinearity=nonlinearities.LeakyRectify(leakiness=0.1),
	dropout1_p=dropout1_p,

	hidden2_num_units=hidden2_num_units,
	hidden2_nonlinearity=nonlinearities.LeakyRectify(leakiness=0.1),
	dropout2_p=dropout2_p,

	hidden3_num_units=hidden3_num_units,
	hidden3_nonlinearity=nonlinearities.LeakyRectify(leakiness=0.1),
	dropout3_p=dropout3_p,

	output_num_units=1,
	output_nonlinearity=None,

	regression=True,
	objective=RMSE,
	objective_alpha=L2_alpha,
	batch_iterator_train=ShuffleBatchIterator(batch_size = 64),

	#update=adagrad,#0.001
	update=rmsprop,
	update_learning_rate=theano.shared(float32(learning_rate)),

	eval_size=0.0,
	verbose=1,
	max_epochs=max_epochs,

	on_epoch_finished=[
		AdjustVariable('update_learning_rate', start=learning_rate, stop=0.00005),
		#EarlyStopping(patience=20),
		],
	)


      model = basemodel
      #model = BaggingRegressor(base_estimator=basemodel,n_estimators=3,n_jobs=1,verbose=2,random_state=None,max_samples=max_samples,max_features=max_features,bootstrap=False)
      score = buildModel(model,X,y,cv=KLabelFolds(pd.Series(ta), n_folds=5, repeats =1),scoring=scoring_func,n_jobs=1,trainFull=False,verbose=True)
      #score =  buildXvalModel(model,X,y,refit=False,cv=KLabelFolds(pd.Series(ta), n_folds=5, repeats =1))


      print(">>score: %6.3f (+/- %6.3f)"%(-1*score.mean(),score.std()))
      print("elapsed: {}s \n".format( int( round( time() - s ))))
      return -1*score.mean()

def func_xgb(params):
      global counter
      global ta

      counter += 1
      print("Iteration:        %d"%(counter))
      s = time()

      learning_rate, max_depth, subsample,colsample_bytree,gamma,min_child_weight = params
      #learning_rate, max_depth, subsample,colsample_bytree,max_features,max_samples,bootstrap = params
      print("learning_rate:    %6.4f"% (learning_rate))
      print("max_depth:        %6.4f" %(max_depth))
      print("subsample:        %6.4f"%(subsample))
      print("colsample_bytree: %6.4f"%(colsample_bytree))
      print("gamma: 		 %6.4f"%(gamma))
      print("min_child_weight: %6.4f"%(min_child_weight))
      #print "max_features: 	 %6.4f"%(max_features)
      #print "max_samples: 	 %6.4f"%(max_samples)
      #print "bootstrap: 	 %6d"%(bootstrap)

      #model = XgboostRegressor(n_estimators=400,learning_rate=learning_rate,max_depth=max_depth,subsample=subsample,colsample_bytree=colsample_bytree,min_child_weight=5,n_jobs=2,objective='reg:linear',eval_metric='rmse',booster='gbtree',eval_size=0.0,silent=1)
      model = XgboostRegressor(n_estimators=4000,learning_rate=learning_rate,max_depth=max_depth,subsample=subsample,colsample_bytree=colsample_bytree,min_child_weight=5,n_jobs=1,objective='reg:linear',eval_metric='rmse',booster='gbtree',eval_size=0.0,silent=1)
      #model = BaggingRegressor(base_estimator=model,n_estimators=3,n_jobs=1,verbose=0,random_state=None,max_samples=max_samples,max_features=max_features,bootstrap=bootstrap)
      score = buildModel(model,X,y,cv=KLabelFolds(pd.Series(ta), n_folds=8, repeats =1),scoring=scoring_func,n_jobs=8,trainFull=False,verbose=True)

      #Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.15, random_state=42)
      #model.fit( Xtrain, ytrain )
      #p = model.predict_proba( Xtest )

      #score = multiclass_log_loss(ytest, p)
      print(">>score: %6.3f (+/- %6.3f)"%(-1*score.mean(),score.std()), end=' ')
      print(" elapsed: {}s \n".format( int( round( time() - s ))))
      return -1*score.mean()

space_keras  = (
	hp.quniform( 'hidden1_num_units', 50,2000,10),
	hp.quniform( 'hidden2_num_units', 50,2000,10),
	#hp.quniform( 'hidden3_num_units', 100,1000,10),
	#hp.quniform( 'hidden4_num_units', 100,1000,10),
	#hp.uniform( 'dropout0_p', 0.0, 0.0),
	hp.uniform( 'dropout1_p', 0.0, 0.5 ),
	hp.uniform( 'dropout2_p', 0.0, 0.5 ),
	#hp.uniform( 'dropout3_p', 0.0, 0.5 ),
	#hp.uniform( 'dropout4_p', 0.0, 0.5 ),
	hp.quniform( 'max_epochs', 20,60,5),
	hp.choice( 'batch_size', [124,256,512] ),
	hp.loguniform( 'learning_rate', np.log( 1E-5 ), np.log( 1E-3 )),
	#hp.choice( 'learning_rate', [0.0001]),
	hp.choice( 'activation', ['tanh']),
	hp.choice('bootstrap',[True,False]),
	hp.uniform('max_features',0.8,1.0)
)

def func_keras(params):
	global counter
	global X,y

	counter += 1

	s = time()

	#hl1,hl2,do1,do2,max_epochs, bs,learning_rate, act = params
	hl1,hl2,do1,do2,max_epochs, bs,learning_rate, act, boot, maxf = params
	#hl1,hl2,hl3,hl4,do1,do2,do3,do4,max_epochs, bs,learning_rate, act = params
	model  = KerasNN(dims=int(X.shape[1]*maxf),nb_classes=1,nb_epoch=int(max_epochs),learning_rate=learning_rate,validation_split=0.0,batch_size=bs,verbose=0,activation=act, layers=[hl1,hl2], dropout=[do1,do2],loss='mse')
	basemodel = Pipeline([('scaler', StandardScaler()), ('nn',model)])
	print(int(X.shape[1]*maxf))
	basemodel = BaggingRegressor(basemodel,n_estimators=5, max_samples=1.0, max_features=int(X.shape[1]*maxf), bootstrap=boot)
	print(basemodel)
	#cv_labels = pd.Series.from_csv('./data/labels_for_cv.csv')
	#cv = LabelKFold(cv_labels, n_folds=8)
	#score = buildModel(basemodel,X,y,cv=cv, scoring=scoring_func, n_jobs=1,trainFull=False,verbose=True)

	basemodel.fit(X,y)
	yval_pred = basemodel.predict(Xval)
	score = root_mean_squared_error(yval,np.clip(yval_pred,1.0,3.0))
	print(" Eval-score: %5.4f"%(score))

	print("Iteration: %d >>score: %6.3f (+/- %6.3f)"%(counter,score.mean(),score.std()))
	print("hidden1_num_units:    %6d"% (hl1))
	print("hidden2_num_units:    %6d"% (hl2))
	#print "hidden3_num_units:    %6d"% (hl3)
	#print "hidden4_num_units:    %6d"% (hl4)
	print("dropout1_p:          %6.2f"%(do1))
	print("dropout2_p:          %6.2f"%(do2))
	#print "dropout3_p:          %6.2f"%(do3)
	#print "dropout4_p:          %6.2f"%(do4)
	print("max_epochs:          %6d"% (max_epochs))
	print("batch_size:          %6d"% (bs))
	print("learning_rate:       %6.2e"%(learning_rate))
	print("activation:       %s"%(act))
	print("bootstrap:       %s"%(boot))
	print("max_features:          %6.2f"%(maxf))
	print("elapsed: {}s \n".format( int( round( time() - s ))))
	print(''.join(['-'] * 60))

	if greater_is_better:
		return -1*score.mean()
	else:
		return score.mean()

counter=0
#stdbuf -o 0 ./hyperopt_driver.py | tee log
np.random.seed(1234)
scoring_func = make_scorer(root_mean_squared_error, greater_is_better = False)
greater_is_better=False
#XGB
#Xtest,X,y,idx,ta,_ =prepareDataset(seed=123,nsamples='shuffle',log1p=True,standardize=None,balance=base_cols+comp_cols+spec_cols,removeRare=10,oneHotenc=['supplier'],createFeatures=True,createSupplierFeatures=['supplier','quantity'],createVerticalFeatures=True,shapeFeatures=True,timeFeatures=True,materialCost=True,removeLowVariance=True,removeSpec=True)
#Xtest,X,y,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,standardize=None,balance=base_cols+comp_cols+spec_cols,bagofwords=None,createFeatures=True,createVerticalFeatures=True,oneHotenc=['supplier'],removeRare=10,removeSpec=True,dropFeatures=None)
#best = fmin(fn=func_xgb,space=space_xgb,algo=tpe.suggest,max_evals=50,rseed=123)

#NN
#Xtest, X, y, _, _, Xval, yval = prepareDataset('./data/store_db1b.h5')
Xtest, X, y, Xval, yval, test_idx, val_idx = prepareAllFeatures()
#X = removeCorrelations(X,threshhold=0.98)

for col in X.columns:
		if X[col].min()>1E-15:
			X[col] = X[col].map(np.log1p)
			Xval[col] = Xval[col].map(np.log1p)

print(X.describe())
#showCorrelations(X)

best = fmin(fn=func_keras,space=space_keras,algo=tpe.suggest,max_evals=50,rseed=123)
#best = fmin(fn=func_nn,space=space_nn,algo=tpe.suggest,max_evals=200,rseed=1234)




