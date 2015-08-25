#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""  
Ensemble helper tools

Chrissly31415
October,September 2014

using stacking for ensemble building
for stacking versus blending: see:

http://mlwave.com/kaggle-ensembling-guide/


"""

from FullModel import *
import itertools
from scipy.optimize import fmin,fmin_cobyla,minimize

from random import randint
import sys
from sklearn.externals.joblib import Parallel, delayed, logger
from sklearn.base import clone
from cater import *

from sklearn import preprocessing


def createModels_stage2(models_stage1):
    "Ensembling 2nd stage"
    ensemble_stage2=[]
    
    #RIDGE 
    #Xtest,Xtrain,ytrain,idx,ta,_ = trainEnsemble(models_stage1,mode='oob',score_func='rmse')
    #model = Ridge()
    #xmodel = XModel("ridge_l2",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta,bag_mode=True)
    #ensemble_stage2.append(xmodel)
    
    #XRF
    #Xtest,Xtrain,ytrain,idx,ta,_ = trainEnsemble(models_stage1,mode='oob',score_func='rmse')
    #model = ExtraTreesRegressor(n_estimators=50,max_depth=None,min_samples_leaf=5,n_jobs=1,criterion='mse', max_features=18,oob_score=False)
    #xmodel = XModel("xrf_l2",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta,bag_mode=True)
    #ensemble_stage2.append(xmodel)
    
    #XGB
    #Xtest,Xtrain,ytrain,idx,ta,_ = trainEnsemble(models_stage1,mode='oob',score_func='rmse')
    #model = XgboostRegressor(n_estimators=200,learning_rate=0.03,max_depth=2,subsample=.5,n_jobs=4,min_child_weight=1,objective='reg:linear',eval_metric='rmse',booster='gbtree',silent=1)
    #xmodel = XModel("xgb_l2",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta,bag_mode=True)
    #ensemble_stage2.append(xmodel)
    
    #BAGGEDNET
    """
    Xtest,Xtrain,ytrain,idx,ta,_ = trainEnsemble(models_stage1,mode='oob',score_func='rmse')
    model = NeuralNet(layers=[('input', layers.InputLayer),
	('dropout0', layers.DropoutLayer),
	('hidden1', layers.DenseLayer),
	('dropout1', layers.DropoutLayer),
	('hidden2', layers.DenseLayer),
	('dropout2', layers.DropoutLayer),
	('output', layers.DenseLayer)],

	input_shape=(None, Xtrain.shape[1]),
	dropout0_p=0.0,

	hidden1_num_units=128,
	hidden1_nonlinearity=nonlinearities.rectify,
	dropout1_p=0.0,

	hidden2_num_units=128,
	hidden2_nonlinearity=nonlinearities.rectify,
	dropout2_p=0.0,

	output_num_units=1,
	output_nonlinearity=None,

	regression=True,
	objective=RMSE,
	objective_alpha=0.002,
	batch_iterator_train=ShuffleBatchIterator(batch_size = 32),

	#update=adagrad,#0.001
	update=rmsprop,
	update_learning_rate=theano.shared(float32(0.002)),
	
	eval_size=0.0,
	verbose=1,
	max_epochs=75,

	on_epoch_finished=[
		AdjustVariable('update_learning_rate', start=0.002, stop=0.0005),
		#EarlyStopping(patience=20),
		],
    ) 
    model = Pipeline([('scaler', StandardScaler()), ('model',model)])
    model = BaggingRegressor(base_estimator=model,n_estimators=25,n_jobs=1,verbose=2,random_state=None,max_samples=1.0,max_features=1.0,bootstrap=False)
    xmodel = XModel("nn3_l2",classifier=model,Xtrain=Xtrain.values,Xtest=Xtest,ytrain=ytrain,cv_labels=ta,bag_mode=True)
    ensemble_stage2.append(xmodel)
    """
    
    for m in ensemble_stage2:
	m.summary()
    return(ensemble_stage2)

def createModels():
    categoricals = [ 'supplier', 'material_id', 'end_a_1x', 'end_a_2x', 'end_x_1x', 'end_x_2x', 'end_a', 'end_x', 'component_id_1', 'component_id_2', 'component_id_3', 'component_id_4', 'component_id_5', 'component_id_6', 'component_id_7', 'component_id_8', 'spec1', 'spec2', 'spec3', 'spec4', 'spec5', 'spec6', 'spec7', 'spec8']
    categoricals_nospec = [ 'supplier', 'material_id', 'end_a_1x', 'end_a_2x', 'end_x_1x', 'end_x_2x', 'end_a', 'end_x', 'component_id_1', 'component_id_2', 'component_id_3', 'component_id_4', 'component_id_5', 'component_id_6', 'component_id_7', 'component_id_8']
    categoricals_small = [ 'supplier', 'material_id', 'end_a_1x', 'end_a_2x', 'end_x_1x', 'end_x_2x', 'end_a', 'end_x']
    base_cols = ['supplier', 'bracket_pricing', 'material_id', 'end_a_1x', 'end_a_2x', 'end_x_1x', 'end_x_2x', 'end_a', 'end_x']
    comp_cols = ['component_id_1', 'component_id_2', 'component_id_3', 'component_id_4', 'component_id_5', 'component_id_6', 'component_id_7', 'component_id_8']
    spec_cols = ['spec1','spec2','spec3','spec4','spec5','spec6','spec7','spec8']
    numerical_cols = ['annual_usage','min_order_quantity','quantity','diameter','wall','length','num_bends','bend_radius','num_boss','num_bracket','other','quantity_1','quantity_2','quantity_3','quantity_4','quantity_5','quantity_6','quantity_7','quantity_8','year','month']
    log_cols = ['annual_usage','min_order_quantity','quantity','diameter','wall','length','num_bends','bend_radius']
    new_feats = ['nspecs', 'nparts', 'max_quantity', 'mean_quantity', 'n_positions']
    
    ensemble=[]
   
    #quantity
    #Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,balance=base_cols+comp_cols+spec_cols,logtransform=['quantity'],useRdata=False,standardize=None)
    #model = FeaturePredictor('quantity')
    #xmodel = XModel("quantity",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta)
    #ensemble.append(xmodel)
    
    #LR1
    #Xtest,Xtrain,ytrain,idx,ta,ta_test = prepareDataset(seed=123,nsamples='shuffle',log1p=True,useRdata=False,createFeatures=True,createVolumeFeats=True,standardize='all',balance=base_cols+comp_cols+spec_cols,oneHotenc=['material_id','supplier'])
    #model = LassoLarsCV()
    #xmodel = XModel("lr1_r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta)
    #ensemble.append(xmodel)
    
    #quantity_inv
    #Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,balance=base_cols+comp_cols+spec_cols,invertQuantity=True,useRdata=False,standardize=None)
    #model = FeaturePredictor('quantity_inv')
    #xmodel = XModel("quantity_inv",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta)
    #ensemble.append(xmodel)
    
    #annual_usage
    #Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,balance=base_cols+comp_cols+spec_cols,logtransform=['annual_usage'],useRdata=False,standardize=None)
    #model = FeaturePredictor('annual_usage')
    #xmodel = XModel("annual_usage",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta)
    #ensemble.append(xmodel)
   
    #bracket_pricing
    #Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,balance=base_cols+comp_cols+spec_cols,useRdata=False,standardize=None)
    #model = FeaturePredictor('bracket_pricing')
    #xmodel = XModel("bracket_pricing",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta)
    #ensemble.append(xmodel)
   
   
    #RF1 CV=0.262 CV_8f=0.263
    #Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,balance=base_cols+comp_cols+spec_cols,useRdata=False,createFeatures=True,standardize=None,oneHotenc=['supplier'])
    #model = RandomForestRegressor(n_estimators=250,max_depth=None,min_samples_leaf=1,n_jobs=4, max_features=20)
    #xmodel = XModel("rf1_r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta)
    #xmodel = XModel("rf1_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta,bag_mode=True)
    #ensemble.append(xmodel)
    
    #RF2  CV=0.268
    #Xtest,Xtrain,ytrain,idx,ta = prepareDataset(seed=123,nsamples='shuffle',log1p=True,balance=base_cols+comp_cols+spec_cols,useRdata=True,createFeatures=False,standardize=False)
    #model = RandomForestRegressor(n_estimators=250,max_depth=None,min_samples_leaf=1,n_jobs=4, max_features=20)
    #xmodel = XModel("rf2_r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta)
    #xmodel = XModel("rf2_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta,bag_mode=True)
    #ensemble.append(xmodel)
    
    #RF3 CV=0.260 on sparse data (CV_3f = 0.267)? CV_8f=0.257
    #Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,balance=base_cols+comp_cols+spec_cols,createSparse=True,standardize=False,oneHotenc=categoricals)
    #model = RandomForestRegressor(n_estimators=250,max_depth=None,min_samples_leaf=1,n_jobs=4, max_features=Xtrain.shape[1]/2)
    #xmodel = XModel("rf3_r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta)
    #xmodel = XModel("rf3_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta,bag_mode=True)
    #ensemble.append(xmodel)  
    
    #RF4 CV_10f=0.244 # sample_weights!!!
    #Xtest,Xtrain,ytrain,idx,ta,sample_weight = prepareDataset(seed=123,nsamples='shuffle',log1p=True,standardize=None,balance=base_cols+comp_cols+spec_cols,removeRare=10,oneHotenc=['supplier'],createFeatures=True,createSupplierFeatures=['supplier','quantity','annual_usage','diameter'],createMaterialFeatures=['material_id','quantity','diameter'],createVerticalFeatures=False,shapeFeatures=True,timeFeatures=True,materialCost=False,removeLowVariance=True,removeSpec=True,useSampleWeights=True)
    #model = RandomForestRegressor(n_estimators=400,max_depth=None,min_samples_leaf=1,n_jobs=1, max_features=Xtrain.shape[1]/2,bootstrap=True)
    #xmodel = XModel("rf4_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta,bag_mode=True,sample_weight=sample_weight)
    #ensemble.append(xmodel)
    
    #RF5 CV_10f=0.244
    #Xtest,Xtrain,ytrain,idx,ta,sample_weight = prepareDataset(seed=123,nsamples='shuffle',log1p=True,standardize=None,balance=base_cols+comp_cols+spec_cols,removeRare=10,oneHotenc=['supplier'],createFeatures=True,createSupplierFeatures=['supplier','quantity','annual_usage','diameter'],createMaterialFeatures=['material_id','quantity','diameter'],createVerticalFeatures=False,shapeFeatures=True,timeFeatures=True,materialCost=False,removeLowVariance=True,removeSpec=True,useSampleWeights=True)
    #model = RandomForestRegressor(n_estimators=400,max_depth=None,min_samples_leaf=1,n_jobs=1, max_features=Xtrain.shape[1]/2,bootstrap=True)
    #iterativeFeatureSelection(model,Xtrain,Xtest,ytrain,iterations=10,nrfeats=1,scoring='rmse',cv=KLabelFolds(pd.Series(ta_train), n_folds=8, repeats =1),n_jobs=8)
    #xmodel = XModel("rf5_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta,bag_mode=True,sample_weight=sample_weight)
    #ensemble.append(xmodel)
    
    #GBR1 CV=0.218 CV_5f = 0.232 CV_8f = 0.229 
    #Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,balance=base_cols+comp_cols+spec_cols,useRdata=False,createFeatures=True,standardize=None,oneHotenc=['supplier'],removeRare=30,removeSpec=True)
    #model = GradientBoostingRegressor(n_estimators=2000,learning_rate=0.05,max_depth=7,subsample=.5)
    #xmodel = XModel("gbr1_r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta)
    #xmodel = XModel("gbr1_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta,bag_mode=True)
    #ensemble.append(xmodel)
    
    
    
    #NN1 (CV_8f=0.260, LB=0.257
    """
    Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,balance=base_cols+comp_cols+spec_cols,logtransform=log_cols,createFeatures=True,standardize=numerical_cols,oneHotenc=categoricals_nospec,removeRare=30,removeSpec=True)
    model = NeuralNet(layers=[('input', layers.InputLayer),
	('dropout0', layers.DropoutLayer),
	('hidden1', layers.DenseLayer),
	('dropout1', layers.DropoutLayer),
	('hidden2', layers.DenseLayer),
	('dropout2', layers.DropoutLayer),
	('output', layers.DenseLayer)],

	input_shape=(None, 271),
	dropout0_p=0.0,

	hidden1_num_units=256,
	hidden1_nonlinearity=nonlinearities.sigmoid,
	dropout1_p=0.15,

	hidden2_num_units=256,
	hidden2_nonlinearity=nonlinearities.sigmoid,
	dropout2_p=0.15,

	output_num_units=1,
	output_nonlinearity=None,

	regression=True,
	objective=RMSE,
	batch_iterator_train=ShuffleBatchIterator(batch_size = 32),

	update=rmsprop,#0.005
	update_learning_rate=theano.shared(float32(0.005)),
	
	eval_size=0.0,
	verbose=1,
	max_epochs=150,

	on_epoch_finished=[
		AdjustVariable('update_learning_rate', start=0.005, stop=0.001),
		#EarlyStopping(patience=20),
		],
	)
    #xmodel = XModel("nn1_r1",classifier=model,Xtrain=Xtrain.values,Xtest=Xtest.values,ytrain=ytrain.reshape((ytrain.shape[0],1)),cv_labels=ta)
    xmodel = XModel("nn1_br1",classifier=model,Xtrain=Xtrain.values,Xtest=Xtest.values,ytrain=ytrain.reshape((ytrain.shape[0],1)),cv_labels=ta,bag_mode=True)
    ensemble.append(xmodel)
    """
    
    #NN2 (CV_8f=0.257, LB=0.260
    """
    Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,balance=base_cols+comp_cols+spec_cols,logtransform=log_cols,createFeatures=True,standardize=numerical_cols,oneHotenc=categoricals_nospec,removeRare=30,removeSpec=True)   
    model = NeuralNet(layers=[('input', layers.InputLayer),
	('dropout0', layers.DropoutLayer),
	('hidden1', layers.DenseLayer),
	('dropout1', layers.DropoutLayer),
	('hidden2', layers.DenseLayer),
	('dropout2', layers.DropoutLayer),
	('hidden3', layers.DenseLayer),
	('dropout3', layers.DropoutLayer),
	('output', layers.DenseLayer)],

	input_shape=(None, 271),
	dropout0_p=0.0,

	hidden1_num_units=512,
	hidden1_nonlinearity=nonlinearities.rectify,
	dropout1_p=0.0,

	hidden2_num_units=512,
	hidden2_nonlinearity=nonlinearities.rectify,
	dropout2_p=0.2,
	  
	hidden3_num_units=512,
	hidden3_nonlinearity=nonlinearities.rectify,
	dropout3_p=0.2,

	output_num_units=1,
	output_nonlinearity=None,

	regression=True,
	objective=RMSE,
	batch_iterator_train=ShuffleBatchIterator(batch_size = 64),

	#update=adagrad,#0.001
	update=rmsprop,
	update_learning_rate=theano.shared(float32(0.001)),
	
	eval_size=0.0,
	verbose=1,
	max_epochs=50,

	on_epoch_finished=[
		AdjustVariable('update_learning_rate', start=0.001, stop=0.00005),
		#EarlyStopping(patience=20),
		],
	)
    #xmodel = XModel("nn2_r1",classifier=model,Xtrain=Xtrain.values,Xtest=Xtest.values,ytrain=ytrain.reshape((ytrain.shape[0],1)),cv_labels=ta)
    xmodel = XModel("nn2_br1",classifier=model,Xtrain=Xtrain.values,Xtest=Xtest.values,ytrain=ytrain.reshape((ytrain.shape[0],1)),cv_labels=ta,bag_mode=True)
    ensemble.append(xmodel)
    """
    
    #NN6 after hyperopt NO BAGMODE! (CV_10f_1r=0.246 LB=0.247)
    """
    Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,balance=base_cols+comp_cols+spec_cols,logtransform=log_cols,createFeatures=True,createVerticalFeatures=True,standardize=numerical_cols+new_feats,oneHotenc=categoricals_nospec,removeRare=30,removeSpec=True)  
    model = NeuralNet(layers=[('input', layers.InputLayer),
	('dropout0', layers.DropoutLayer),
	('hidden1', layers.DenseLayer),
	('dropout1', layers.DropoutLayer),
	('hidden2', layers.DenseLayer),
	('dropout2', layers.DropoutLayer),
	('hidden3', layers.DenseLayer),
	('dropout3', layers.DropoutLayer),
	('output', layers.DenseLayer)],

	input_shape=(None, 274),
	dropout0_p=0.0,

	hidden1_num_units=550,
	hidden1_nonlinearity=nonlinearities.rectify,
	dropout1_p=0.14,

	hidden2_num_units=450,
	hidden2_nonlinearity=nonlinearities.rectify,
	dropout2_p=0.10,
	
	hidden3_num_units=450,
	hidden3_nonlinearity=nonlinearities.rectify,
	dropout3_p=0.13,

	output_num_units=1,
	output_nonlinearity=None,

	regression=True,
	objective=RMSE,
	batch_iterator_train=ShuffleBatchIterator(batch_size = 64),

	#update=adagrad,#0.001
	update=rmsprop,
	update_learning_rate=theano.shared(float32(1.18e-03)),
	
	eval_size=0.0,
	verbose=1,
	max_epochs=50,

	on_epoch_finished=[
		AdjustVariable('update_learning_rate', start=1.18e-03, stop=0.00005),
		#EarlyStopping(patience=20),
		],
	)
    xmodel = XModel("nn6_r1",classifier=model,Xtrain=Xtrain.values,Xtest=Xtest.values,ytrain=ytrain.reshape((ytrain.shape[0],1)),cv_labels=ta)
    xmodel = XModel("nn6_br1",classifier=model,Xtrain=Xtrain.values,Xtest=Xtest.values,ytrain=ytrain.reshape((ytrain.shape[0],1)),cv_labels=ta,bag_mode=True)
    ensemble.append(xmodel)
    """
    
    #NN7 BAGMODE IS BACK 10f 25 repeats!!! LB=0.245/0.233
    """
    Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,balance=base_cols+comp_cols+spec_cols,logtransform=log_cols,createFeatures=True,createVerticalFeatures=True,standardize=numerical_cols+new_feats,oneHotenc=categoricals_nospec,removeRare=30,removeSpec=True)  
    model = NeuralNet(layers=[('input', layers.InputLayer),
	('dropout0', layers.DropoutLayer),
	('hidden1', layers.DenseLayer),
	('dropout1', layers.DropoutLayer),
	('hidden2', layers.DenseLayer),
	('dropout2', layers.DropoutLayer),
	('hidden3', layers.DenseLayer),
	('dropout3', layers.DropoutLayer),
	('output', layers.DenseLayer)],

	input_shape=(None, 274),
	dropout0_p=0.0,

	hidden1_num_units=550,
	hidden1_nonlinearity=nonlinearities.rectify,
	dropout1_p=0.14,

	hidden2_num_units=450,
	hidden2_nonlinearity=nonlinearities.rectify,
	dropout2_p=0.10,
	
	hidden3_num_units=450,
	hidden3_nonlinearity=nonlinearities.rectify,
	dropout3_p=0.13,

	output_num_units=1,
	output_nonlinearity=None,

	regression=True,
	objective=RMSE,
	batch_iterator_train=ShuffleBatchIterator(batch_size = 64),

	#update=adagrad,#0.001
	update=rmsprop,
	update_learning_rate=theano.shared(float32(1.18e-03)),
	
	eval_size=0.0,
	verbose=1,
	max_epochs=50,

	on_epoch_finished=[
		AdjustVariable('update_learning_rate', start=1.18e-03, stop=0.00005),
		#EarlyStopping(patience=20),
		],
	)
    xmodel = XModel("nn7_br25",classifier=model,Xtrain=Xtrain.values,Xtest=Xtest.values,ytrain=ytrain.reshape((ytrain.shape[0],1)),cv_labels=ta,bag_mode=True)
    ensemble.append(xmodel)
    """
    
    """
    #NN8 bs=64 ->0.248  bs=32->0.250  bs=32/LR=2.0*1e-03->0.253  max_epochs=100->0.246 stop=0.0005->0.287 stop=0.00001->0.243 dropout3_p=0.05->0.245 dropout3_p=0.15->0.246 max_epoch=120->0.243 , objective_alpha->1.0*1E-4 0.238!!!
    Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,balance=base_cols+comp_cols+spec_cols,logtransform=log_cols,createFeatures=True,createVerticalFeatures=True,standardize=numerical_cols+new_feats,oneHotenc=categoricals_nospec,removeRare=30,removeSpec=True)  
    model = NeuralNet(layers=[('input', layers.InputLayer),
	('dropout0', layers.DropoutLayer),
	('hidden1', layers.DenseLayer),
	('dropout1', layers.DropoutLayer),
	('hidden2', layers.DenseLayer),
	('dropout2', layers.DropoutLayer),
	('hidden3', layers.DenseLayer),
	('dropout3', layers.DropoutLayer),
	('output', layers.DenseLayer)],

	input_shape=(None, Xtrain.shape[1]),
	dropout0_p=0.0,

	hidden1_num_units=600,
	hidden1_nonlinearity=nonlinearities.LeakyRectify(leakiness=0.1),
	dropout1_p=0.10,

	hidden2_num_units=600,
	hidden2_nonlinearity=nonlinearities.LeakyRectify(leakiness=0.1),
	dropout2_p=0.10,
	
	hidden3_num_units=600,
	hidden3_nonlinearity=nonlinearities.LeakyRectify(leakiness=0.1),
	dropout3_p=0.10,

	output_num_units=1,
	output_nonlinearity=None,

	regression=True,
	objective=RMSE,
	objective_alpha=1.0*1E-4,#key!!!
	batch_iterator_train=ShuffleBatchIterator(batch_size = 64),#->32?

	#update=adagrad,#0.001
	update=rmsprop,
	update_learning_rate=theano.shared(float32(1.0*1e-03)),
	
	eval_size=0.0,
	verbose=1,
	max_epochs=100,

	on_epoch_finished=[
		AdjustVariable('update_learning_rate', start=1.0*1e-03, stop=0.00001),
		#EarlyStopping(patience=20),
		],
	)
    xmodel = XModel("nn8_br1",classifier=model,Xtrain=Xtrain.values,Xtest=Xtest.values,ytrain=ytrain.reshape((ytrain.shape[0],1)),cv_labels=ta,bag_mode=True)
    ensemble.append(xmodel)
    """
    
    
    #NN9 bs=64 ->0.248  bs=32->0.250  bs=32/LR=2.0*1e-03->0.253  max_epochs=100->0.246 stop=0.0005->0.287 stop=0.00001->0.243 dropout3_p=0.05->0.245 dropout3_p=0.15->0.246 max_epoch=120->0.243 , objective_alpha->1.0*1E-4 0.238! -> 0.229
    Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples=-1,log1p=True,balance=base_cols+comp_cols+spec_cols,logtransform=log_cols,createFeatures=True,createVerticalFeatures=True,standardize=numerical_cols+new_feats,oneHotenc=categoricals_nospec,removeRare=30,removeSpec=True)  
    #Xtrain.to_csv('./share/Xtrain_nn9_br20.csv',index=False)
    #Xtest.to_csv('./share/Xtest_nn9_br20.csv',index=False)
    model = NeuralNet(layers=[('input', layers.InputLayer),
	('dropout0', layers.DropoutLayer),
	('hidden1', layers.DenseLayer),
	('dropout1', layers.DropoutLayer),
	('hidden2', layers.DenseLayer),
	('dropout2', layers.DropoutLayer),
	('hidden3', layers.DenseLayer),
	('dropout3', layers.DropoutLayer),
	('output', layers.DenseLayer)],

	input_shape=(None, Xtrain.shape[1]),
	dropout0_p=0.0,

	hidden1_num_units=600,
	hidden1_nonlinearity=nonlinearities.LeakyRectify(leakiness=0.1),
	dropout1_p=0.10,

	hidden2_num_units=600,
	hidden2_nonlinearity=nonlinearities.LeakyRectify(leakiness=0.1),
	dropout2_p=0.10,
	
	hidden3_num_units=600,
	hidden3_nonlinearity=nonlinearities.LeakyRectify(leakiness=0.1),
	dropout3_p=0.10,

	output_num_units=1,
	output_nonlinearity=None,

	regression=True,
	objective=RMSE,
	objective_alpha=.5*1E-4,#key!!!
	batch_iterator_train=ShuffleBatchIterator(batch_size = 64),#->32?

	#update=adagrad,#0.001
	update=rmsprop,
	update_learning_rate=theano.shared(float32(1.0*1e-03)),
	
	eval_size=0.0,
	verbose=1,
	max_epochs=100,

	on_epoch_finished=[
		AdjustVariable('update_learning_rate', start=1.0*1e-03, stop=0.00001),
		#EarlyStopping(patience=20),
		],
	)
    xmodel = XModel("nn9_br20",classifier=model,Xtrain=Xtrain.values,Xtest=Xtest.values,ytrain=ytrain.reshape((ytrain.shape[0],1)),cv_labels=ta,bag_mode=True)
    ensemble.append(xmodel)
    
    
    #KERAS1 CV_10f = 0.2600
    #Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,balance=base_cols+comp_cols+spec_cols,logtransform=log_cols,createFeatures=True,createVerticalFeatures=False,standardize=numerical_cols+new_feats,oneHotenc=categoricals_nospec,removeRare=30,removeSpec=True)  
    #model = KerasNNReg(dims=Xtrain.shape[1])
    #xmodel = XModel("keras1_br1",classifier=model,Xtrain=Xtrain.values,Xtest=Xtest.values,ytrain=ytrain.reshape((ytrain.shape[0],1)),cv_labels=ta,bag_mode=True)
    #ensemble.append(xmodel)
    
    #KERAS2
    #Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,balance=base_cols+comp_cols+spec_cols,logtransform=log_cols,createFeatures=True,createVerticalFeatures=False,standardize=numerical_cols+new_feats,oneHotenc=categoricals_nospec,removeRare=30,removeSpec=True)  
    #model = KerasNNReg2(dims=Xtrain.shape[1])
    #xmodel = XModel("keras2_br1",classifier=model,Xtrain=Xtrain.values,Xtest=Xtest.values,ytrain=ytrain.reshape((ytrain.shape[0],1)),cv_labels=ta,bag_mode=True)
    #ensemble.append(xmodel)
    
    #XRF1  CV=0.259  CV_8f = 0.264 #needs a lot of mxfeatures!!! 
    #Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,standardize=None,balance=base_cols+comp_cols+spec_cols,bagofwords=None,createFeatures=True,oneHotenc=['material_id','supplier'],removeSpec=True,dropFeatures=None)
    #model = ExtraTreesRegressor(n_estimators=250,max_depth=None,min_samples_leaf=1,n_jobs=4, max_features=90,bootstrap=False)
    #xmodel = XModel("xrf1_r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta)
    #xmodel = XModel("xrf1_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta,bag_mode=True)
    #ensemble.append(xmodel)
    
    #XRF2  CV= CV_8f = 0.249	PL=0.242 !!!  CV_10f_bm = 0.248
    #Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,standardize=None,balance=base_cols+comp_cols+spec_cols,removeRare=15,oneHotenc=['supplier','material_id'],createFeatures=True,createSupplierFeatures=['supplier','quantity'],createVerticalFeatures=True,removeSpec=True)
    #model = ExtraTreesRegressor(n_estimators=500,max_depth=None,min_samples_leaf=1,n_jobs=2, max_features=70,bootstrap=False)
    #xmodel = XModel("xrf2_r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta)
    #xmodel = XModel("xrf2_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta,bag_mode=True)
    #ensemble.append(xmodel)
    
    #XRF3 ALL component data CV_10f_bm=0.286
    #Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,balance=base_cols+comp_cols+spec_cols,NA_filler=0,comptypes=7,createSparse=True,removeLowVariance=True,removeSpec=False)
    #model = ExtraTreesRegressor(n_estimators=250,max_depth=None,min_samples_leaf=1,n_jobs=1, max_features=3*Xtrain.shape[1]/4,bootstrap=False)
    #xmodel = XModel("xrf3_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta,bag_mode=True)
    #ensemble.append(xmodel)
    
    #XRF4b more owen encoding + financial features CV_10f_bm=0.277
    #Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,standardize=None,balance=base_cols+comp_cols+spec_cols,removeRare=None,oneHotenc=None,createFeatures=True,owenEncoding=['supplier','material_id'],createInflationData=True)
    #model = ExtraTreesRegressor(n_estimators=250,max_depth=None,min_samples_leaf=1,n_jobs=1, max_features=3*Xtrain.shape[1]/4,bootstrap=False)
    #xmodel = XModel("xrf4b_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta,bag_mode=True)
    #ensemble.append(xmodel)
    
    #XRF4 createInflationData=False more owen encoding  CV_10f_bm=0.278 
    #Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,standardize=None,balance=base_cols+comp_cols+spec_cols,removeRare=None,oneHotenc=None,createFeatures=True,owenEncoding=['supplier','material_id'],createInflationData=False)
    #model = ExtraTreesRegressor(n_estimators=250,max_depth=None,min_samples_leaf=1,n_jobs=1, max_features=3*Xtrain.shape[1]/4,bootstrap=False)
    #xmodel = XModel("xrf4_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta,bag_mode=True)
    #ensemble.append(xmodel)
    
    
    #XRF5 discount
    #Xtest,Xtrain,ytrain,idx,ta,_ =  prepareDataset(seed=123,nsamples='shuffle',log1p=False,standardize=None,balance=base_cols+comp_cols+spec_cols,removeRare=15,oneHotenc=['supplier'],createFeatures=True,createSupplierFeatures=['supplier','quantity'],removeSpec=True,computeDiscount=True)
    #model = ExtraTreesRegressor(n_estimators=250,max_depth=None,min_samples_leaf=1,n_jobs=1, max_features=3*Xtrain.shape[1]/4,bootstrap=False)
    #xmodel = XModel("xrf_discount_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta,bag_mode=True)
    #ensemble.append(xmodel)
    
    #XRF6 ybinning
    #Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,standardize=None,balance=base_cols+comp_cols+spec_cols,removeRare=10,oneHotenc=['supplier'],yBinning=20,removeSpec=True)
    #model = RandomForestClassifier(n_estimators=200,max_depth=None,min_samples_leaf=1,n_jobs=1, max_features=Xtrain.shape[1]/2,bootstrap=False)
    #xmodel = XModel("xrf6_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta,bag_mode=True)
    #ensemble.append(xmodel)

    #XRF5 CV_10f=0.266
    #Xtest,Xtrain,ytrain,idx,ta,sample_weight = prepareDataset(seed=123,nsamples='shuffle',log1p=True,standardize=None,balance=base_cols+comp_cols+spec_cols,removeRare=10,oneHotenc=['supplier'],createFeatures=True,createSupplierFeatures=['supplier','quantity','annual_usage','diameter'],createMaterialFeatures=['material_id','quantity','diameter'],createVerticalFeatures=False,shapeFeatures=True,timeFeatures=True,materialCost=False,removeLowVariance=True,removeSpec=True,useSampleWeights=True)
    #model = ExtraTreesRegressor(n_estimators=250,max_depth=None,min_samples_leaf=1,n_jobs=1, max_features=5*Xtrain.shape[1]/6,bootstrap=False)
    #xmodel = XModel("xrf5_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta,bag_mode=True,sample_weight=sample_weight)
    #ensemble.append(xmodel)
    
    
    #XGB1 CV=0.232 PL=0.246 CV_8f= 0.251
    #Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,balance=base_cols+comp_cols+spec_cols,useRdata=False,createFeatures=True,standardize=None,oneHotenc=['material_id','supplier'],bagofwords_v2_0=True)
    #model = XgboostRegressor(n_estimators=400,learning_rate=0.05,max_depth=15,subsample=.5,colsample_bytree=0.8,n_jobs=4,objective='reg:linear',eval_metric='rmse',booster='gbtree',silent=1,eval_size=0.0)
    #xmodel = XModel("xgb1_r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta)
    #ensemble.append(xmodel)
    
    #XGB2 Best model so far CV_8f=0.229 PL=0.226  CV_10f_bm=0.225
    #Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,standardize=None,balance=base_cols+comp_cols+spec_cols,bagofwords=None,createFeatures=True,oneHotenc=['material_id','supplier'],removeSpec=True)
    #model = XgboostRegressor(n_estimators=2000,learning_rate=0.05,max_depth=7,subsample=.8,colsample_bytree=0.8,min_child_weight=5,n_jobs=4,objective='reg:linear',eval_metric='rmse',booster='gbtree',silent=1,eval_size=0.0)
    #xmodel = XModel("xgb2_r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta)
    #xmodel = XModel("xgb2_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta,bag_mode=True)
    #ensemble.append(xmodel) """
    
    #XGB3  balance also for comp and spec! PL=0.228 CV_8f=0.228 
    #dropFeatures=['supplier_2','supplier_45','supplier_40','supplier_33','supplier_23','supplier_21','supplier_8','supplier_7','supplier_3','supplier_1','quantity_8','supplier_11','supplier_0','supplier_27','supplier_28','supplier_34','supplier_6','component_id_8','material_id_1','supplier_44','component_id_7','quantity_7']
    #Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,standardize=None,balance=base_cols+comp_cols+spec_cols,bagofwords=None,createFeatures=True,oneHotenc=['material_id','supplier'],removeSpec=True,dropFeatures=dropFeatures)
    #model = XgboostRegressor(n_estimators=2000,learning_rate=0.05,max_depth=7,subsample=.8,colsample_bytree=0.8,min_child_weight=5,n_jobs=2,objective='reg:linear',eval_metric='rmse',booster='gbtree',silent=1,eval_size=0.0)
    #xmodel = XModel("xgb3_r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta)
    #xmodel = XModel("xgb3_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta,bag_mode=True)
    #ensemble.append(xmodel)
    
    #XGB4 linear xgboost using sparse matrix CV_8f = 0.233
    #Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,balance=base_cols+comp_cols+spec_cols,createSparse=True,standardize=False,oneHotenc=categoricals)
    #model = XgboostRegressor(booster='gblinear',n_estimators=900,alpha_L1=0,lambda_L2=0,n_jobs=2,objective='reg:linear',eval_metric='rmse',silent=1)#0.63
    #xmodel = XModel("xgb4_r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta,bag_mode=True)
    #xmodel = XModel("xgb4_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta,bag_mode=True)
    #ensemble.append(xmodel)
    
    #XGB5 BAG_OF_WORDS! CV_8f=0.240
    #Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,standardize=None,balance=base_cols+comp_cols+spec_cols,bagofwords_v2_0=True,createFeatures=True,useFrequencies=False,removeSpec=False,removeComp=False)
    #model = XgboostRegressor(n_estimators=2000,learning_rate=0.05,max_depth=7,subsample=.8,colsample_bytree=0.8,min_child_weight=5,n_jobs=4,objective='reg:linear',eval_metric='rmse',booster='gbtree',silent=1,eval_size=0.0)
    #xmodel = XModel("xgb5_r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta)
    #xmodel = XModel("xgb5_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta,bag_mode=True)
    #ensemble.append(xmodel)
    
    #XGB6 FEATURE FREQUENCIES CV_10f=0.2367
    #Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,standardize=None,balance=base_cols+comp_cols+spec_cols,bagofwords_v2_0=None,createFeatures=True,useFrequencies=True,removeSpec=True)
    #model = XgboostRegressor(n_estimators=8000,learning_rate=0.01,max_depth=8,subsample=.7,colsample_bytree=.7,min_child_weight=5,n_jobs=4,objective='reg:linear',eval_metric='rmse',booster='gbtree',silent=1,eval_size=0.0)
   #xmodel = XModel("xgb6_r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta)
    #xmodel = XModel("xgb6_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta,bag_mode=True)
    #ensemble.append(xmodel)
    
    """
    #XGB7 MANY ITERATIONS + SUPPLIER_QUANTITY CV_8f=0.0.223 PL=0.223 super stable base model!!!
    #Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,standardize=None,balance=base_cols+comp_cols+spec_cols,removeRare=15,oneHotenc=['supplier','material_id'],createFeatures=True,createSupplierFeatures=['supplier','quantity'],removeSpec=True)
    Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples=-1,log1p=True,standardize=None,balance=base_cols+comp_cols+spec_cols,removeRare=15,oneHotenc=['supplier','material_id'],createFeatures=True,createSupplierFeatures=['supplier','quantity'],removeSpec=True)
    print ytrain
    raw_input()
    Xtrain_ta = pd.concat([pd.DataFrame(ta,columns=['tube_assembly_id'],index=Xtrain.index),Xtrain],axis=1)
    Xtrain_ta.to_csv('./share/Xtrain_xgb7_br1.csv',index=False)
    Xtest_ta = pd.concat([pd.DataFrame(_,columns=['tube_assembly_id'],index=Xtest.index),Xtest],axis=1)
    Xtest_ta.to_csv('./share/Xtest_xgb7_br1.csv',index=False)
    #model = XgboostRegressor(n_estimators=8000,learning_rate=0.01,max_depth=8,subsample=.7,colsample_bytree=.7,min_child_weight=5,n_jobs=4,objective='reg:linear',eval_metric='rmse',booster='gbtree',silent=1,eval_size=0.0)
    #xmodel = XModel("xgb7_r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta)
    #xmodel = XModel("xgb7_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta,bag_mode=True)
    #ensemble.append(xmodel)
    """
    
    #XGB8 ALL component data CV_10f_bag=0.226
    #Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,balance=base_cols+comp_cols+spec_cols,NA_filler=0,comptypes=7,createSparse=True,removeLowVariance=True,removeSpec=False)
    #model = XgboostRegressor(NA=0,n_estimators=8000,learning_rate=0.01,max_depth=8,subsample=.7,colsample_bytree=.7,min_child_weight=5,n_jobs=1,objective='reg:linear',eval_metric='rmse',booster='gbtree',silent=1,eval_size=0.0)
    #xmodel = XModel("xgb8_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta,bag_mode=True)
    #ensemble.append(xmodel)
    
    #XGB9 simple owen encoding
    #Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,standardize=None,balance=base_cols+comp_cols+spec_cols,removeRare=None,oneHotenc=None,createFeatures=True,owenEncoding=['supplier'])
    #model = XgboostRegressor(NA=0,n_estimators=400,learning_rate=0.05,max_depth=15,subsample=.5,colsample_bytree=0.8,n_jobs=1,objective='reg:linear',eval_metric='rmse',booster='gbtree',silent=1,eval_size=0.0)   
    #xmodel = XModel("xgb9_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta,bag_mode=True)
    #ensemble.append(xmodel)
    
    #XGB10b more owen encoding + financial features CV_10f_bm=0.226/0.224
    #Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,standardize=None,balance=base_cols+comp_cols+spec_cols,removeRare=None,oneHotenc=None,createFeatures=True,owenEncoding=['supplier','material_id'],createInflationData=True)
    #model = XgboostRegressor(booster='gbtree',NA=0,n_estimators=8000,learning_rate=0.01,max_depth=8,subsample=.5,colsample_bytree=0.8,n_jobs=1,objective='reg:linear',eval_metric='rmse',silent=1,eval_size=0.0)   
    #xmodel = XModel("xgb10b_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta,bag_mode=True)
    #ensemble.append(xmodel)
    
    #XGB10 createInflationData=False  more owen encoding  CV_10f_bm=0.225
    #Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,standardize=None,balance=base_cols+comp_cols+spec_cols,removeRare=None,oneHotenc=None,createFeatures=True,owenEncoding=['supplier','material_id'],createInflationData=False)
    #model = XgboostRegressor(booster='gbtree',NA=0,n_estimators=8000,learning_rate=0.01,max_depth=8,subsample=.5,colsample_bytree=0.8,n_jobs=1,objective='reg:linear',eval_metric='rmse',silent=1,eval_size=0.0)   
    #xmodel = XModel("xgb10_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta,bag_mode=True)
    #ensemble.append(xmodel)
    
    #XGB11 ALL component data gblinear CV_10f_bag=0.633 ->useless
    #Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,balance=base_cols+comp_cols+spec_cols,comptypes=1,createSparse=False,standardize=numerical_cols,removeLowVariance=True,removeSpec=False,createInflationData=False)
    #model = XgboostRegressor(booster='gblinear',learning_rate=0.1,n_estimators=1000,alpha_L1=5.0,lambda_L2=5.0,n_jobs=4,objective='reg:linear',eval_metric='rmse',silent=1)#0.63
    #xmodel = XModel("xgb11_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta,bag_mode=True)
    #ensemble.append(xmodel)
    
    #XGB12 computing discount!!!
    #Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=False,standardize=None,balance=base_cols+comp_cols+spec_cols,removeRare=15,oneHotenc=['supplier','material_id'],createFeatures=True,createSupplierFeatures=['supplier','quantity'],removeSpec=True,computeDiscount=True)
    #model = XgboostRegressor(n_estimators=2000,learning_rate=0.05,max_depth=7,subsample=.8,colsample_bytree=0.8,min_child_weight=5,n_jobs=1,objective='reg:linear',eval_metric='rmse',booster='gbtree',silent=1,eval_size=0.0)
    #xmodel = XModel("xgb_discount_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta,bag_mode=True)
    #ensemble.append(xmodel)
    
 
    #XGB13 ybinning->too long?
    #Xtest,Xtrain,ytrain,idx,ta,_ = Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,standardize=None,balance=base_cols+comp_cols+spec_cols,removeRare=15,oneHotenc=['supplier','material_id'],createSupplierFeatures=['supplier','quantity'],createVerticalFeatures=True,yBinning=25)
    #model = XgboostClassifier(n_estimators=2000,learning_rate=0.05,max_depth=7,subsample=.8,colsample_bytree=0.8,min_child_weight=5,n_jobs=1,objective='multi:softmax',eval_metric='rmse',booster='gbtree',silent=1,eval_size=0.0)
    #xmodel = XModel("xgb13_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta,bag_mode=True)
    #ensemble.append(xmodel)
    
    #XGB14  total owen encoding  CV_10f_bm=0.225
    #Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,standardize=None,balance=base_cols+comp_cols+spec_cols,removeRare=None,oneHotenc=None,owenEncoding=categoricals_nospec,shapeFeatures=True,materialCost=True)
    #model = XgboostRegressor(booster='gbtree',NA=0,n_estimators=8000,learning_rate=0.01,max_depth=8,subsample=.5,colsample_bytree=0.8,n_jobs=1,objective='reg:linear',eval_metric='rmse',silent=1,eval_size=0.0)   
    #xmodel = XModel("xgb14_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta,bag_mode=True)
    #ensemble.append(xmodel)
    
    #XGB15 MANY ITERATIONS ALL FEATURES using weights
    #Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,standardize=None,balance=base_cols+comp_cols+spec_cols,removeRare=20,oneHotenc=['supplier'],createFeatures=True,createSupplierFeatures=['supplier','quantity'],createVerticalFeatures=False,shapeFeatures=True,timeFeatures=True,materialCost=False,removeLowVariance=True,removeSpec=True)
    #model = XgboostRegressor(n_estimators=4000,learning_rate=0.0269,max_depth=7,subsample=0.9587,colsample_bytree=0.5772,min_child_weight=6,gamma=2.1712, n_jobs=2,objective='reg:linear',eval_metric='rmse',booster='gbtree',silent=1,eval_size=0.0)
    #xmodel = XModel("xgb15_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta,bag_mode=True)
    #ensemble.append(xmodel)
    
    """
    #XGB16 MANY ITERATIONS ALL FEATURES CV_10f=0.218 PL=0.229 material_cost overfit? timeFeatures overfit? ->NOW PL = 0.225250
    Xtest,Xtrain,ytrain,idx,ta,sample_weight = prepareDataset(seed=123,nsamples='shuffle',log1p=True,standardize=None,balance=base_cols+comp_cols+spec_cols,removeRare=10,oneHotenc=['supplier'],createFeatures=True,createSupplierFeatures=['supplier','quantity','annual_usage','diameter'],createMaterialFeatures=['material_id','quantity','diameter'],createVerticalFeatures=False,shapeFeatures=True,timeFeatures=True,materialCost=False,removeLowVariance=True,removeSpec=True,computeDiscount=True,useSampleWeights=True)
    model = XgboostRegressor(booster='gbtree',NA=0,n_estimators=4000,learning_rate=0.02,max_depth=8,subsample=.5,colsample_bytree=0.8,min_child_weight=5,n_jobs=2,objective='reg:linear',eval_metric='rmse',silent=1,eval_size=0.0)   
    xmodel = XModel("xgb16_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta,bag_mode=True,sample_weight=sample_weight)
    ensemble.append(xmodel)
    """
    
    #XGB17 discount again
    #Xtest,Xtrain,ytrain,idx,ta,sample_weight = prepareDataset(seed=123,nsamples='shuffle',log1p=True,standardize=None,balance=base_cols+comp_cols+spec_cols,removeRare=None,oneHotenc=None,encodeKeepNumber=True,createFeatures=True,createSupplierFeatures=['supplier','quantity','annual_usage','diameter'],createMaterialFeatures=['material_id','quantity','diameter'],createVerticalFeatures=False,shapeFeatures=True,timeFeatures=True,computeDiscount=False,removeLowVariance=True,removeSpec=True,useSampleWeights=False)
    #model = XgboostRegressor(booster='gbtree',NA=0,n_estimators=8000,learning_rate=0.01,max_depth=8,subsample=.5,colsample_bytree=0.8,min_child_weight=5,n_jobs=4,objective='reg:linear',eval_metric='rmse',silent=1,eval_size=0.0)   
    #xmodel = XModel("xgb17_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta,bag_mode=True,sample_weight=sample_weight)
    #ensemble.append(xmodel)
    
    #XGB18 linear xgboost using sparse matrix 
    #dropFeatures=['end_a_1x','end_a_2x','end_x_1x','end_x_2x','end_a','end_x']
    #Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,balance=base_cols+comp_cols+spec_cols,createSparse=True,standardize=False,oneHotenc=None,useTubExtended=True,dropFeatures=dropFeatures,removeComp=True,removeSpec=True)
    #model = XgboostRegressor(booster='gblinear',n_estimators=500,alpha_L1=10.0,lambda_L2=1.0,n_jobs=2,objective='reg:linear',eval_metric='rmse',silent=1)#0.63
    #model = XgboostRegressor(n_estimators=400,learning_rate=0.05,max_depth=15,subsample=.5,colsample_bytree=0.8,n_jobs=4,objective='reg:linear',eval_metric='rmse',booster='gbtree',silent=1,eval_size=0.0)
    #xmodel = XModel("xgb18_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta,bag_mode=True)
    #ensemble.append(xmodel)
    
       
    
    
    #LR2 CV_5f=0.474
    #Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,balance=base_cols+comp_cols+spec_cols,logtransform=log_cols,createFeatures=True,standardize=numerical_cols,oneHotenc=categoricals_nospec,removeRare=30,removeSpec=True)
    #model = RidgeCV()
    #xmodel = XModel("lr2_r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta)
    #xmodel = XModel("lr2_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta,bag_mode=True)
    #ensemble.append(xmodel)
    
    #LR3 CV_10f_bm=0.510
    #Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,balance=base_cols+comp_cols+spec_cols,logtransform=log_cols,NA_filler=0,comptypes=7,createSparse=False,removeLowVariance=True,oneHotenc=categoricals_nospec,removeSpec=False,standardize='all')
    #model = SGDRegressor(alpha=1E-6,n_iter=250,shuffle=True,penalty='l1',learning_rate='optimal')
    #model = RidgeCV()#0.510
    #model = LassoLarsCV()#
    #xmodel = XModel("lr3_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta,bag_mode=True)
    #ensemble.append(xmodel)
    
    #SVM2 CV_8f=0.314
    #Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,balance=base_cols+comp_cols+spec_cols,logtransform=log_cols,createFeatures=True,standardize=numerical_cols,oneHotenc=categoricals_nospec,removeRare=30,removeSpec=True)
    #model  = SVR(C=100,gamma=0.0)
    #xmodel = XModel("svm2_r1",classifier=model,Xtrain=Xtrain.values,Xtest=Xtest.values,ytrain=ytrain,cv_labels=ta)
    #xmodel = XModel("svm2_br1",classifier=model,Xtrain=Xtrain.values,Xtest=Xtest.values,ytrain=ytrain,cv_labels=ta,bag_mode=True)
    #ensemble.append(xmodel)
        
    #KNN1 CV=0.589 CV_8f=0.551
    #Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,standardize='all',balance=base_cols+comp_cols+spec_cols,bagofwords=None,createFeatures=True,createVolumeFeats=True,oneHotenc=['material_id','supplier'],removeSpec=True)
    #model = KNeighborsRegressor(n_neighbors=5)
    #xmodel = XModel("knn1_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta,bag_mode=True)
    #ensemble.append(xmodel)
    
    
    #LR4 CV= CV_8f=
    #Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,balance=base_cols+comp_cols+spec_cols,logtransform=log_cols,createFeatures=False,createVerticalFeatures=False,standardize=numerical_cols+new_feats,oneHotenc=None,removeRare=None,removeComp=True,removeSpec=True)  
    #model = Pipeline([('poly', PolynomialFeatures(degree=2,interaction_only=True)),('scaler', StandardScaler()),('linear', LassoCV(max_iter=5000, n_alphas=2, n_jobs=2))])
    #xmodel = XModel("lr4_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta,bag_mode=True)
    #ensemble.append(xmodel)
    
    
    #BAGXGB5 alternative ... CV_8f=0.222 PL=0.225 only slighlty overfitted!?
    #Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,standardize=None,balance=base_cols+comp_cols+spec_cols,bagofwords=None,createFeatures=True,createVerticalFeatures=True,oneHotenc=['supplier'],removeRare=10,removeSpec=True,dropFeatures=None)
    #model = XgboostRegressor(n_estimators=2000,learning_rate=0.06,max_depth=8,subsample=.57,colsample_bytree=0.96,min_child_weight=5,n_jobs=2,objective='reg:linear',eval_metric='rmse',booster='gbtree',silent=1,eval_size=0.0)
    #model = BaggingRegressor(base_estimator=model,n_estimators=3,n_jobs=1,verbose=2,random_state=None,max_samples=0.96,max_features=.96,bootstrap=False)
    #xmodel = XModel("bagxgb5_br1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta,bag_mode=True)
    #ensemble.append(xmodel) 
    
    
    #BAGMODELS
    """   
    #BAGXGB1 CV=0.232 PL=??? OVERFITTED?
    #Xtest,Xtrain,ytrain,idx,ta,_ =  prepareDataset(seed=123,nsamples='shuffle',log1p=True,standardize=None,balance=base_cols+comp_cols+spec_cols,bagofwords=None,createFeatures=True,createVerticalFeatures=True,oneHotenc=['supplier'],removeRare=10,removeSpec=True,dropFeatures=None)
    #model = XgboostRegressor(n_estimators=2000,learning_rate=0.06,max_depth=8,subsample=.55,colsample_bytree=0.7,min_child_weight=5,n_jobs=2,objective='reg:linear',eval_metric='rmse',booster='gbtree',silent=1,eval_size=0.0)
    #model = BaggingRegressor(base_estimator=model,n_estimators=1,n_jobs=1,verbose=1,random_state=None,max_samples=0.91,max_features=.91,bootstrap=False)
    #xmodel = XModel("bagxgb1_r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta)
    #ensemble.append(xmodel)
    
    
    #BAGXGB2 CV_8f=0.221 PL= OVERFITTED?
    #Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,standardize=None,balance=base_cols+comp_cols+spec_cols,bagofwords=None,createFeatures=True,createVerticalFeatures=True,oneHotenc=['supplier'],removeRare=10,removeSpec=True,dropFeatures=None)
    #model = XgboostRegressor(n_estimators=2000,learning_rate=0.06,max_depth=8,subsample=.55,colsample_bytree=0.7,min_child_weight=5,n_jobs=2,objective='reg:linear',eval_metric='rmse',booster='gbtree',silent=1,eval_size=0.0)
    #model = BaggingRegressor(base_estimator=model,n_estimators=3,n_jobs=1,verbose=1,random_state=None,max_samples=0.91,max_features=.91,bootstrap=False)
    #xmodel = XModel("bagxgb2_r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta)
    #ensemble.append(xmodel)
    
    
    #BAGXGB3 CV_8f=0.222 PL=0.227 OVERFITTED?
    #Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,standardize=None,balance=base_cols+comp_cols+spec_cols,bagofwords=None,createFeatures=True,createVerticalFeatures=True,oneHotenc=['supplier'],removeRare=10,removeSpec=True,dropFeatures=None)
    #model = XgboostRegressor(n_estimators=2000,learning_rate=0.06,max_depth=8,subsample=.55,colsample_bytree=0.7,min_child_weight=5,n_jobs=2,objective='reg:linear',eval_metric='rmse',booster='gbtree',silent=1,eval_size=0.0)
    #model = BaggingRegressor(base_estimator=model,n_estimators=10,n_jobs=1,verbose=1,random_state=None,max_samples=0.91,max_features=.91,bootstrap=False)
    #xmodel = XModel("bagxgb3_r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta)
    #ensemble.append(xmodel)
    
    #BAGXGB4 CV_8f=0.232 PL= OVERFITTED?
    #Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,standardize=None,balance=base_cols+comp_cols+spec_cols,bagofwords=None,createFeatures=True,createVerticalFeatures=True,oneHotenc=['supplier'],removeRare=10,removeSpec=True,dropFeatures=None)
    #model = XgboostRegressor(n_estimators=2000,learning_rate=0.06,max_depth=8,subsample=.55,colsample_bytree=0.7,min_child_weight=5,n_jobs=2,objective='reg:linear',eval_metric='rmse',booster='gbtree',silent=1,eval_size=0.0)
    #model = BaggingRegressor(base_estimator=model,n_estimators=20,n_jobs=1,verbose=1,random_state=None,max_samples=0.91,max_features=.91,bootstrap=False)
    #xmodel = XModel("bagxgb4_r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta)
    #ensemble.append(xmodel)
    
    #BAGXGB6 alternative ... CV_8f=0.229 PL=??? no verticals...
    #Xtest,Xtrain,ytrain,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,standardize=None,balance=base_cols+comp_cols+spec_cols,bagofwords=None,createFeatures=True,createVerticalFeatures=False,oneHotenc=['supplier'],removeRare=10,removeSpec=True,dropFeatures=None)
    #model = XgboostRegressor(n_estimators=2000,learning_rate=0.06,max_depth=8,subsample=.57,colsample_bytree=0.96,min_child_weight=5,n_jobs=2,objective='reg:linear',eval_metric='rmse',booster='gbtree',silent=1,eval_size=0.0)
    #model = BaggingRegressor(base_estimator=model,n_estimators=3,n_jobs=1,verbose=2,random_state=None,max_samples=0.96,max_features=.96,bootstrap=False)
    #xmodel = XModel("bagxgb6_r1",classifier=model,Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,cv_labels=ta)
    #ensemble.append(xmodel)
    """
    
    for m in ensemble:
	m.summary()
    return(ensemble)

    
def finalizeModel(m,use_proba=True):
	"""
	Make predictions and save them
	"""
	print "Make predictions and save them..."
	#oob from crossvalidation
	yoob = m.oob_preds
	#final prediction
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
	filename="./share/"+m.name+".csv"
	print "Saving oob + predictions as csv to:",filename
	allpred.to_csv(filename,index=False)
	
	#XModel.saveModel(m,"/home/loschen/Desktop/datamining-kaggle/higgs/data/"+m.name+".pkl")
	XModel.saveCoreData(m,"./share/"+m.name+".pkl")
	return(m)
    

def createOOBdata(ensemble,repeats=2,n_folds=5,n_jobs=1,score_func='log_loss',verbose=False,calibrate=False,use_proba=True):
    """
    parallel oob creation
    """
    global funcdict

    for m in ensemble:
	bag_mode = m.bag_mode
	print "\nComputing oob predictions for:",m.name
	print m.classifier.get_params
	if m.class_names is not None:
	    n_classes = len(m.class_names)
	else:
	    n_classes = 1
	print "n_classes",n_classes
	
	oob_preds=np.zeros((m.ytrain.shape[0],n_classes,repeats))
	preds = np.zeros((m.Xtest.shape[0],n_classes,repeats))
	
	ly=m.ytrain
	oobscore=np.zeros(repeats)
	maescore=np.zeros(repeats)
	
	#outer loop
	for j in xrange(repeats):
	    if m.cv_labels is not None:
		print "KLabelFold wrapper..."
		#cv = LeavePLabelOutWrapper(m.cv_labels,n_folds=n_folds,p=1)
		cv = list(KLabelFolds(m.cv_labels, n_folds=n_folds, repeats =1))
		
	    else:
		cv = KFold(ly.shape[0], n_folds=n_folds,shuffle=True,random_state=None)
	    
	    scores=np.zeros(len(cv))
	    scores2=np.zeros(len(cv))
	    
	    #parallel stuff
	    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
			    pre_dispatch='2*n_jobs')
	    
	    #parallel run, returns a list of oob predictions
	    results = parallel(delayed(fit_and_score)(clone(m.classifier), m.Xtrain, ly, train, test,sample_weight=m.sample_weight,use_proba=use_proba,returnModel=bag_mode)
			  for train, test in cv)

	    for i,(train,test) in enumerate(cv):
		oob_pred,cv_model = results[i]
	        oob_pred = oob_pred.reshape(oob_pred.shape[0],n_classes)
		oob_preds[test,:,j] = oob_pred
		
		scores[i]=funcdict[score_func](ly[test],oob_preds[test,:,j])
		
		if bag_mode:
		  print "Using cv models for test set(bag_mode)..."
		  if use_proba:
		    p = cv_model.predict_proba(m.Xtest)
		    p = p.reshape(p.shape[0],n_classes)
		    preds[:,:,j] = p
		  else:
		    p = cv_model.predict(m.Xtest)
		    p = p.reshape(p.shape[0],n_classes)
		    preds[:,:,j] = p
		    
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
	m.oob_preds=np.mean(oob_preds,axis=2)
	#if not use_proba:
	#  print "Warning: rounding oob data!"
	#  m.oob_preds = np.round(m.oob_preds)
	
	#print m.oob_preds[:10]

	score_oob = funcdict[score_func](ly,m.oob_preds)
	print "Summary: <score,oob>: %6.3f +- %6.3f   score,oob-total: %0.3f (after %d repeats)\n" %(oobscore.mean(),oobscore.std(),score_oob,repeats)
	
	if not bag_mode:
	    #Train full model on total train data
	    print "Train full model on whole train set..."	
	    if m.sample_weight is not None:
		print "... with sample weights"
		m.classifier.fit(m.Xtrain,ly,m.sample_weight)
	    else:
		m.classifier.fit(m.Xtrain,ly)
	    
	    if use_proba:
	      m.preds = m.classifier.predict_proba(m.Xtest)
	    else:
	      m.preds = m.classifier.predict(m.Xtest)
	  
	else:
	      print "bag_mode: averaging all cv classifier results"
	      #print preds[:10]
	      m.preds= np.mean(preds,axis=2)
	      m.preds[:10]
	      #if not use_proba:
	      #	  print "Warning: rounding test data!"
	      #   m.preds = np.round(m.preds)
	      #print m.preds[:10]
	
	m = finalizeModel(m,use_proba=use_proba)
	
    return(ensemble)
    
def fit_and_score(xmodel,X,y,train,valid,sample_weight=None,scale_wt=None,use_proba=False,returnModel=True):
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
	print "Using sample weight...",sample_weight[train]
	xmodel.fit(Xtrain,ytrain,sample_weight=sample_weight[train])
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
       return local_pred, None

def trainEnsemble(ensemble,mode='linear',score_func='log_loss',useCols=None,addMetaFeatures=False,use_proba=True,dropCorrelated=False,skipCV=False,subfile=""):
    """
    Train the ensemble
    """
    basedir="./share/"

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
    Xtest.colums = Xtrain.columns
    #print "spearman-correlation:\n",Xtrain.corr(method='spearman')
    print "pearson-correlation :\n",Xtrain.corr(method='pearson')

    #print Xtrain.describe()
    print Xtest.shape
    #print Xtest.describe()
   
    if mode is 'classical':
	results=classicalBlend(ensemble,Xtrain,Xtest,y,score_func=score_func,use_proba=use_proba,skipCV=skipCV,subfile=subfile,cv_labels=xmodel.cv_labels,dropCorrelated=dropCorrelated)
    elif mode is 'mean':
	results=linearBlend(ensemble,Xtrain,Xtest,y,score_func=score_func,takeMean=True,subfile=subfile,dropCorrelated=dropCorrelated)
    elif mode is 'voting':
        results=voting_multiclass(ensemble,Xtrain,Xtest,y,score_func=score_func,n_classes=1,subfile=subfile,dropCorrelated=dropCorrelated)
    elif mode is 'oob':
	 return (Xtest,Xtrain,y,None,xmodel.cv_labels,None)
	 
    else:
	results=linearBlend(ensemble,Xtrain,Xtest,y,score_func=score_func,takeMean=False,subfile=subfile,dropCorrelated=dropCorrelated)
    return(results)

def voting_multiclass(ensemble,Xtrain,Xtest,y,n_classes=9,use_proba=False,score_func='log_loss',plotting=True,subfile=None):
    """
    Voting for multi classifiction result
    """
    if use_proba:
      print "Majority voting for predictions using proba"
      voter = np.reshape(Xtrain.values,(Xtrain.shape[0],-1,n_classes)).swapaxes(0,1)

      for model in voter:
	  max_idx=model.argmax(axis=1)
	  for row,idx in zip(model,max_idx):
	      row[:]=0.0
	      row[idx]=1.0
      
      voter = voter.mean(axis=0)
      print voter
      print voter.shape
    else:
      print "Majority voting for predictions"
      #assuming all classes are predicted
      if Xtrain.shape[1]%2==0:
	  print "Warning: Even number of voters..."
      
      classes = np.unique(Xtrain.values)
      
      votes_train = np.zeros((Xtrain.shape[0],classes.shape[0]))
      votes_test = np.zeros((Xtest.shape[0],classes.shape[0]))
      
      for i,c in enumerate(classes):
	  votes_train[:,i] = np.sum(Xtrain.values==c,axis=1)
	  votes_test[:,i] = np.sum(Xtest.values==c,axis=1)
      
      votes_train = np.argmax(votes_train,axis=1)
      votes_test = np.argmax(votes_test,axis=1)
      
      encoder= preprocessing.LabelEncoder()
      encoder.fit(y)
      ypred = encoder.inverse_transform(votes_train)
      preds = encoder.inverse_transform(votes_test)
      
      score=funcdict[score_func](y, ypred)
      print score_func+": %0.3f" %(score)
      
      
    if subfile is not None:
	analyze_predictions(ypred,preds)
	makePredictions(None,Xtest=preds,filename=subfile)
	
	if plotting:
	  plt.hist(ypred,bins=50,alpha=0.3,label='oob')
	  plt.hist(preds,bins=50,alpha=0.3,label='pred')
	  plt.legend()
	  plt.show()
	
    else:
	return score
      
      
      
def analyze_predictions(ypred,preds):
    #ypred = ypred.astype(int)
    plt.hist(ypred,bins=50,alpha=0.3,label='oob')
    plt.hist(preds,bins=50,alpha=0.3,label='pred')
    plt.legend()
    plt.show()
  

def preprocess(oobpreds,testset,verbose=False):
    #print "Clipping data  data..."
    #lowerb = 0.41
    #upperb = 6.91
    #oobpreds = oobpreds.clip(lower=lowerb,upper=upperb,axis=0)
    #testset = pd.DataFrame(np.clip(testset.values,lowerb, upperb))#all labels are the same!

    #overfittet models
    noise_columns=[]#['bagxgb5_br1_Class','nn7_br25_Class']
    print "Adding random noise:",noise_columns
    for col in noise_columns:
	if col in oobpreds.columns:
	  oobpreds[col] = oobpreds[col].map(lambda x: x + np.random.normal(loc=0.0, scale=.05))
    
    if verbose:
	oobpreds.describe()
	showCorrelations(oobpreds)
    
    return  oobpreds,testset

def classicalBlend(ensemble,oobpreds,testset,ly,use_proba=True,score_func='log_loss',subfile=None,cv=5,skipCV=False,**kwargs):
    """
    Blending using sklearn classifier
    """
    oobpreds,testset = preprocess(oobpreds,testset)

    if kwargs['dropCorrelated']:
	#showCorrelations(oobpreds)
	oobpreds,testset = removeCorrelations(oobpreds,testset,0.99)
	print oobpreds.shape
	
    blender=Ridge(alpha=10.0)#0.212644
    #blender = Pipeline([('pca', PCA(n_components=19,whiten=False)), ('model', LinearRegression())])
    #blender = Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=1.0))])    
    blender = ConstrainedLinearRegressor(lowerbound=0,upperbound=.25, n_classes=1, alpha=None, corr_penalty = None, normalize=False,loss='rmse',greater_is_better=False)#0.216467
    #blender=ExtraTreesRegressor(n_estimators=500,max_depth=None,min_samples_leaf=5,n_jobs=4,criterion='mse', max_features=4*oobpreds.shape[1]/5,oob_score=False)#0.215702
    #blender = Pipeline([('ohc', OneHotEncoder(sparse=False)), ('model',ExtraTreesClassifier(n_estimators=300,max_depth=None,min_samples_leaf=7,n_jobs=4,criterion='gini', max_features=3,oob_score=False))])
    #blender=RandomForestRegressor(n_estimators=100,max_depth=None,min_samples_leaf=10,n_jobs=1,criterion='entropy', max_features=5,oob_score=False)
    #blender = XgboostRegressor(n_estimators=200,learning_rate=0.03,max_depth=2,subsample=.5,n_jobs=4,min_child_weight=1,objective='reg:linear',eval_metric='rmse',booster='gbtree',silent=1)#0.216854
    
    #blender = Pipeline([('scaler', StandardScaler()), ('model',nnet_ensembler3)])
    #blender = Pipeline([('scaler', StandardScaler()), ('model', blender)])    
    
    #blender = BaggingRegressor(base_estimator=blender,n_estimators=25,n_jobs=1,verbose=2,random_state=None,max_samples=.5,max_features=.5,bootstrap=False)
    
    if not skipCV:
	#blender = CalibratedClassifierCV(baseblender, method='sigmoid', cv=3)
	#cv = KFold(ly.shape[0], n_folds=10,shuffle=True)
	print kwargs['cv_labels']
	cv = KLabelFolds(pd.Series(kwargs['cv_labels']), n_folds=2, repeats =1)
	#cv = LeavePLabelOutWrapper(ta,n_folds=8,p=1)
	#score_func = make_scorer(funcdict[score_func], greater_is_better = False)
	#parameters = {'n_estimators':[300],'max_depth':[3],'learning_rate':[0.03],'subsample':[0.5],'colsample_bytree':[0.5],'min_child_weight':[1]}#XGB
	#parameters = {'n_estimators':[200,300],'max_features':[5,7],'min_samples_leaf':[1,5,10],'criterion':['mse']}#XGB
	#parameters = {'max_features':[0.9,0.95,1.0],'max_samples':[0.9,0.95,1.0],'bootstrap':[False,True]}#XGB
	#parameters = {'model__hidden1_num_units': [128],'model__dropout1_p':[0.0],'model__hidden2_num_units': [128],'model__dropout2_p':[0.0],'model__max_epochs':[75],'model__objective_alpha':[0.002]}
	#blender=makeGridSearch(blender,oobpreds,ly,n_jobs=1,refit=False,cv=cv,scoring=score_func,parameters=parameters,random_iter=-1)
	blend_scores=np.zeros(len(cv))
	n_classes = 1 
	blend_oob=np.zeros((oobpreds.shape[0],n_classes))
	print blender
	for i, (train, test) in enumerate(cv):
	    clf = clone(blender)
	    Xtrain = oobpreds.iloc[train]
	    Xtest = oobpreds.iloc[test]
	    clf.fit(Xtrain.values, ly[train])	
	    if use_proba:
		blend_oob[test] = clf.predict_proba(Xtest)
	    else:
		blend_oob[test] = clf.predict(Xtest).reshape(blend_oob[test].shape)
	    blend_scores[i]=funcdict[score_func](ly[test],blend_oob[test])
	    print "Fold: %3d <%s>: %0.6f ~mean: %6.4f std: %6.4f" % (i,score_func,blend_scores[i],blend_scores[:i+1].mean(),blend_scores[:i+1].std())
	
	print " <"+score_func+">: %0.6f (+/- %0.4f)" % (blend_scores.mean(), blend_scores.std()),
	oob_auc=funcdict[score_func](ly,blend_oob)
	#showMisclass(ly,blend_oob,oobpreds,index=kwargs['cv_labels'])
	print " "+score_func+": %0.6f" %(oob_auc)
	
	if subfile is not None:
	  print "Make full model fit..."
	  blender.fit(oobpreds,ly)

	  
	if hasattr(blender,'coef_'):
	  print "%-3s %-24s %10s %10s" %("nr","model",score_func,"coef")
	  for i,model in enumerate(oobpreds.columns):
	    coldata=np.asarray(oobpreds.iloc[:,i])
	    score=funcdict[score_func](ly, coldata)
	    print "%-3d %-24s %10.4f%10.4f" %(i+1,model.replace("_Class",""),score,blender.coef_[i])
	  print "sum coef: %4.4f"%(np.sum(blender.coef_))

	if subfile is not None:
	  info_dist(ly,"orig")
	  info_dist(blender.predict(oobpreds),"fit")
    
    if subfile is not None:
	print "Make final ensemble prediction..."
	#blend results
	if use_proba:
	  preds=blender.predict_proba(testset)
	else:
	  preds=blender.predict(testset)
	  preds=preds.flatten()
	  
	#print preds
	info_dist(preds,"preds")
	makePredictions(None,preds,filename=subfile)
	analyze_predictions(blend_oob,preds)

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

def linearBlend(ensemble,Xtrain,Xtest,y,score_func='log_loss',greater_is_better=False,use_proba=False,normalize=False,removeZeroModels=-1,takeMean=False,alpha=None,subfile=None,plotting=False,**kwargs):
    """
    Blending for multiclass systems
    """
    def fopt(params):
	# nxm  * m*1 ->n*1
	if np.isnan(np.sum(params)):
	    print "We have NaN here!!"
	    score=0.0
	else:
	    ypred = blend_mult(Xtrain,params,n_classes)
	    #if not use_proba: ypred = np.round(ypred).astype(int)
	    score=funcdict[score_func](y,ypred)
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
    
    lowerbound=0.0
    upperbound=.5
    constr=None
    constr=[lambda x,z=i: x[z]-lowerbound for i in range(n_models)]
    constr2=[lambda x,z=i: upperbound-x[z] for i in range(n_models)]
    constr=constr+constr2
      
    cons = ({'type': 'ineq', 'fun': [lambda x,z=i: x[z]-lowerbound for i in range(n_models)]},
            {'type': 'ineq', 'fun': [lambda x,z=i: upperbound-x[z] for i in range(n_models)]})

    x0 = np.ones((n_models, 1)) / float(n_models)
    
    if not takeMean:
      xopt = fmin_cobyla(fopt, x0,constr,rhoend=1e-10,maxfun=10000)
      #xopt = minimize(fopt, x0,method='Nelder-Mead')
      #xopt = minimize(fopt, x0,method='COBYLA',constraints=cons)
      print xopt
      #xopt = xopt.x
    else:
      xopt = x0
	

    
    #normalize coefficient
    if normalize: 
	  xopt=xopt/np.sum(xopt)
	  print "Normalized coefficients:",xopt

    if np.isnan(np.sum(xopt)):
    	    print "We have NaN here!!"
    
    ypred=blend_mult(Xtrain,xopt,n_classes)
    #ymean= blend_mult(Xtrain,x0,n_classes).flatten()
    ymean=np.mean(Xtrain.values,axis=1)
    #ymean=np.median(Xtrain.values,axis=1)
    
    if takeMean:
      print "Taking the mean/median..."
      ypred = ymean
    
    #print ymean[:10]
    #if not use_proba:
    #  ymean = np.round(ymean+1E-2).astype(int)
    #  ypred = np.round(ypred+1E-6).astype(int)
      
    print "ypred:",ypred.sum()
    print "ypred:",ypred
    print "ymean:",ymean.sum()
    print "ymean:",ymean
    
    score=funcdict[score_func](y,ymean)
    print "->score,mean: %4.4f" %(score)
    oob_score=funcdict[score_func](y,ypred)
    print "->score,opt: %4.4f" %(oob_score)
       
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
    print "training -  max: %4.2f mean: %4.2f median: %4.2f min: %4.2f"%(np.amax(ypred),ypred.mean(),np.median(ypred),np.amin(ypred))
    
    
    if removeZeroModels>0.0:
	print "Dropping ",len(zero_models)," columns:",zero_models
	Xtrain=Xtrain.drop(zero_models,axis=1)
	Xtest=Xtest.drop(zero_models,axis=1)
	return (Xtrain,Xtest)
    
    #prediction flatten makes a n-dim row vector from a nx1 column vector...
    if takeMean:
	print "Taking the mean/median for predictions..."
	preds=np.mean(Xtest.values,axis=1)
    else:
	preds = blend_mult(Xtest,xopt,n_classes).flatten()
    #if not use_proba: preds = np.round(preds).astype(int)

    if subfile is not None:
	print "predictions - max: %4.2f mean: %4.2f median: %4.2f min: %4.2f"%(np.amax(preds),preds.mean(),np.median(preds),np.amin(preds))
	analyze_predictions(ypred,preds)
	#makePredictions(None,Xtest=preds,filename=subfile)
	makePredictions(None,Xtest=preds,idx=None,filename=subfile,log1p=True)
    else:
	return oob_score
      

def info_dist(y,info):
    print info+"-  max: %4.2f mean: %4.2f median: %4.2f min: %4.2f"%(np.amax(y),y.mean(),np.median(y),np.amin(y))
  
def selectModels(ensemble,startensemble=[],niter=10,mode='linear',useCols=None): 
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
	    #score=trainEnsemble(actensemble,mode=mode,useCols=None,use_proba=False)
	    #score = trainEnsemble(actensemble,mode=mode,score_func='quadratic_weighted_kappa',use_proba=False,subfile=None)
	    score = trainEnsemble(actensemble,mode=mode,score_func='rmse',use_proba=False,useCols=None,subfile=None,dropCorrelated=dropCorrelated)
	    print "##(Current top score: %4.4f | overall best score: %4.4f) current score: %4.4f  - " %(maxscore,bestscore,score)
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
    np.random.seed(123)
    ta = None
    """
    # 1nd LEVEL MODEL BUILDING			     
    """
    ensemble=createModels()
    ensemble=createOOBdata(ensemble,repeats=20,n_folds=10,n_jobs=1,use_proba=False,score_func='rmse') #oob data averaging leads to significant variance reduction
    #all_models_old=['xgb1_r1','xgb2_r1','xgb3_r1','xrf1_r1','rf1_r1','rf2_r1','knn1_r1','lr1_r1','bagxgb1_r1','bagxgb2_r1','xgb4_r1','rf3_r1','svm1_r1','nn1_r1','gbr1_r1','lr2_r1','svm2_r1']
    #all_models = ['xgb2_r1', 'nn1_r1','nn2_r1','nn5_r1','nn6_r1', 'gbr1_r1','lr2_r1','xrf1_r1','xrf2_r1','svm2_r1','xgb4_r1','rf3_r1','rf1_r1','xgb1_r1','xgb3_r1','knn1_r1','bagxgb5_r1','xgb5_r1','xgb6_r1','xgb7_r1']
    #all_models_no_bagmode = ['xgb2_r1', 'nn1_r1','nn2_r1','nn6_r1', 'gbr1_r1','lr2_r1','xrf1_r1','xrf2_r1','svm2_r1','xgb4_r1','rf3_r1','rf1_r1','xgb1_r1','xgb3_r1','knn1_r1','xgb5_r1','xgb6_r1','xgb7_r1']
    
    """
    # 1nd LEVEL ENSEMBLING			     
    """
    all_models_bagmode = ['xgb16_br1','xgb17_br1','xgb18_br1','xgb2_br1','xgb3_br1','xgb4_br1','xgb5_br1','xgb6_br1','xgb7_br1','xgb8_br1','xgb9_br1','xgb10_br1','xgb_discount_br1','xgb13_br1','nn6_br1','xrf1_br1','xrf2_br1','xrf3_br1','xrf4_br1','xrf_discount_br1','xrf6_br1','rf1_br1','rf3_br1','nn2_br1','nn1_br1','lr2_br1','lr3_br1','quantity','quantity_inv','gbr1_br1','lr3_br1','annual_usage','svm2_br1','knn1_br1','bagxgb5_br1','nn7_br25','keras1_br1','keras2_br1','rf4_br1','nn8_br1','bracket_pricing']
    all_models_bagmode_manually = ['xgb5_br1','xgb6_br1','xgb7_br1','xgb8_br1','xgb9_br1','xgb10_br1','xgb17_br1','xgb18_br1','gbr1_br1','nn6_br1','xrf2_br1','xrf4_br1','xrf5_br1','rf1_br1','rf4_br1','nn2_br1','lr2_br1','quantity_inv','annual_usage','bagxgb5_br1','nn7_br25','nn8_br1','nn9_br20','bracket_pricing','nn9_br20']
    gbm_models=['xgb5_br1','xgb6_br1','xgb7_br1','xgb8_br1','xgb9_br1','xgb10_br1','xgb16_br1','xgb17_br1','gbr1_br1']
    best_submission=['xgb2_br1','xgb3_br1','xgb4_br1','xgb5_br1','xgb6_br1','xgb7_br1','xgb8_br1','xgb9_br1','xgb10_br1','nn6_br1','xrf1_br1','xrf2_br1','xrf3_br1','xrf4_br1','rf1_br1','rf3_br1','nn2_br1','nn1_br1','lr2_br1','quantity','gbr1_br1','lr3_br1','annual_usage','svm2_br1','knn1_br1','bagxgb5_br1','nn7_br25']
    best_submission_nofinance=['xgb2_br1','xgb3_br1','xgb4_br1','xgb5_br1','xgb6_br1','xgb7_br1','xgb8_br1','xgb9_br1','xgb10_br1','nn6_br1','xrf1_br1','xrf2_br1','xrf3_br1','xrf4_br1','rf1_br1','rf3_br1','nn2_br1','nn1_br1','lr2_br1','quantity','gbr1_br1','lr3_br1','annual_usage','svm2_br1','knn1_br1','bagxgb5_br1','nn7_br25']
    best_submission_nofinance_nofeat=['xgb2_br1','xgb3_br1','xgb4_br1','xgb5_br1','xgb6_br1','xgb7_br1','xgb8_br1','xgb9_br1','xgb10_br1','nn6_br1','xrf1_br1','xrf2_br1','xrf3_br1','xrf4_br1','rf1_br1','rf3_br1','nn2_br1','nn1_br1','lr2_br1','gbr1_br1','lr3_br1','svm2_br1','knn1_br1','bagxgb5_br1','nn7_br25']
    #all_models_bagmode_old = ['xgb2_br1','xgb3_br1','xgb4_br1','xgb5_br1','xgb6_br1','xgb7_br1','xgb8_br1','xgb9_br1','xgb10_br1','nn6_br1','xrf1_br1','xrf2_br1','xrf3_br1','xrf4_br1','rf1_br1','rf3_br1','nn2_br1','nn1_br1','lr2_br1','quantity','gbr1_br1','lr3_br1','annual_usage','svm2_br1','knn1_br1','bagxgb5_br1','nn7_br25']
    #best_models = ['xgb2_r1', 'nn1_r1', 'gbr1_r1','svm2_r1','xgb4_r1','xgb5_r1','xgb6_r1','nn6_r1','xgb6_r1','xgb7_r1','xrf2_r1']
    #models = ['svm7_br3','xrf4_br3','nn1_br3']
    #manual=['xgb7_r1','nn6_r1']
    #models = ['xgb2_br1','xgb3_br1','xgb4_br1','xgb5_br1','xgb6_br1','xgb7_br1','xgb8_br1','xgb9_br1']#['xgb2_r1','nn1_r1','gbr1_r1','xrf1_r1','knn1_r1','lr1_r1']
    models = ['nn9_br20']#best_submission_nofinance_nofeat# ['nn8_br1']#all_models_bagmode_manually#all_models_bagmode_manually#['xgb16_br1','xgb5_br1','xgb6_br1','xgb7_br1','xgb8_br1','xgb9_br1','xgb10_br1']
    useCols=None
    trainEnsemble(models,mode='classical',score_func='rmse',useCols=None,addMetaFeatures=False,use_proba=False,dropCorrelated=False,subfile='./submissions/sub23082015a.csv')
    #selectModelsGreedy(models,startensemble=['xgb7_r1','nn6_r1','quantity','annual_usage','xgb_discount_br1'],niter=11,mode='classical',greater_is_better=False,dropCorrelated=True)
   
    
    """
    # 2nd LEVEL MODEL BUILDING			     
    """
    #ensemble2 = createModels_stage2(all_models_bagmode_manually)
    #createOOBdata(ensemble2,repeats=1,n_folds=10,n_jobs=1,use_proba=False,score_func='rmse') #we need some iterations 
    """
    # 2nd LEVEL ENSEMBLING			     
    """
    l2_models = ['nn3_l2']
    #trainEnsemble(l2_models,mode='classical',score_func='rmse',useCols=None,addMetaFeatures=False,use_proba=False,dropCorrelated=True,subfile='./submissions/sub15082015d.csv')
    
    
