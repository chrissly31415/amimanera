#!/usr/bin/python 
# coding: utf-8
# Code source: Chrissly31415
# License: BSD

import pandas as pd
import numpy as np
import sklearn as sl
import scipy as sp
from sklearn.base import clone
from qsprLib import *
import random

def prepareDatasets(nsamples=-1,onlySpectra=False,deleteSpectra=False,plotting=False,standardize=True,doPCA=5,findPeaks=None,makeDerivative=False):
    Xtrain = pd.read_csv('/home/loschen/Desktop/datamining-kaggle/african_soil/training.csv')
    Xtest = pd.read_csv('/home/loschen/Desktop/datamining-kaggle/african_soil/sorted_test.csv')
    ymat = Xtrain[['Ca','P','pH','SOC','Sand']]
    Xtrain.drop(['Ca', 'P', 'pH', 'SOC', 'Sand', 'PIDN'], axis=1, inplace=True)
    
    if nsamples != -1: 
	rows = random.sample(Xtrain.index, nsamples)
	Xtrain = Xtrain.ix[rows]
      
    Xtest.drop('PIDN', axis=1, inplace=True)
    
    #combine data
    X_all = pd.concat([Xtest, Xtrain],ignore_index=True)
    
    X_all = pd.concat([X_all, pd.get_dummies(X_all['Depth'])],axis=1)
    X_all.drop(['Depth','Subsoil'], axis=1, inplace=True)
    
    if findPeaks is not None:
      if findPeaks is 'load':
	  X_ir = peakfinder('load')
      else:
	  X_ir = peakfinder(X_all)
    
      #X_all.drop([:3578], axis=1, inplace=True)
      X_all = pd.concat([X_all, X_ir],axis=1)
      #remove zero columns
      X_all = X_all.ix[:,(X_all != 0).any(axis=0)]
    
    if makeDerivative:
	X_diff = firstDerivative(X_all.iloc[:,:3578])
	X_all = pd.concat([X_all, X_diff],axis=1)
	X_all = X_all.ix[:,(X_all != 0).any(axis=0)]
    
    if onlySpectra:
	X_all = X_all.iloc[:,:3578]
	
    if deleteSpectra:
	X_all = X_all.iloc[:,3578:-1]
    
    if standardize:
	X_all = scaleData(X_all)
	
    if doPCA is not None:
	pca = PCA(n_components=doPCA)
	X_r = pca.fit_transform(np.asarray(X_all)) 
	print "explained variance:",pca.explained_variance_ratio_
	print "components: %5d sum: %6.2f"%(doPCA,pca.explained_variance_ratio_.sum())
       
    #split data again
    Xtrain = X_all[len(Xtest.index):]
    Xtest = X_all[:len(Xtest.index)]
    
    #print Xtrain.describe()
    #print Xtest.describe()
    
    #analyze data
    if plotting and not onlySpectra:
	Xtrain.hist(bins=50)
	ymat.hist(bins=50)
	plt.show()
    
    print X_all.describe()
    print X_all.columns
    print "Dim train set:",Xtrain.shape    
    print "Dim test set :",Xtest.shape
    
    
    return(Xtrain,Xtest,ymat)

def firstDerivative(X):
    """
    make derivative
    """
    print "Making 1st derivative..."
    tutto=[]
    for ind in X.index:
	row=[]
	row.append(ind)
	for el in np.diff(X.ix[ind,:].values):
	    row.append(el)
	tutto.append(row)
    newdf = pd.DataFrame(tutto).set_index(0)
    colnames = [ "diff_"+str(x) for x in xrange(newdf.shape[1]) ]
    newdf.columns=colnames
    #newdf.iloc[3,:].plot()
    #plt.show()
    
    #print newdf.head(10)
    #print newdf.describe()
    return(newdf)
    
def peakfinder(X_all,verbose=False):
    """
    Finds peak in spectrum, and creates dataframe from it
    """
    from scipy import signal
    #xs = np.arange(0, 4*np.pi, 0.05)
    #data = np.sin(xs)
    
    print "Locating IR-peaks..."
    if X_all is "load":
	newdf = pd.read_csv('/home/loschen/Desktop/datamining-kaggle/african_soil/irpeaks.csv')
	return(newdf)
    
    print X_all.describe()
    tutto=[]
    print X_all.index
    print X_all.index[-1]
    for ind in X_all.index:
	print "processing spectra %4d/%4d "%(ind,X_all.shape[0])
	row=[]
	row.append(ind)
	data = X_all.ix[ind,:3578].values.flatten()
	peakind = signal.find_peaks_cwt(data, widths=np.arange(5,25),max_distances=np.arange(5,25)/5,noise_perc=10,min_snr=1,min_length=1,gap_thresh=5)
	(hi,edges) = np.histogram(peakind,bins=100,range=(0,4000))
	edges=edges[0:-1]
	for el in hi:
	    row.append(el)
	tutto.append(row)
	
	if verbose:
	    print hi
	    plt.plot(data)
	    plt.bar(edges,hi>0)
	    plt.show()
	    
    newdf = pd.DataFrame(tutto).set_index(0)
    colnames = [ "m"+str(x) for x in edges.tolist() ]
    newdf.columns=colnames   
    #print newdf.head(10)
    #print newdf.describe()
    newdf.to_csv("/home/loschen/Desktop/datamining-kaggle/african_soil/irpeaks.csv")
    return(newdf)
    
    
def buildmodels(lmodels,lX,lymat,fit_params=None,scoring='mean_squared_error',cv_split=8,n_jobs=8):
    cv = cross_validation.ShuffleSplit(lX.shape[0], n_iter=cv_split, test_size=0.25, random_state=0)
    scores=np.zeros((lymat.shape[1],cv_split))
    for i in range(lymat.shape[1]):
	print "TARGET: %-12s"%(lymat.columns[i]),
	ly = lymat.iloc[:,i].values
	#be carefull sign is flipped
	scores[i,:] = (-1*cross_validation.cross_val_score(lmodels[i],lX,ly,fit_params=fit_params, scoring='mean_squared_error',cv=cv,n_jobs=n_jobs))**0.5
	print " - <score>= %0.4f (+/- %0.4f) runs: %4d" % (scores[i].mean(), scores[i].std(),scores.shape[1])
	#FIT FULL MODEL
	lmodels[i].fit(lX,ly)

    print 
    print "Total cv-score: %6.3f (+/- %6.3f) "%(scores.mean(axis=1).mean(),scores.mean(axis=1).std())
    return(models)
    

    
def makePrediction(models,Xtrain,Xtest,nt,filename='subXXX.csv'):
    preds = np.zeros((Xtest.shape[0], nt))
    f, axarr = plt.subplots(nt, sharex=True)
    rmse_list = np.zeros((nt,1))
    for i in range(nt):
	y_true = np.asarray(ymat.iloc[:,i])
	y_pred = models[i].predict(Xtrain).astype(float)
	
	rmse_score = np.sqrt(mean_squared_error(y_true, y_pred))
	print "TARGET: %-10s RMSE,train: %6.3f" %(ymat.columns[i],rmse_score)
	preds[:,i] = models[i].predict(Xtest).astype(float)
	axarr[i].scatter(y_pred,y_true)
	axarr[i].set_ylabel(ymat.columns[i])
	rmse_list[i]=rmse_score
    
    print "<RMSE,train>: %6.3f +/- %6.3f" %(rmse_list.mean(),rmse_list.std())
    plt.show()
    sample = pd.read_csv('/home/loschen/Desktop/datamining-kaggle/african_soil/sample_submission.csv')
    sample['Ca'] = preds[:,0]
    sample['P'] = preds[:,1]
    sample['pH'] = preds[:,2]
    sample['SOC'] = preds[:,3]
    sample['Sand'] = preds[:,4]
    sample.to_csv(filename, index = False)
    
    
    
if __name__=="__main__":
    #TODO https://gist.github.com/sixtenbe/1178136
    #TODO http://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks_cwt.html
    #TODO 2nd derivative using np.diff twice
  
    t0 = time()
	
    print "numpy:",np.__version__
    print "scipy:",sp.__version__
    print "pandas:",pd.__version__
    print "sklearn:",sl.__version__
    pd.set_option('display.max_columns', 14)
    pd.set_option('display.max_rows', 40)

    nsamples=-1
    onlySpectra=False
    deleteSpectra=False
    plotting=False
    standardize=True
    doPCA=None
    findPeaks='load'
    makeDerivative=False
    
    (Xtrain,Xtest,ymat) = prepareDatasets(nsamples,onlySpectra,deleteSpectra,plotting,standardize,doPCA,findPeaks,makeDerivative)
    #ymat = ymat.iloc[:,0:3]
    nt = ymat.shape[1]

    #generate models
    models=[]
    for i in range(nt):
	#model = Pipeline([('filter', SelectPercentile(f_regression, percentile=5)), ('model', LinearRegression())])
	#model = SGDRegressor(alpha=0.1,n_iter=50,shuffle=True,loss='squared_loss',penalty='l1')#too many features
	#model = Pipeline([('filter', SelectPercentile(f_regression, percentile=25)), ('model', SGDRegressor(alpha=0.00001,n_iter=150,shuffle=True,loss='squared_loss',penalty='l2'))])
	#model = Pipeline([('pca', PCA(n_components=doPCA)), ('model', LinearRegression())])
	#model = Pipeline([('pca', PCA(n_components=doPCA)), ('model', SGDRegressor(alpha=0.00001,n_iter=150,shuffle=True,loss='squared_loss',penalty='l2'))])
	#model = Pipeline([('pca', PCA(n_components=200)), ('model', RandomForestRegressor(n_estimators=250,max_depth=None,min_samples_split=2,min_samples_leaf=5,n_jobs=1,criterion='mse', max_features='auto',oob_score=False))])
	#model = Pipeline([('svd', TruncatedSVD(n_components=25, algorithm='randomized', n_iter=5, tol=0.0)), ('model', RandomForestRegressor(n_estimators=250,max_depth=None,min_samples_split=2,min_samples_leaf=5,n_jobs=1,criterion='mse', max_features='auto',oob_score=False))])
	#model = Pipeline([('pca', PCA(n_components=doPCA)), ('model', SVR(C=1.0, gamma=0.0, verbose = 0))])
	
	model = SVR(C=10000.0, gamma=0.0, verbose = 0)
	#model = SVR(C=10000.0, gamma=0.0005, verbose = 0)
	#model = RandomForestRegressor(n_estimators=250,max_depth=None,min_samples_split=2,min_samples_leaf=5,n_jobs=8,criterion='mse', max_features='auto',oob_score=False)
	#model = LinearRegression(normalize=False)
	models.append(model) 
    
    
    
    #make the training
    models = buildmodels(models,Xtrain,ymat,cv_split=16)
    print models[0]
    #makePrediction(models,Xtrain,Xtest,nt,filename='/home/loschen/Desktop/datamining-kaggle/african_soil/submissions/sub2009a.csv'):
    #make the predictions   
  

    print("Model building done on %d samples in %fs" % (Xtrain.shape[0],time() - t0))

