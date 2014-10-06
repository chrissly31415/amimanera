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
from savitzky_golay import *

def prepareDatasets(nsamples=-1,onlySpectra=False,deleteSpectra=False,plotting=False,standardize=True,doPCA=5,findPeaks=None,makeDerivative=None,featureFilter=None,loadFeatures=None,deleteFeatures=None,removeVar=0.0,removeCor=0.99,useSavitzkyGolay=True,addNoiseColumns=None,addLandscapes=False):
    Xtrain = pd.read_csv('/home/loschen/Desktop/datamining-kaggle/african_soil/training.csv')
    Xtest = pd.read_csv('/home/loschen/Desktop/datamining-kaggle/african_soil/sorted_test.csv')
    ymat = Xtrain[['Ca','P','pH','SOC','Sand']]
    Xtrain.drop(['Ca', 'P', 'pH', 'SOC', 'Sand', 'PIDN'], axis=1, inplace=True)
    
    if nsamples != -1: 
	rows = random.sample(Xtrain.index, nsamples)
	Xtrain = Xtrain.ix[rows]
      
    Xtest.drop('PIDN', axis=1, inplace=True)
    
    #combine data
    X_orig = pd.concat([Xtest, Xtrain],ignore_index=True)
    
    X_all = pd.concat([X_orig, pd.get_dummies(X_orig['Depth'])],axis=1)
    X_all.drop(['Depth','Subsoil'], axis=1, inplace=True)
    X_all.rename(columns = {'Topsoil':'Depth'})
    
    if findPeaks is not None and not False:
      if findPeaks is 'load':
	  X_ir = peakfinder('load')
      else:
	  X_ir = peakfinder(X_all)
      print "X_ir",X_ir['int3720.0'].values
      X_all = pd.concat([X_all, X_ir],axis=1)
      print "da shape",X_all.shape
      #remove zero columns
      #X_all = X_all.ix[:,(X_all != 0).any(axis=0)]
    
    if makeDerivative is not None:
	X_diff = makeDiff(X_all.iloc[:,:3578])
	if '2nd' in makeDerivative:
	    X_diff = makeDiff(X_diff)
	X_all = pd.concat([X_all, X_diff],axis=1)
	X_all = X_all.ix[:,(X_all != 0).any(axis=0)]
    
    if onlySpectra:
	X_all = X_all.iloc[:,:3578]

    if useSavitzkyGolay:
	spectra = [m for m in list(X_all.columns) if m[0]=='m']
	X_SP = applySavitzkyGolay(X_all[spectra])
	X_all.drop(spectra, axis=1, inplace=True) 
	X_all = pd.concat([X_all, X_SP],axis=1)
	
    if deleteSpectra:
	spectra = [m for m in list(X_all.columns) if m[0]=='m']
	X_all.drop(spectra, axis=1, inplace=True) 
    
    if featureFilter is not None:
	print "Using featurefilter..."
	X_all=X_all[featureFilter]
    

	
    if doPCA is not None:
	pca = PCA(n_components=doPCA)
	X_all = pd.DataFrame(pca.fit_transform(np.asarray(X_all)))
	for col in X_all.columns:
	    X_all=X_all.rename(columns = {col:'pca'+str(col+1)})
	
	print "explained variance:",pca.explained_variance_ratio_
	print "components: %5d sum: %6.2f"%(doPCA,pca.explained_variance_ratio_.sum())
      
    if deleteFeatures is not None:
	X_all.drop(deleteFeatures, axis=1, inplace=True) 
    
    if addLandscapes is True:
	groups = pd.read_csv('/home/loschen/Desktop/datamining-kaggle/african_soil/groupings.csv',sep=';')
	groups['TMAP']=np.round(groups['TMAP'],5)
	X_all['TMAP']=np.round(X_orig['TMAP'],5)
	X_all = pd.merge(X_all, groups, on='TMAP')
	X_all.drop(['TMAP'], axis=1, inplace=True)
	
	X_all = pd.concat([X_all, pd.get_dummies(X_all['LANDSCAPE'])],axis=1)
	X_all.drop(['LANDSCAPE'], axis=1, inplace=True)
	
	print X_all
	print X_all.describe()
    
    if addNoiseColumns is not None:
	Xrnd = pd.DataFrame(np.random.randn(X_all.shape[0],addNoiseColumns))
	#print "Xrnd:",Xrnd.shape
	#print Xrnd
	for col in Xrnd.columns:
	    Xrnd=Xrnd.rename(columns = {col:'rnd'+str(col+1)})
	
	X_all = pd.concat([X_all, Xrnd],axis=1)
    
    if removeCor is not None:
	X_all = removeCorrelations(X_all,removeCor)

   
    if standardize:
	X_all = scaleData(X_all)
	
    X_all = removeLowVariance(X_all,removeVar)
    
    #split data again
    Xtrain = X_all[len(Xtest.index):]
    Xtest = X_all[:len(Xtest.index)]
      
    
    
    #print Xtrain.describe()
    #print Xtest.describe()
    
    #analyze data
    #if plotting:
	#axs = Xtrain.hist(bins=30,color='b',alpha=0.3)
	#for ax, (colname, values) in zip(axs.flat, Xtest.iteritems()):
	    #values.hist(ax=ax, bins=30,color='g',alpha=0.3)
	#plt.show()
    
    if loadFeatures is not None:
	tmp1 = pd.read_csv('/home/loschen/Desktop/datamining-kaggle/african_soil/training_all.csv')[loadFeatures]
	tmp1.index = Xtrain.index
	Xtrain = pd.concat([Xtrain,tmp1],axis=1)
	tmp2 = pd.read_csv('/home/loschen/Desktop/datamining-kaggle/african_soil/test_all.csv')[loadFeatures]
	tmp2.index = Xtest.index
	print tmp2.head(10)
	print Xtest.head(10)
	Xtest = pd.concat([Xtest,tmp2],axis=1)
   
    #analyze data
    if plotting:
	Xtrain.hist(bins=50)
	Xtest.hist(bins=50)
      
	#somehow ordering is wrong???
	#axs2 = Xtrain.hist(bins=30)
	#for ax2, (colname, values) in zip(axs2.flat, Xtest.iteritems()):
	#    values.hist(ax=ax2,alpha=0.3, bins=30)
	plt.show()
    
    print Xtest.describe()
    #print Xtest.index
    print "Dim train set:",Xtrain.shape    
    print "Dim test set :",Xtest.shape
    
    #Xtrain.to_csv("/home/loschen/Desktop/datamining-kaggle/african_soil/training_all.csv",index=False)
    #Xtest.to_csv("/home/loschen/Desktop/datamining-kaggle/african_soil/test_all.csv",index=False)
    
    return(Xtrain,Xtest,ymat)

def applySavitzkyGolay(X,window_size=31, order=3,plotting=False):
    """
    use filter
    """
    print "Use Savitzky-Golay (windowsize=%4d, order=%4d)"%(window_size,order)
    tutto=[]
    for ind in X.index:
	row=[]
	row.append(ind)
	yorig = X.ix[ind,:].values
	ysg = savitzky_golay(yorig, window_size=41, order=3)
	
	for el in ysg:
	    row.append(el)
	tutto.append(row)
	#
	if plotting:
	    plt.plot(yorig, label='Noisy signal')
	    plt.plot( ysg, 'r', label='Filtered signal')
	    plt.legend(loc="best")
	    plt.show()
	
    newdf = pd.DataFrame(tutto).set_index(0)
    colnames = [ "sago_"+str(x) for x in xrange(newdf.shape[1]) ]
    newdf.columns=colnames
    print newdf.head(10)
    return(newdf)
    
    
    
def makeDiff(X):
    """
    make derivative
    """
    print "Making 1st derivative..."
    tutto=[]
    for ind in X.index:
	row=[]
	row.append(ind)
	for el in np.gradient(X.ix[ind,:].values):
	    row.append(el)
	tutto.append(row)
    newdf = pd.DataFrame(tutto).set_index(0)
    colnames = [ "diff_"+str(x) for x in xrange(newdf.shape[1]) ]
    newdf.columns=colnames
    newdf.iloc[3,:].plot()
    plt.show()
    
    print newdf.head(10)
    print newdf.describe()
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
    colnames = [ "int"+str(x) for x in edges.tolist() ]
    newdf.columns=colnames   
    #print newdf.head(10)
    #print newdf.describe()
    newdf.to_csv("/home/loschen/Desktop/datamining-kaggle/african_soil/irpeaks.csv",index=False)
    return(newdf)
    
    
def buildmodels(lmodels,lX,lymat,fit_params=None,scoring='mean_squared_error',cv_split=8,n_jobs=8,gridSearch=False):
    cv = cross_validation.ShuffleSplit(lX.shape[0], n_iter=cv_split, test_size=0.25, random_state=0)
    scores=np.zeros((lymat.shape[1],cv_split))
    for i in range(lymat.shape[1]):
	ly = lymat.iloc[:,i].values
	#be carefull sign is flipped
	if gridSearch is True:
	    #parameters = {'filter__percentile': [50,25], 'model__C': [12000,10000,8000],'model__gamma': [0.0] }#pipeline	    
	    #parameters = {'filter__k': [2000,3000,3594],'pca__n_components':[0.99,0.995], 'model__alpha': [10000,100,1.0,0.01,0.0001] }#pipeline
	    #parameters = {'filter__k': [2000,3000,3594],'pca__n_components':[0.99], 'model__alpha': [10000,100,1.0,0.01,0.0001] }#pipeline
	    #parameters = {'filter__k': [3250,3594],'model__gamma':[0.05,0.005,0.0005,0.0001], 'model__C': [100,1000,10000,100000] }#pipeline
	    #parameters = {'filter__param': [99],'model__alpha': [0.001],'model__loss':['huber'],'model__penalty':['elasticnet'],'model__epsilon':[1.0],'model__n_iter':[200]}#pipeline
	    #parameters = {'varfilter__threshold': [0.0,0.1,0.001,0.0001,0.00001] }#pipeline
	    #parameters = {'filter__k': [10,20],'pca__n_components':[10,20], 'model__alpha': [1.0] }#pipeline
	    parameters = {'filter__param': [100]}#pipeline
	    #parameters = {'filter__param': [100,99],'model__n_components':[5,8,10,15]}#pipeline
	    #parameters = {'filter__param': [99],'model__alpha':[1.0],'model__l1_ratio':[0.5]}#KNN
	    #parameters = {'filter__param': [100,99],'model__n_neighbors':[3,4]}#KNN
	    #parameters = {'model__max_depth':[5,6], 'model__learning_rate':[0.1],'model__n_estimators':[200,300,400],'model__subsample':[1.0],'model__loss':['huber'],'model__min_samples_leaf':[10],'model__max_features':[None]}
	    #parameters = {'model_loss': ['ls','lad','huber', 'quantile']}#GBR
	    clf  = grid_search.GridSearchCV(lmodels[i], parameters,n_jobs=n_jobs,verbose=0,scoring=scoring,cv=cv,fit_params=fit_params,refit=True)
	    clf.fit(lX,ly)
	    best_score=1.0E5
	    print("%6s %6s %6s %r" % ("OOB", "MEAN", "SDEV", "PARAMS"))
	    for params, mean_score, cvscores in clf.grid_scores_:
		oob_score = (-1*mean_score)**0.5
		cvscores = (-1*cvscores)**0.5
		mean_score = cvscores.mean()
		print("%6.3f %6.3f %6.3f %r" % (oob_score, mean_score, cvscores.std(), params))
		if mean_score < best_score:
		    best_score = mean_score
		    scores[i,:] = cvscores
		
	    lmodels[i] = clf.best_estimator_
	     
	else:    
	    scores[i,:] = (-1*cross_validation.cross_val_score(lmodels[i],lX,ly,fit_params=fit_params, scoring=scoring,cv=cv,n_jobs=n_jobs))**0.5   

	print "TARGET: %-12s"%(lymat.columns[i]),    
	print " - <score>= %0.3f (+/- %0.3f) runs: %4d" % (scores[i].mean(), scores[i].std(),scores.shape[1])
	#FIT FULL MODEL
	lmodels[i].fit(lX,ly)

    print 
    print "Total cv-score: %6.3f (+/- %6.3f) "%(scores.mean(axis=1).mean(),scores.mean(axis=1).std())
    return(models)
    

def gridSearchModels(lmodels,lX,lymat,fit_params=None,scoring='mean_squared_error',cv_split=8,n_jobs=8):
    """
    Do grid search on several models
    """
    pass
    
    
def makePrediction(models,Xtrain,Xtest,nt,filename='subXXX.csv'):
    preds = np.zeros((Xtest.shape[0], nt))
    f, axarr = plt.subplots(nt, sharex=True)
    rmse_list = np.zeros((nt,1))
    print "%-10s %6s" %("TARGET","RMSE,train")
    for i in range(nt):
	y_true = np.asarray(ymat.iloc[:,i])
	y_pred = models[i].predict(Xtrain).astype(float)
	rmse_score = np.sqrt(mean_squared_error(y_true, y_pred))
	print "%-10s %6.3f" %(ymat.columns[i],rmse_score)
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

def modelsFeatureSelection(lmodels,Xold,Xold_test,lymat):
    for i,model in enumerate(lmodels):
	iterativeFeatureSelection(model,Xold,Xold_test,lymat.iloc[:,i],1,1)

	
def getFeatures(key):
    """
    Collections of some use features
    """
    #all vars
    features={}
    # optimzed R leaps features
    features['Ca']=["m7328.26","m7067.91","m7056.34","m6265.66","m5237.78","m4751.8","m4406.6","m3853.12","m3851.19","m3642.92","m3627.49","m2917.8","m2873.45","m2850.31","m2603.46","m2547.53","m2537.89","m2505.11","m1859.06","m1853.28","m1833.99","m1801.21","m1795.42","m1793.49","m1791.57","m1787.71","m1781.92","m1776.14","m1770.35","m1610.29","m1608.36","m1529.29","m1448.3","m1313.3","m1267.02","m1230.38","m1228.45","m1191.81","m1135.88","m1076.1","m784.895","m779.109","m723.183","m713.541","m711.612","m647.972","m628.687","m599.76","LSTN","REF2"]
    features['P']=["m7494.11","m7353.33","m7254.97","m7235.69","m7173.98","m6265.66","m4528.09","m4524.23","m3752.84","m3704.63","m2917.8","m2873.45","m2850.31","m2547.53","m2537.89","m2514.75","m2360.47","m2335.4","m2065.41","m2061.55","m1735.64","m1712.5","m1687.43","m1610.29","m1513.86","m1506.15","m1448.3","m1442.51","m1425.15","m1232.3","m1230.38","m1228.45","m1106.95","m1105.02","m1079.95","m1078.03","m1074.17","m1064.53","m952.673","m877.462","m873.605","m852.392","m815.751","m813.822","m777.181","m678.828","BSAN","EVI","REF7","RELI"]
    features['pH']=["m7177.83","m7170.12","m4836.65","m4406.6","m3851.19","m3534.92","m3515.63","m3394.14","m2850.31","m2489.68","m2238.98","m2215.83","m2159.91","m2065.41","m1758.78","m1735.64","m1726","m1700.93","m1685.5","m1664.29","m1652.71","m1623.79","m1618","m1583.29","m1575.58","m1438.65","m1338.37","m1294.02","m1270.87","m1130.09","m1079.95","m1072.24","m1024.03","m1008.6","m954.602","m917.961","m908.318","m835.036","m821.536","m788.752","m617.116","m599.76","BSAN","BSAV","ELEV","EVI","LSTD","LSTN","RELI","TMAP"]
    features['SOC']=["m7177.83","m7067.91","m5237.78","m3733.55","m3687.27","m2514.75","m2238.98","m2217.76","m2042.27","m1959.34","m1830.14","m1814.71","m1780","m1756.85","m1716.35","m1710.57","m1687.43","m1673.93","m1627.64","m1621.86","m1585.22","m1562.08","m1544.72","m1542.79","m1519.65","m1496.51","m1477.22","m1243.88","m1191.81","m1164.81","m1162.88","m1160.95","m1114.67","m1110.81","m1097.31","m1054.88","m1022.1","m917.961","m906.39","m894.819","m892.89","m890.962","m869.748","m767.539","m750.182","m701.97","m647.972","REF1","REF2","RELI"]
    features['sand']=["m7490.25","m7459.39","m7347.54","m6265.66","m4836.65","m4751.8","m4510.74","m4196.39","m3658.34","m2159.91","m1922.7","m1920.77","m1841.71","m1772.28","m1687.43","m1685.5","m1637.29","m1592.93","m1488.79","m1394.3","m1326.8","m1324.87","m1191.81","m1164.81","m1162.88","m1160.95","m1157.09","m1155.16","m1130.09","m1097.31","m1078.03","m1037.53","m983.529","m952.673","m850.464","m844.678","m806.108","m802.251","m796.466","m784.895","m779.109","m769.467","m767.539","m723.183","BSAV","ELEV","LSTD","REF1","REF7","TMAP"]
    #only spectra
    return feature[key]
	  
if __name__=="__main__":
    #TODO https://gist.github.com/sixtenbe/1178136
    #TODO http://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks_cwt.html
    #TODO 2nd derivative using np.diff twice, using np.gradient
    #TODO Savitzky-Golay or something selfmade?
    #TODO continuum removal
    #TODO PLS
    #TODO outlier detection for P
    #TODO RFECV
    #TODO polynomial on extra features
    #TODO make TMAP categorical/binary
    #TODO python string to integer:df[0] = df[0].str.replace(r'[$,]', '').astype('float')
    #http://stackoverflow.com/questions/3172509/numpy-convert-categorical-string-arrays-to-an-integer-array
    #http://stackoverflow.com/questions/15356433/how-to-generate-pandas-dataframe-column-of-categorical-from-string-column
    #http://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html

    #boruta results: 
  
    t0 = time()
	
    print "numpy:",np.__version__
    print "scipy:",sp.__version__
    print "pandas:",pd.__version__
    print "sklearn:",sl.__version__
    pd.set_option('display.max_columns', 14)
    pd.set_option('display.max_rows', 40)

    np.random.seed(123)
    nsamples=-1
    onlySpectra=False
    deleteSpectra=False
    plotting=False
    standardize=False
    doPCA=None
    findPeaks=None
    #findPeaks='load'
    makeDerivative=None#CHECK indices after derivative making...
    featureFilter=None#["BSAN","BSAS","BSAV","ELEV","EVI","LSTD","LSTN","REF1","REF2","REF3","REF7","TMAP","TMFI","m120.0","m200.0","m1160.0","m1200.0","m2080.0","m2160.0","m2200.0","m2240.0","m2640.0","m3240.0"]
    removeVar=0.1
    useSavitzkyGolay=False
    addNoiseColumns=None
    addLandscapes=True

    addfeatures=['BSAN','BSAS','BSAV','CTI','ELEV','EVI','LSTD','LSTN','REF1','REF2','REF3','REF7','RELI','TMAP','TMFI']
    loadFeatures=None
    co2 = ['m2379.76', 'm2377.83', 'm2375.9', 'm2373.97','m2372.04', 'm2370.11', 'm2368.18', 'm2366.26','m2364.33', 'm2362.4', 'm2360.47', 'm2358.54','m2356.61', 'm2354.68', 'm2352.76']
    deleteFeatures=None
    removeCor=None
    
    (Xtrain,Xtest,ymat) = prepareDatasets(nsamples,onlySpectra,deleteSpectra,plotting,standardize,doPCA,findPeaks,makeDerivative,featureFilter,loadFeatures,deleteFeatures,removeVar,removeCor,useSavitzkyGolay,addNoiseColumns,addLandscapes)
    raw_input()
    
    #pcAnalysis(Xtrain,Xtest)
    print Xtrain.columns
    print Xtest.columns
    #ymat = ymat.iloc[:,1:2]
    nt = ymat.shape[1]

    #generate models
    #C:Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
    models=[]
    for i in range(nt):
	#model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=99,mode='percentile')), ('model', LinearRegression())])
	#model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=100,mode='percentile')), ('model', ElasticNet(alpha=.001,l1_ratio=0.15))])
	#model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=100,mode='percentile')), ('model', RidgeCV(alphas=[0.1]))])
	#model = RidgeCV(alphas=[ 0.05,0.1])
	#model = SGDRegressor(alpha=0.1,n_iter=50,shuffle=True,loss='squared_loss',penalty='l1')#too many features
	#model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=99,mode='percentile')), ('model', SGDRegressor(alpha=0.001,n_iter=300,shuffle=True,loss='huber',epsilon=1.0,penalty='elasticnet'))])
	#model = Pipeline([('pca', PCA(n_components=doPCA)), ('model', LinearRegression())])
	#model = Pipeline([('pca', PCA(n_components=doPCA)), ('model', SGDRegressor(alpha=0.00001,n_iter=150,shuffle=True,loss='squared_loss',penalty='l2'))])
	#model = Pipeline([('pca', PCA(n_components=200)), ('model', RandomForestRegressor(n_estimators=250,max_depth=None,min_samples_split=2,min_samples_leaf=5,n_jobs=1,criterion='mse', max_features='auto',oob_score=False))])
	#model = Pipeline([('svd', TruncatedSVD(n_components=25, algorithm=Xtrain'randomized', n_iter=5, tol=0.0)), ('model', RandomForestRegressor(n_estimators=250,max_depth=None,min_samples_split=2,min_samples_leaf=5,n_jobs=1,criterion='mse', max_features='auto',oob_score=False))])
	#model = Pipeline([('pca', PCA(n_components=doPCA)), ('model', SVR(C=1.0, gamma=0.0, verbose = 0))])
	#model = Pipeline([('filter', SelectPercentile(f_regression, percentile=50)), ('model', SVR(kernel='rbf',epsilon=0.1,C=10000.0, gamma=0.0, verbose = 0))])
	#model = Pipeline([('filter',SelectPercentile(f_regression, percentile=99)), ('model', SGDRegressor(alpha=0.001,n_iter=250,shuffle=True,loss='huber',penalty='elasticnet',epsilon=1.0))])
	#model = Pipeline([('filter',SelectPercentile(f_regression, percentile=99)), ('model', BayesianRidge())])
	#model = Pipeline([('filter',SelectPercentile(f_regression, percentile=100)), ('model',GradientBoostingRegressor(loss='huber',n_estimators=150, learning_rate=0.1, max_depth=2,subsample=1.0))])
	#model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=99,mode='percentile')), ('model',KNeighborsRegressor(n_neighbors=5, weights='uniform') )])
	#model = Pipeline([('filter', SelectKBest(f_regression, k=10)),('pca', PCA(n_components=10)), ('model', Ridge())])
	#model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=99,mode='percentile')), ('model', SVR(C=1.0, gamma=0.0, verbose = 0))])
	#model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=100,mode='percentile')), ('model', PLSRegression(n_components=30))])
	#model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=100,mode='percentile')), ('model', Lasso())])
	model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=100,mode='percentile')), ('model', LassoLarsCV())])
	#model = Pipeline([('poly', PolynomialFeatures(degree=3)), ('linear', LinearRegression(fit_intercept=False))])
	#model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=100,mode='percentile')), ('model', LarsCV())])#grottig
	#model = SVR(C=10000.0, gamma=0.0, verbose = 0)
	#model = SVR(C=10000.0, gamma=0.0005, verbose = 0)
	#model = RandomForestRegressor(n_estimators=500,max_depth=None,min_samples_split=2,min_samples_leaf=5,n_jobs=1,criterion='mse', max_features='auto',oob_score=False)
	#model = LinearRegression(normalize=False)
	models.append(model) 
    #individual model SVR
    #models[0] =  Pipeline([('filter', GenericUnivariateSelect(f_regression, param=99,mode='percentile')), ('model', SVR(C=100.0, gamma=0.005, verbose = 0))])#Ca RMSE=0.287
    #models[1] =  Pipeline([('filter', GenericUnivariateSelect(f_regression, param=100,mode='percentile')), ('model', SVR(C=100000.0, gamma=0.0005, verbose = 0))]) #P RMSE=0.886
    #models[2] =  Pipeline([('filter', GenericUnivariateSelect(f_regression, param=95,mode='percentile')), ('model', SVR(C=10000.0, gamma=0.0005, verbose = 0))])#pH RMSE=0.321
    #models[3] =  Pipeline([('filter', GenericUnivariateSelect(f_regression, param=90,mode='percentile')), ('model', SVR(C=10000.0, gamma=0.0005, verbose = 0))])#SOC RMSE=0.278
    #models[4] =  Pipeline([('filter', GenericUnivariateSelect(f_regression, param=99,mode='percentile')), ('model', SVR(C=10000.0, gamma=0.0005, verbose = 0))])#Sand RMSE=0.316
    
    #individual model PLS
    #models[0] = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=100,mode='percentile')), ('model', PLSRegression(n_components=20))])#Ca RMSE=0.384
    #models[1] = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=98,mode='percentile')), ('model', PLSRegression(n_components=20))])#P RMSE=0.886
    #models[2] = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=98,mode='percentile')), ('model', PLSRegression(n_components=50))])#pH RMSE=0.346
    #models[3] = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=100,mode='percentile')), ('model', PLSRegression(n_components=40))])#SOC RMSE =0.348
    #models[4] = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=99,mode='percentile')), ('model', PLSRegression(n_components=40))])#Sand RMSE=0.356
    #make the training
    models = buildmodels(models,Xtrain,ymat,cv_split=24,gridSearch=True,n_jobs=4)
    #modelsFeatureSelection(models,Xtrain,Xtest,ymat)
    #greedyFeatureSelection(lmodel,lX,ymat.iloc[:,1:2],itermax=10,good_features=None, folds= 8)
    
    for i in range(nt):
	print "TARGET: %-10s" %(ymat.columns[i])
	#print models[i].alpha_#optimized alpha from ridge 0.1 0.05
	print models[i]

      
    #makePrediction(models,Xtrain,Xtest,nt,filename='/home/loschen/Desktop/datamining-kaggle/african_soil/submissions/sub2609a.csv')
    #make the predictions 

    print("Model building done on %d samples in %fs" % (Xtrain.shape[0],time() - t0))

