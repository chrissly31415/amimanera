#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""  Modified starter code from BSMan@Kaggle
"""

from time import time
import itertools

import json
import numpy as np
import numpy.random as nprnd
import pandas as pd
import scipy as sp
from scipy import sparse

import matplotlib.pyplot as plt
import pylab as pl

from sklearn.feature_extraction.text import CountVectorizer,HashingVectorizer,TfidfVectorizer
from sklearn import metrics
from sklearn import cross_validation,grid_search
from sklearn.cross_validation import StratifiedKFold,KFold
from sklearn.metrics import roc_auc_score
#from sklearn.utils.extmath import density
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import TruncatedSVD

from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression,RandomizedLogisticRegression,SGDClassifier,Perceptron,SGDRegressor
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,ExtraTreesClassifier,AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from nltk import word_tokenize,sent_tokenize
#from nltk.stem import SnowballStemmer # no english?
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.wordnet import WordNetStemmer
#nltk.stem.porter.PorterStemmer(ignore_stopwords=False)
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import GermanStemmer
#http://nltk.googlecode.com/svn/trunk/doc/howto/collocations.html


from crawldata import *
from featureDesign import *


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

#TODO build class containing classifiers + dataset
#TODO remove infrequent sparse features to new class
#TODO look at amazon challenge winners
#TODO dicretize continous data by cut and qcut?
#TODO use variable names for importance analysis
#TODO #use tags
#TODO feature_extraction.text.CountVectorizer(lowercase=False,analyzer='char',ngram_range=(1,5),)
#TODO https://github.com/cbrew/Insults/blob/master/Insults/insults.py
#TODO Univariate Feature Selection sparse matrix
#TODO winner code insults
#TODO pipelining??
#TODO look at wrong classified ones
#TODO analyze misclassifications
#TODO crawl raw data for linkedin facebook and twitter links
#TODO look after like buttons
#TODO remove duplicate features boilerplate length & food stuff
#TODO analyse html raw data
#TODO boruta with rf features
#TODO implement char ngram with SVD
#TODO ngram of text within <>
#TODO SDGregressor
#TODO http://nltk.org/book/ch05.html
#TODO http://scikit-learn.org/stable/auto_examples/plot_rfe_with_cross_validation.html#example-plot-rfe-with-cross-validation-py
#TODO weka

class NLTKTokenizer(object):
    """
    http://scikit-learn.org/stable/modules/feature_extraction.html
    http://stackoverflow.com/questions/15547409/how-to-get-rid-of-punctuation-using-nltk-tokenizer
    http://nltk.org/api/nltk.tokenize.html
    """
    def __init__(self):
      #self.wnl = LancasterStemmer()
      self.wnl = PorterStemmer()#best so far
      #self.wnl = GermanStemmer()
      #self.wnl = EnglishStemmer(ignore_stopwords=True)
      #self.wnl = WordNetStemmer()
    def __call__(self, doc):
      words=[word_tokenize(t) for t in sent_tokenize(doc)]
      #flatten list
      words=[item for sublist in words for item in sublist]
      #print words
      #print words
      #words=[self.wnl.lemmatize(t) for t in words]
      words=[self.wnl.stem(t) for t in words]
      #print words
      return words
    
def dfinfo(X_all):
    print "##Basic data##\n",X_all
    print "##Details##\n",X_all.ix[:,0:2].describe()
    print "##Details##\n",X_all.ix[:,2:3].describe()
    print "##Details##\n",X_all.ix[:,3:7].describe()

def prepareDatasets(vecType='hV',useSVD=0,useJson=True,useHTMLtag=True,useAddFeatures=True,usePosTag=True,useAlcat=False):
    """
    Load Data into pandas and preprocess features
    """
    #pd.set_printoptions(max_rows=200, max_columns=5)
    
    print "loading dataset..."
    X = pd.read_csv('../stumbled_upon/data/train.tsv', sep="\t", na_values=['?'], index_col=1)
    X_test = pd.read_csv('../stumbled_upon/data/test.tsv', sep="\t", na_values=['?'], index_col=1)
    y = X['label']
    X = X.drop(['label'], axis=1)
    # Combine test and train while we do our preprocessing
    X_all = pd.concat([X_test, X])
    print "Original shape:",X_all.shape
    
    #vectorize data
    #vectorizer = HashingVectorizer(ngram_range=(1,2), non_negative=True)
    if vecType=='hV':
	vectorizer = HashingVectorizer(stop_words='english',ngram_range=(1,2),analyzer="word", non_negative=True, norm='l2', n_features=2**19)
    elif vecType=='tfidfV':
	print "Using tfidfV..."
	#vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1,2),stop_words=None,max_features=None,binary=False,min_df=3,strip_accents='unicode',tokenizer=NLTKTokenizer())
	#vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1,2),stop_words=None,max_features=None,binary=True,min_df=5,strip_accents='unicode')
	#vectorizer = TfidfVectorizer(ngram_range=(1,1),stop_words=None,max_features=2**14,sublinear_tf=True,min_df=4,tokenizer=NLTKTokenizer())
	vectorizer = TfidfVectorizer(ngram_range=(1,1),stop_words=None,max_features=2**14,sublinear_tf=True,min_df=4)#fast
	#vectorizer = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode',analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), sublinear_tf=True)#opt
    else:
	#vectorizer = CountVectorizer(ngram_range=(1,2),analyzer='word',max_features=2**18)
	#vectorizer = CountVectorizer(lowercase=False,analyzer="char_wb",ngram_range=(4,4),max_features=2**14,stop_words='english')#AUC = 0.781
	#vectorizer = CountVectorizer(lowercase=False,analyzer="char",ngram_range=(4,4),max_features=2**14,stop_words='english')#AUC= 0.786
	#vectorizer = CountVectorizer(lowercase=False,analyzer="char",ngram_range=(4,4),max_features=2**18,stop_words='english')#AUC= 0.798
	#vectorizer = CountVectorizer(lowercase=False,analyzer="char",ngram_range=(5,5),max_features=2**14,stop_words='english')#slow and low score 0.786
	#vectorizer = CountVectorizer(lowercase=False,analyzer="char",ngram_range=(5,5),max_features=2**18,stop_words='english')#AUC=  0.815 1400s
	#vectorizer = CountVectorizer(lowercase=True,analyzer="char",ngram_range=(5,5),max_features=2**18,stop_words='english')#AUC= 0.813  1400s
	#vectorizer = CountVectorizer(lowercase=False,analyzer="char",ngram_range=(4,5),max_features=2**18,stop_words='english')#AUC=   aufgehängt memory?
	vectorizer = CountVectorizer(lowercase=False,analyzer="char",ngram_range=(5,5),max_features=2**16,stop_words='english')#AUC= 0.806 682s
    
    
    #transform data using json
    if useJson:
	print "Xtracting data using json..."
	#take only boilerplate data
	X_all['boilerplate'] = X_all['boilerplate'].apply(json.loads)
	
	#print X_all['boilerplate']
	#print X_all['boilerplate'][2]
	# Initialize the data as a unicode string
	X_all['body'] = u'empty'
	extractBody = lambda x: x['body'] if x.has_key('body') and x['body'] is not None else u'empty'
	X_all['body'] = X_all['boilerplate'].map(extractBody)
	
	X_all['title'] = u'empty'
	extractBody = lambda x: x['title'] if x.has_key('title') and x['title'] is not None else u'empty'
	X_all['title'] = X_all['boilerplate'].map(extractBody)
	
	X_all['url2'] = u'empty'
	extractBody = lambda x: x['url'] if x.has_key('url') and x['url'] is not None else u'empty'
	X_all['url2'] = X_all['boilerplate'].map(extractBody)
	
	X_all['body'] = X_all.body+u' '+X_all.url2
	X_all['body'] = X_all.body+u' '+X_all.title
	#print X_all['body'].head(30).to_string()
	#print X_all['body'].tail(30).to_string()		
	body_counts = vectorizer.fit_transform(X_all['body'])
	
	#body_counts = body_counts.tocsr()
	print "body,title+url, dim:",body_counts.shape
	print "density:",density(body_counts)		

    #simple transform
    else:
        print "Creating dataset by simple method..."
        body_counts=list(X_all['boilerplate'])
	body_counts = vectorizer.fit_transform(body_counts)
	print "Final dim:",body_counts.shape	
    #feature_names = None
    #if hasattr(vectorizer, 'get_feature_names'):
    	#feature_names = np.asarray(vectorizer.get_feature_names())
    #X & X_test are converted to sparse matrix
    
    #bringt seltsamerweise nichts
    #X_alcat=pd.DataFrame(X_all['alchemy_category'])
    #X_alcat=X_alcat.fillna('NA')
    #X_alcat = one_hot_encoder(X_alcat, ['alchemy_category'], replace=True)
    #print X_alcat
    #X_alcat=sparse.csr_matrix(pd.np.array(X_alcat))
    #body_counts = sparse.hstack((body_counts,X_alcat),format="csr")

    y = pd.np.array(y)
    if useSVD>1:
	if useAddFeatures==True:
	    X_raw=crawlRawData(X_all)
	    X_all=pd.concat([X_all,X_raw], axis=1)
	    X_all = featureEngineering(X_all)
	
	print "Actual shape:",X_all.shape
	#SVD of text data (LSA)
	print "SVD of sparse data with n=",useSVD
	tsvd=TruncatedSVD(n_components=useSVD, algorithm='randomized', n_iterations=5, random_state=42, tol=0.0)
	X_svd=tsvd.fit_transform(body_counts)
	X_svd=pd.DataFrame(np.asarray(X_svd),index=X_all.index)
	#char_ngrams
	if useHTMLtag:
	    #char ngrams
	    X_raw=crawlHTML(X_all)
	    #char_vectorizer=CountVectorizer(lowercase=False,analyzer="char",ngram_range=(5,5),max_features=2**18,stop_words=None)
	    char_vectorizer=CountVectorizer(lowercase=True,analyzer="char",ngram_range=(5,5),max_features=2**14,stop_words=None)
	    char_ngrams = char_vectorizer.fit_transform(X_raw['htmltag'])
	    #char_ngrams = char_vectorizer.fit_transform(X_all['body'])
	    print "char ngrams, dim:",char_ngrams.shape
	    print "density:",density(char_ngrams)
	    useSVD2=10
	    print "SVD of char ngrams data with n=",useSVD2
	    tsvd=TruncatedSVD(n_components=useSVD2, algorithm='randomized', n_iterations=5, random_state=42, tol=0.0)
	    X_char=tsvd.fit_transform(char_ngrams)
	    X_char=pd.DataFrame(np.asarray(X_char),index=X_all.index,columns=["char"+str(x) for x in xrange(useSVD2)])
	    print "X_char",X_char
	    X_svd=pd.concat([X_svd,X_char], axis=1)
	
	if usePosTag:
	    #posTagging(X_all)
	    X_pos=pd.read_csv('../stumbled_upon/data/postagged.csv', sep=",", na_values=['?'], index_col=0)
	    X_svd=pd.concat([X_svd,X_pos], axis=1)
	
	#print "##X_svd##\n",X_svd
	X_all= X_all.drop(['body','url2','title','boilerplate','url'], axis=1)
	X_all= X_all.drop(['hasDomainLink','framebased','news_front_page','embed_ratio'], axis=1)
	if useJson: 
	    X_rest= X_all.drop(['alchemy_category'], axis=1)
	else:
	    X_rest= X_all.drop(['url','alchemy_category','boilerplate'], axis=1)
	X_rest = X_rest.astype(float)
	X_rest=X_rest.fillna(X_rest.median())	
	X_rest.corr().to_csv("corr.csv")
	X_svd=pd.concat([X_rest,X_svd], axis=1)
	#print "##X_svd,int##\n",X_svd
	#add alchemy category again, but now one hot encode, bringt nichts...
	if useAlcat:
	    X_alcat=pd.DataFrame(X_all['alchemy_category'])
	    X_alcat=X_alcat.fillna('unknown')
	    X_alcat = one_hot_encoder(X_alcat, ['alchemy_category'], replace=True)
	    X_alcat = pd.DataFrame(X_alcat)
	    X_svd=pd.concat([X_svd,X_alcat], axis=1)
	print "##X_svd,final##\n",X_svd
	#X_rest=X_svd
	print "Dim: X_svd:",X_svd.shape   
	X_svd_train = X_svd[len(X_test.index):]
	X_svd_test = X_svd[:len(X_test.index)]
	return(X_svd_train,y,X_svd_test,X_test.index)
    else:
	Xs = body_counts[len(X_test.index):]
	Xs_test = body_counts[:len(X_test.index)]
	#conversion to array necessary to work with integer indexing, .iloc does not work with this version
	return (Xs,y,Xs_test,X_test.index)
	
    

def one_hot_encoder(data, cols, replace=False):
    """ Takes a dataframe and a list of columns that need to be encoded.
    Returns a 3-tuple comprising the data, the vectorized data,
    and the fitted vectorizor.
    credits to https://gist.github.com/kljensen/5452382
    """
    #temporarily out of order...?
    vec = DictVectorizer()
#    def mkdict(row):
#	  d={}
#	  for col in cols:
#	      print "col",col
#	      print "row",row[col]
#	      d.update({col: row[col]})
#	  print d
#	  return d
    #mkdict = lambda row: dict((col, row[col]) for col in cols)
    tmp=data[cols].apply(mkdict, axis=1)
    print "tmp",tmp
    tmp=vec.fit_transform(tmp).toarray()
    print type(tmp)
    vecData = pd.DataFrame(tmp)
    vecData.columns = vec.get_feature_names()
    vecData.index = data.index
    if replace is True:
	data = data.drop(cols, axis=1)
	data = data.join(vecData)
    return data
    
    
def makePredictions(lmodel,lXs_test,lidx,filename):
    """
    Uses priorily fit model to make predictions
    """
    print "Saving predictions to: ",filename
    print "Final test dataframe:",lXs_test.shape
    preds = lmodel.predict_proba(lXs_test)[:,1]
    pred_df = pd.DataFrame(preds, index=lidx, columns=['label'])
    pred_df.to_csv(filename)

def analyzeModel(lmodel,feature_names):
    """
    Analysis of data if feature_names are available
    """
    if hasattr(lmodel, 'coef_'): 
	print("Analysis of data...")
	print("Dimensionality: %d" % lmodel.coef_.shape[1])
	print("Density: %4.3f" % density(lmodel.coef_))
	if feature_names is not None:
	  top10 = np.argsort(lmodel.coef_)[0,-10:][::-1]
	  #print model.coef_[top10b]
	  for i in xrange(top10.shape[0]):
	      print("Top %2d: coef: %0.3f %20s" % (i+1,lmodel.coef_[0,top10[i]],feature_names[top10[i]]))
	      
def modelEvaluation(lmodel,lXs,ly):
    """
    MODEL EVALUATION
    """
    print "Model evaluation..."
    folds=8
    #parameters=np.logspace(-14, -7, num=8, base=2.0)#SDG
    #parameters=np.logspace(-7, 0, num=8, base=2.0)#LG
    #parameters=[250,500,1000,2000]#rf
    parameters=[5000,10000,16000,50000,100000,150000,'all']#chi2
    #parameters=[2,3,4,5]#gbm
    #parameters=np.logspace(-7, -0, num=8, base=2.0)
    print "Parameter space:",parameters
    #feature selection within xvalidation
    oobpreds=np.zeros((lXs.shape[0],len(parameters)))
    for j,p in enumerate(parameters):
	#if isinstance(lmodel,SGDClassifier):
	#    lmodel.set_params(alpha=p)
	#if (isinstance(lmodel,LogisticRegression) or isinstance(lmodel,SVC)) and p<1000:
	#    lmodel.set_params(C=p)
	if isinstance(lmodel,RandomForestClassifier) :
	    lmodel.set_params(max_features=p)
	if isinstance(lmodel,GradientBoostingClassifier):
	    lmodel.set_params(max_depth=p)
        #print lmodel.get_params()
        cv = StratifiedKFold(lXs.shape[0], n_folds=folds,indices=True)
	scores=np.zeros(folds)	
	for i, (train, test) in enumerate(cv):
	    print("Extracting %s best features by a chi-squared test" % p)
	    ch2 = SelectKBest(chi2, k=p)
	    Xtrain = ch2.fit_transform(lXs[train], ly[train])
	    Xtest = ch2.transform(lXs[test]) 
	    #Xtrain = lXs[train]
	    #Xtest = lXs[test]
	    lmodel.fit(Xtrain, ly[train])
	    oobpreds[test,j] = lmodel.predict_proba(Xtest)[:,1]
	    scores[i]=roc_auc_score(ly[test],oobpreds[test,j])
	    #print "AUC: %0.2f " % (scores[i])
	    #save oobpredictions
	print "Iteration:",j," parameter:",p,
	print " <AUC>: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std()),
	print " AUC oob: %0.3f" %(roc_auc_score(ly,oobpreds[:,j]))
    scores=[roc_auc_score(ly,oobpreds[:,j]) for j in xrange(len(parameters))]
    plt.plot(parameters,scores,'ro')
    

def sigmoid(x):
  y = 1.0/(1.0 + exp(-x))
  return(y)
    

def ensembleBuilding(lXs,ly):
    """
    train ensemble
    """
    print "Ensemble training..."
    folds=8
    parameters=np.logspace(-14, -7, num=10, base=2.0)
    #parameters=nprnd.choice(parameters, 8)
    classifiers = {}
    for p in parameters:
        l1ratio=nprnd.ranf()
	dic ={'SDG_alpha'+str(p)+'_L1'+str(l1ratio): SGDClassifier(alpha=p, n_iter=50,penalty='elasticnet',l1_ratio=l1ratio,shuffle=True,random_state=np.random.randint(0,100),loss='log')}
	classifiers.update(dic)
    dic ={'NB': BernoulliNB(alpha=1.0)}
    classifiers.update(dic)
    dic ={'LG1': LogisticRegression(penalty='l2', tol=0.0001, C=1.0,random_state=42)}
    classifiers.update(dic)
    dic ={'SDG1': SGDClassifier(alpha=0.0001, n_iter=50,shuffle=True,random_state=42,loss='log',penalty='l2')}
    classifiers.update(dic)
    #dic ={'SDG2': SGDClassifier(alpha=0.0005, n_iter=50,shuffle=True,random_state=42,loss='log',penalty='l1')}
    #classifiers.update(dic)
    #dic ={'LG2': LogisticRegression(penalty='l1', tol=0.0001, C=1.0,random_state=42)}
    #classifiers.update(dic)
    #dic ={'KNN': KNeighborsClassifier(n_neighbors=5)}
    #classifiers.update(dic)  
    #dic ={'SDG3': SGDClassifier(alpha=.0001220703125, n_iter=50,penalty='elasticnet',l1_ratio=0.2,shuffle=True,random_state=42,loss='log')}
    #classifiers.update(dic)
    oobpreds=np.zeros((lXs.shape[0],len(classifiers)))
    for j,(key, lmodel) in enumerate(classifiers.iteritems()):
        #print lmodel.get_params()
        cv = StratifiedKFold(ly, n_folds=folds, indices=True)
	scores=np.zeros(folds)	
	for i, (train, test) in enumerate(cv):
	    Xtrain = lXs[train]
	    Xtest = lXs[test]
	    lmodel.fit(Xtrain, ly[train])
	    oobpreds[test,j] = lmodel.predict_proba(Xtest)[:,1]
	    scores[i]=roc_auc_score(ly[test],oobpreds[test,j])
	    #print "AUC: %0.2f " % (scores[i])
	    #save oobpredictions
	print "Iteration:",j," model:",key,
	print " <AUC>: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std()),
	oobscore=roc_auc_score(ly,oobpreds[:,j])
	print " AUC oob: %0.3f" %(oobscore)
	
    scores=[roc_auc_score(ly,oobpreds[:,j]) for j in xrange(len(classifiers))]
    #simple averaging of blending
    oob_avg=np.mean(oobpreds,axis=1)
    print " AUC oob, simple mean: %0.3f" %(roc_auc_score(ly,oob_avg))
    
    #do another crossvalidation for weights
    blender=LogisticRegression(penalty='l2', tol=0.0001, C=1.0)
    cv = StratifiedKFold(ly, n_folds=folds, indices=True)
    blend_scores=np.zeros(folds)
    blend_oob=np.zeros((lXs.shape[0]))
    for i, (train, test) in enumerate(cv):
	Xtrain = oobpreds[train]
	Xtest = oobpreds[test]
	blender.fit(Xtrain, ly[train])
	blend_oob[test] = blender.predict_proba(Xtest)[:,1]
	blend_scores[i]=roc_auc_score(ly[test],blend_oob[test])
    print " <AUC>: %0.3f (+/- %0.3f)" % (blend_scores.mean(), blend_scores.std()),
    print " AUC oob after blending: %0.3f" %(roc_auc_score(ly,blend_oob))
    print "Coefficients:",blender.coef_
    
    plt.plot(range(len(classifiers)),scores,'ro')
    return(classifiers,blender)
    

def ensemblePredictions(classifiers,blender,lXs_test,lidx,filename):
    """   
    Makes prediction
    """ 
    print "Make final ensemble prediction..."
    #make prediction for each classifiers
    preds=np.zeros((lXs_test.shape[0],len(classifiers)))
    for j,(key, lmodel) in enumerate(classifiers.iteritems()):
	preds[:,j]=lmodel.predict_proba(lXs_test)[:,1]
    #blend results
    finalpred=blender.predict_proba(preds)[:,1]   
    print "Saving predictions to: ",filename
    print "Final test dataframe:",lXs_test.shape
    pred_df = pd.DataFrame(finalpred, index=lidx, columns=['label'])
    pred_df.to_csv(filename)
    
def pyGridSearch(lmodel,lXs,ly):  
    """   
    Grid search with sklearn internal tool
    """ 
    print "Grid search..."
    #parameters = {'C':[1000,10000,100], 'gamma':[0.001,0.0001]}
    #parameters = {'max_depth':[5], 'learning_rate':[0.001],'n_estimators':[3000,5000,10000]}#gbm
    #parameters = {'max_depth':[2], 'learning_rate':[0.01,0.001],'n_estimators':[3000]}#gbm
    #parameters = {'n_estimators':[500], 'max_features':[5,10,15]}#rf
    parameters = {'n_estimators':[500], 'max_features':[10,15,20],'min_samples_leaf':[8,10,12]}#rf
    #parameters = {'C':[0.1,1,10]}#SVC
    clf_opt = grid_search.GridSearchCV(lmodel, parameters,cv=8,scoring='roc_auc',n_jobs=4,verbose=1)
    clf_opt.fit(lXs,ly)
    print type(clf_opt.grid_scores_)
    
    for params, mean_score, scores in clf_opt.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score.mean(), scores.std(), params))
    return(clf_opt.best_estimator_)
    
def buildModel(lmodel,lXs,ly,feature_names=None):
    """   
    Final model building part
    """ 
    print "Xvalidation..."
    scores = cross_validation.cross_val_score(lmodel, lXs, ly, cv=8, scoring='roc_auc',n_jobs=4)
    print "AUC: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std())
    print "Building model with all instances..."
    if isinstance(lmodel,RandomForestClassifier) or isinstance(lmodel,SGDClassifier):
	    lmodel.set_params(n_jobs=4)
    lmodel.fit(lXs,ly)
    #analyzeModel(lmodel,feature_names)
    return(lmodel)

    
def density(m):
    entries=m.shape[0]*m.shape[1]
    return m.nnz/float(entries)
    
    
def group_sparse_old(Xold,Xold_test, degree=2,append=True):
    """ 
    multiply columns of sparse data
    """
    print "Grouping sparse data..."
    #only for important data
    (lXs,lXs_test) = linearFeatureSelection(model,Xold,Xold_test,200)
    #also transform old data
    #(Xold,Xold_test) = linearFeatureSelection(model,Xold,Xold_test,5000)
    
    Xtmp=sparse.vstack((lXs_test,lXs),format="csr")
    #turn into pandas dataframe for grouping
    new_data=None
    m,n = Xtmp.shape
    for indices in itertools.combinations(range(n), degree):
        #print "idx:",indices
	col1,col2 =indices
	out1 = Xtmp.tocsc()[:,col1]
	out1 = out1.transpose(copy=False)
	out2 = Xtmp.tocsc()[:,col2]
	tmp = np.ravel(np.asarray(out2.todense()))
	diag2 = sparse.spdiags(tmp,[0],out2.shape[0],out2.shape[0],format="csc")
	#out1+diag2-max(out1,diag2)
	prod = out1*diag2
	prod = prod.transpose()
	dens=density(prod)
	#print " Non-zeros: %4.3f " %(dens)
	if new_data is None:  
	    new_data=sparse.csc_matrix(prod)
	elif dens>0.0:
	    new_data=sparse.hstack((new_data,prod),format="csr")
	
    print "Shape of interactions matrix:",new_data.shape,
    print " Non-zeros: %4.3f " %(density(new_data))

    #makting test data
    Xreduced_test = new_data[:Xold_test.shape[0]]
    if append: 
	Xreduced_test=sparse.hstack((Xold_test,Xreduced_test),format="csr")
    print "New test data:",Xreduced_test.shape
    
    #making train data
    Xreduced = new_data[Xold_test.shape[0]:]
    if append:
	Xreduced=sparse.hstack((Xold,Xreduced),format="csr")
    print "New test data:",Xreduced.shape
    
    return(Xreduced,Xreduced_test)

def group_sparse(Xold,Xold_test, degree=2,append=True):
    """ 
    multiply columns of sparse data
    """
    print "Columnwise min of data..."
    #only for important data
    (lXs,lXs_test) = linearFeatureSelection(model,Xold,Xold_test,10)
    #also transform old data
    #(Xold,Xold_test) = linearFeatureSelection(model,Xold,Xold_test,5000)
    Xtmp=sparse.vstack((lXs_test,lXs),format="csc")
    Xtmp=pd.DataFrame(np.asarray(Xtmp.todense()))
    new_data = None
    m,n = Xtmp.shape
    for indices in itertools.combinations(range(n), degree):
	indices=Xtmp.columns[list(indices)]
	print indices
	if not isinstance(new_data,pd.DataFrame):
	  new_data=pd.DataFrame(Xtmp[indices].apply(np.min, axis=1))
	else:
	  new_data = pd.concat([new_data, pd.DataFrame(Xtmp[indices].apply(np.min, axis=1))],axis=1)
	print new_data.shape
    
    
    #making test data
    Xreduced_test = new_data[:Xold_test.shape[0]]
    if append: 
	Xreduced_test=sparse.hstack((Xold_test,Xreduced_test),format="csr")
    print "New test data:",Xreduced_test.shape
    
    #making train data
    Xreduced = new_data[Xold_test.shape[0]:]
    if append:
	Xreduced=sparse.hstack((Xold,Xreduced),format="csr")
    print "New test data:",Xreduced.shape
    
    return(Xreduced,Xreduced_test)
    
    
def rfFeatureImportance(forest,Xold,Xold_test,n):
    """ 
    Selects n best features from a model which has the attribute feature_importances_
    """
    print "Feature importance..."
    if not hasattr(forest,'feature_importances_'): return
    importances = forest.feature_importances_
    #std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)#perhas we need it later
    indices = np.argsort(importances)[::-1]
    # Print the feature ranking
    print("Feature ranking:")

    for f in range(len(indices)):
	print("%d. feature %64s %d (%f)" % (f + 1, Xold.columns[indices[f]], indices[f], importances[indices[f]]))
	
    # Plot the feature importances of the forest  
    plt.bar(left=np.arange(len(indices)),height=importances[indices] , width=0.35, color='r')
    plt.ylabel('Importance')
    plt.title("Feature importances")
    #stack train and test data
    Xreduced = pd.concat([Xold_test, Xold])
    #sorting features
    n=len(indices)-n
    print "Selection of ",n," top features..."
    Xtmp=Xreduced.iloc[:,indices[0:n]]
    print Xtmp.columns
    #split train and test data
    #pd slicing sometimes confusing...last element in slicing is inclusive!!! use iloc for integer indexing (i.e. in case index are float or not ordered)
    pdrowidx=Xold_test.shape[0]-1
    Xreduced_test = Xtmp[:len(Xold_test.index)]
    Xreduced = Xtmp[len(Xold_test.index):]
    print "Xreduced_test:",Xreduced_test
    print "Xreduced_test:",Xreduced_test.shape
    print "Xreduced_train:",Xreduced.shape
    return(Xreduced,Xreduced_test)

def linearFeatureSelection(lmodel,Xold,Xold_test,n):
    """
    Analysis of data if coef_ are available
    """
    print "Selecting features based on important coefficients..."
    if hasattr(lmodel, 'coef_') and isinstance(Xold,sparse.csr.csr_matrix): 
	print("Dimensionality before: %d" % lmodel.coef_.shape[1])
	indices = np.argsort(lmodel.coef_)[0,-n:][::-1]
	#print model.coef_[top10b]
	#for i in xrange(indices.shape[0]):
	#    print("Top %2d: coef: %0.3f col: %2d" % (i+1,lmodel.coef_[0,indices[i]], indices[i]))
	plt.bar(left=np.arange(len(indices)),height=lmodel.coef_[0,indices] , width=0.35, color='r')
	plt.ylabel('Importance')
	#stack train and test data
	#Xreduced=np.vstack((Xold_test,Xold))
	Xreduced=sparse.vstack((Xold_test,Xold),format="csr")
	#sorting features
	#print indices[0:n]
	Xtmp=Xreduced[:,indices[0:n]] 
	print("Dimensionality after: %d" % Xtmp.shape[1])
	#split train and test data
	Xreduced_test = Xtmp[:Xold_test.shape[0]]
	Xreduced = Xtmp[Xold_test.shape[0]:]
	return(Xreduced,Xreduced_test)

	
def iterativeFeatureSelection(lmodel,Xold,Xold_test,ly,iterations,nrfeats):
	"""
	Iterative Feature Selection
	"""
	for i in xrange(iterations):
	    print ">>>Iteration: ",i,"<<<"
	    lmodel = buildModel(model,Xold,ly)
	    (Xold,Xold_test)=rfFeatureImportance(model,Xold,Xold_test,nrfeats)
	return(Xold,Xold_test)



def removeInstances(lXs,ly,preds,t):
	#now remove examples from train
	res=np.abs(ly-preds)
	d={'abs_err' : pd.Series(res)}
	res=pd.DataFrame(d)
	res=pd.DataFrame(d)
	res.index=lXs.index
	lXs_reduced=pd.concat([lXs,res], axis=1)
	boolindex=lXs_reduced['abs_err']<t
	lXs_reduced=lXs_reduced[boolindex]
	#ninst[i]=len(Xtrain.index)-len(lXs_reduced.index)
	lXs_reduced = lXs_reduced.drop(['abs_err'], axis=1)
	#print "New dim:",lXs_reduced.shape
	ly_reduced=ly[np.asarray(boolindex)]
	return (lXs_reduced,ly_reduced)

	
def getOOBCVPredictions(lmodel,lXs,lXs_test,ly,folds=8,repeats=1):
	"""
	Get cv oob predictions for classifiers
	"""
	print "Computing oob predictions..."
	if isinstance(lmodel,RandomForestClassifier) or isinstance(lmodel,SGDClassifier):
		lmodel.set_params(n_jobs=4)
	oobpreds=np.zeros((lXs.shape[0],repeats))
	for j in xrange(repeats):
	    #print lmodel.get_params()
	    cv = KFold(lXs.shape[0], n_folds=folds, indices=True,random_state=j,shuffle=True)
	    scores=np.zeros(folds)	
	    for i, (train, test) in enumerate(cv):
		Xtrain = lXs.iloc[train]
		Xtest = lXs.iloc[test]
		#print Xtest['avglinksize'].head(3)
		lmodel.fit(Xtrain, ly[train])
		oobpreds[test,j] = lmodel.predict_proba(Xtest)[:,1]
		scores[i]=roc_auc_score(ly[test],oobpreds[test,j])
		#print "AUC: %0.2f " % (scores[i])
		#save oobpredictions
	    print "Iteration:",j,
	    print " <AUC>: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std()),
	    oobscore=roc_auc_score(ly,oobpreds[:,j])
	    print " AUC,oob: %0.3f" %(oobscore)
	scores=[roc_auc_score(ly,oobpreds[:,j]) for j in xrange(repeats)]
	#simple averaging of blending
	oob_avg=np.mean(oobpreds,axis=1)
	print "Summary: <AUC,oob>: %0.3f (%d repeats)" %(roc_auc_score(ly,oob_avg),repeats,)
	return(oob_avg)
	
	
def filterClassNoise(lmodel,lXs,lXs_test,ly):
	"""
	Removes training samples which could be class noise
	Done in outer XVal loop
	precision: Wieviel falsche habe ich erwischt
	recall: wieviele richtige sind durch die Lappen gegangen
	"""
	threshhold=[0.68,0.70,0.72,0.75,0.8]
	folds=8
	print "Filter strongly misclassified classes..."
	#rdidx=random.sample(xrange(1000), 20)
	#print rdidx
	#lXs = lXs.iloc[rdidx]
	#ly = ly[rdidx]
	preds=getOOBCVPredictions(model,Xs,Xs_test,y,8,5)
	#print "stdev:",std
	#should be oob or cvalidated!!!!
	#preds = lmodel.predict_proba(lXs)[:,1]
	scores=np.zeros((folds,len(threshhold)))
	for j,t in enumerate(threshhold):
	    #XValidation
	    cv = KFold(lXs.shape[0], n_folds=folds, indices=True,random_state=j,shuffle=True)	    	    
	    ninst=np.zeros(folds)
	    oobpreds=np.zeros((lXs.shape[0],1))
	    for i, (train, test) in enumerate(cv):
		Xtrain = lXs.iloc[train]
		ytrain=  ly[train]		
		#now remove examples from train
		lXs_reduced,ly_reduced = removeInstances(Xtrain,ytrain,preds[train],t)
		ninst[i]=len(Xtrain.index)-len(lXs_reduced.index)
		lmodel.fit(lXs_reduced, ly_reduced)
		
		#testing data, not manipulated
		Xtest = lXs.iloc[test]
		oobpreds[test,0] = lmodel.predict_proba(Xtest)[:,1]
		scores[i,j]=roc_auc_score(ly[test],oobpreds[test,0])

	    print "Threshhold: %0.3f  <AUC>: %0.3f (+/- %0.3f) removed instances: %4.2f" % (t, scores[:,j].mean(), scores[:,j].std(), ninst.mean() ),
	    print " AUC oob: %0.3f" %(roc_auc_score(ly,oobpreds[:,0]))
	scores=np.mean(scores,axis=0)
	print scores
	plt.plot(threshhold,scores,'ro')
	    #scores = cross_validation.cross_val_score(lmodel, lXs_reduced, ly_reduced, cv=8, scoring='roc_auc',n_jobs=4)
	    #print "Threshhold: %0.3f AUC: %0.3f (+/- %0.3f)" % (t,scores.mean(), scores.std())
	
	return(lXs,lXs_test,ly)

    
if __name__=="__main__":
    """   
    MAIN PART
    """ 
    # Set a seed for consistant results
    t0 = time()
    np.random.seed(1234)
    print "numpy:",np.__version__
    print "pandas:",pd.__version__
    #print pd.util.terminal.get_terminal_size()
    pd.set_printoptions(max_rows=300, max_columns=8)
    print "scipy:",sp.__version__
    #variables
    (Xs,y,Xs_test,data_indices) = prepareDatasets('tfidfV',useSVD=50,useJson=True,useHTMLtag=False,useAddFeatures=False,usePosTag=True,useAlcat=False)#opt SVD=50
    Xs.to_csv("../stumbled_upon/data/Xlarge.csv")
    Xs_test.to_csv("../stumbled_upon/data/XXlarge_test.csv")
    
    #(Xs,y,Xs_test,data_indices) = prepareSimpleData()
    print "Dim X (training):",Xs.shape
    print "Type X:",type(Xs)
    print "Dim X (test):",Xs_test.shape
    # Fit a model and predict
    #model = BernoulliNB(alpha=1.0)
    #model = SGDClassifier(alpha=.0001, n_iter=50,penalty='elasticnet',l1_ratio=0.2,shuffle=True,random_state=42,loss='log')
    #model = SGDClassifier(alpha=0.0005, n_iter=50,shuffle=True,random_state=42,loss='log',penalty='l2',n_jobs=4)#opt  
    #model = SGDClassifier(alpha=0.0001, n_iter=50,shuffle=True,random_state=42,loss='log',penalty='l2',n_jobs=4)#opt simple processing
    #model = LogisticRegression(penalty='l2', tol=0.0001, C=1.0)#opt
    #model = LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
    #                         C=1, fit_intercept=True, intercept_scaling=1.0, 
    #                         class_weight=None, random_state=None)#kaggle params
    #model = RandomizedLogisticRegression(C=1, scaling=0.5, sample_fraction=0.75, n_resampling=200, selection_threshold=0.25, tol=0.001, fit_intercept=True, verbose=False, normalize=True, random_state=42)
    #model = KNeighborsClassifier(n_neighbors=10)
    #model=SVC(C=0.3,kernel='linear',probability=True)
    #model=LinearSVC(penalty='l2', loss='l2', dual=True, tol=0.0001, C=1.0)#no proba
    #model = SVC(C=1, cache_size=200, class_weight='auto', gamma=0.0, kernel='rbf', probability=True, shrinking=True,tol=0.001, verbose=False)
    model = RandomForestClassifier(n_estimators=100,max_depth=None,min_samples_leaf=10,n_jobs=1,criterion='entropy', max_features=15,oob_score=False,random_state=42)
    #model = ExtraTreesClassifier(n_estimators=500,max_depth=None,min_samples_leaf=10,n_jobs=1,criterion='entropy', max_features=15,oob_score=False,random_state=42)
    #model = AdaBoostClassifier(n_estimators=500,learning_rate=0.1,random_state=42)
    #model = GradientBoostingClassifier(loss='deviance', learning_rate=0.01, n_estimators=5000, subsample=1.0, min_samples_split=2, min_samples_leaf=10, max_depth=3, init=None, random_state=42,verbose=False)
    #model = SVC(C=1, cache_size=200, class_weight='auto', gamma=0.0, kernel='rbf', probability=True, shrinking=True,tol=0.001, verbose=False)  
    #modelEvaluation(model,Xs,y)
    #model=pyGridSearch(model,Xs,y)
    #(gclassifiers,gblender)=ensembleBuilding(Xs,y)
    #ensemblePredictions(gclassifiers,gblender,Xs_test,data_indices,'sub1309a.csv')
    #fit final model
    model = buildModel(model,Xs,y) 
    
    (Xs,Xs_test,y)=filterClassNoise(model,Xs,Xs_test,y)
    #model = buildModel(model,Xs,y) 
    #(Xs,Xs_test)=iterativeFeatureSelection(model,Xs,Xs_test,y,1,10)
    #model = buildModel(model,Xs,y)  
    #(Xs,Xs_test) = group_sparse(Xs,Xs_test)
    #print "Dim X (after grouping):",Xs.shape
    #makePredictions(model,Xs_test,data_indices,'../stumbled_upon/submissions/sub1509b.csv')	            
    print("Model building done in %fs" % (time() - t0))
    plt.show()
