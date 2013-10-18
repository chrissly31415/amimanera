#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""  Modified starter code from BSMan@Kaggle
"""

from time import time
import itertools
from random import choice

import json
import numpy as np
import numpy.random as nprnd
import pandas as pd
import scipy as sp
from scipy import sparse

import matplotlib.pyplot as plt
import pylab as pl

from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer,HashingVectorizer,TfidfVectorizer
from sklearn import metrics
from sklearn import cross_validation,grid_search
from sklearn.cross_validation import StratifiedKFold,KFold
from sklearn.metrics import roc_auc_score
#from sklearn.utils.extmath import density
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

from sklearn.feature_selection import SelectKBest,SelectPercentile, chi2, f_classif,f_regression
from sklearn.naive_bayes import BernoulliNB,MultinomialNB
from sklearn.linear_model import LogisticRegression,RandomizedLogisticRegression,SGDClassifier,Perceptron,SGDRegressor
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,ExtraTreesClassifier,AdaBoostClassifier,ExtraTreesRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


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
from FullModel import *

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

#TODO build class containing classifiers + dataset
#TODO remove infrequent sparse features to new class
#TODO https://github.com/cbrew/Insults/blob/master/Insults/insults.py
#TODO look at wrong classified ones
#TODO analyze misclassifications
#TODO remove duplicate features boilerplate length & food stuff
#TODO http://nltk.org/book/ch05.html
#TODO http://scikit-learn.org/stable/auto_examples/plot_rfe_with_cross_validation.html#example-plot-rfe-with-cross-validation-py
#TODO using LDA with gensim: http://blog.kaggle.com/2012/07/17/getting-started-with-the-wordpress-competition/
#TODO recursive feature engineering
#TODO top:SDG_alpha0.000136463620667_L10.992081466188
#TODO bumping
#TODO calibration of AUC by reducing uncertain webpages to p=0.5 ->NO
#TODO transformation of variable log of length variables, standardize ->NO
#TODO calibration-> lof>x then  p=0.5+ - >NO
#TODO log transform ->NO...!!
#TODO use meta features....
#TODO dicretize continous data by cut and qcut to enlarge sparse matrix ...?
#TODO use new features to train xtra rf
#TODO pickle to save model + datasets
#TODO mix features...https://github.com/tuzzeg/detect_insults/blob/master/README.md
#TODO https://class.coursera.org/nlp/lecture/index and http://nlp.stanford.edu/~wcmac/papers/20050421-smoothing-tutorial.pdf
#TODO analyse alt= data!!!
#TODO use lof as regular feature


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
      words=[item for sublist in words for item in sublist]
      if hasattr(self.wnl,'stem'):
	  words=[self.wnl.stem(t) for t in words]
      else:
	  words=[self.wnl.lemmatize(t) for t in words]
      return words
    
def dfinfo(X_all):
    print "##Basic data##\n",X_all
    print "##Details##\n",X_all.ix[:,0:2].describe()
    print "##Details##\n",X_all.ix[:,2:3].describe()
    print "##Details##\n",X_all.ix[:,3:7].describe()

def prepareDatasets(vecType='hV',useSVD=0,useJson=True,useHTMLtag=True,useAddFeatures=True,usePosTag=True,useAlcat=False,useGreedyFilter=False,char_ngram=5,loadTemp=False,usewordtagSmoothing=False,usetagwordSmoothing=False,usePosTagNew=False):
    """
    Load Data into pandas and preprocess features
    """
    print "loading dataset..."
    X = pd.read_csv('../stumbled_upon/data/train.tsv', sep="\t", na_values=['?'], index_col=1)
    X_test = pd.read_csv('../stumbled_upon/data/test.tsv', sep="\t", na_values=['?'], index_col=1)
    y = X['label']
    y = pd.np.array(y)
    X = X.drop(['label'], axis=1)
    # Combine test and train while we do our preprocessing
    X_all = pd.concat([X_test, X])
    print "Original shape:",X_all.shape
    
    if loadTemp:
	Xs = pd.read_csv('../stumbled_upon/data/Xtemp.csv', sep=",", index_col=0)
	Xs_test = pd.read_csv('../stumbled_upon/data/Xtemp_test.csv', sep=",", index_col=0)
	
	return (Xs,y,Xs_test,X_test.index,X.index)
    
    #vectorize data#
    #vectorizer = HashingVectorizer(ngram_range=(1,2), non_negative=True)
    if vecType=='hV':
	warnings.filterwarnings("ignore", category=UserWarning)
	print "Using hashing vectorizer..."
	#vectorizer = HashingVectorizer(stop_words='english',ngram_range=(1,2),analyzer="word", non_negative=True, norm='l2', n_features=2**19)
	vectorizer = HashingVectorizer(stop_words=None,ngram_range=(char_ngram,char_ngram),analyzer="char", non_negative=True, norm='l2', n_features=2**18)
    elif vecType=='tfidfV':
	print "Using tfidfV..."
	vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1,2),stop_words=None,max_features=None,binary=False,min_df=4,strip_accents='unicode',tokenizer=NLTKTokenizer())
	#vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1,2),stop_words=None,max_features=None,binary=True,min_df=5,strip_accents='unicode')
	#vectorizer = TfidfVectorizer(ngram_range=(1,1),max_features=2**14,sublinear_tf=True,min_df=3,tokenizer=NLTKTokenizer(),stop_words=None)
	#vectorizer = TfidfVectorizer(ngram_range=(1,1),max_features=2**14,sublinear_tf=True,min_df=2,stop_words=None)#fast
	#vectorizer = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode',analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), sublinear_tf=True, norm=u'l2')#opt
    elif vecType=='tfidfV_small':
	vectorizer = TfidfVectorizer(ngram_range=(1,2),max_features=2**12,sublinear_tf=True,min_df=4,stop_words=None)#fast
    elif vecType=='tfidfV_large':
	vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1,2),stop_words=None,max_features=None,binary=False,min_df=2,strip_accents='unicode',tokenizer=NLTKTokenizer())
	#vectorizer = TfidfVectorizer(min_df=2,  max_features=None, strip_accents='unicode',analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), sublinear_tf=True, norm=u'l2')#opt
    elif vecType=='test':
       print "Test mode..."
       #vectorizer = CountVectorizer(ngram_range=(1,2),analyzer='word',max_features=2**14,min_df=3,tokenizer=NLTKTokenizer(),stop_words=None)
       #vectorizer = HashingVectorizer(binary=False,stop_words=None,ngram_range=(char_ngram,char_ngram),analyzer="char", non_negative=True, norm='l2', n_features=2**18)
       vectorizer = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode',analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 1), sublinear_tf=True, norm=u'l2')
    else:
	print "Using count vectorizer..."
	#vectorizer = CountVectorizer(ngram_range=(1,2),analyzer='word',max_features=2**18)
	#vectorizer = CountVectorizer(lowercase=False,analyzer="char_wb",ngram_range=(4,4),max_features=2**14,stop_words='english')#AUC = 0.781
	#vectorizer = CountVectorizer(lowercase=False,analyzer="char",ngram_range=(4,4),max_features=2**14,stop_words='english')#AUC= 0.786
	#vectorizer = CountVectorizer(lowercase=False,analyzer="char",ngram_range=(4,4),max_features=2**18,stop_words='english')#AUC= 0.798
	#vectorizer = CountVectorizer(lowercase=False,analyzer="char",ngram_range=(5,5),max_features=2**14,stop_words='english')#slow and low score 0.786
	vectorizer = CountVectorizer(lowercase=False,analyzer="char",ngram_range=(char_ngram,char_ngram),max_features=2**18,stop_words=None)#AUC=  0.815 1400s
	#vectorizer = CountVectorizer(lowercase=True,analyzer="char",ngram_range=(5,5),max_features=2**18,stop_words='english')#AUC= 0.813  1400s
	#vectorizer = CountVectorizer(lowercase=False,analyzer="char",ngram_range=(4,5),max_features=2**18,stop_words='english')#AUC=   aufgehängt memory?
	#vectorizer = CountVectorizer(lowercase=False,analyzer="char",ngram_range=(5,5),max_features=2**16,stop_words=None)#AUC= 0.806 682s
    
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

    
    if useSVD>1:
	if useAddFeatures==True:
	    X_raw=crawlRawData(X_all)
	    X_all=pd.concat([X_all,X_raw], axis=1)
	    X_all = featureEngineering(X_all)
	
	print "Actual shape:",X_all.shape
	#SVD of text data (LSA)
	print "SVD of sparse data with n=",useSVD
	tsvd=TruncatedSVD(n_components=useSVD, algorithm='randomized', n_iterations=5, tol=0.0)
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
	    tsvd=TruncatedSVD(n_components=useSVD2, algorithm='randomized', n_iterations=5, tol=0.0)
	    X_char=tsvd.fit_transform(char_ngrams)
	    X_char=pd.DataFrame(np.asarray(X_char),index=X_all.index,columns=["char"+str(x) for x in xrange(useSVD2)])
	    print "X_char",X_char
	    X_svd=pd.concat([X_svd,X_char], axis=1)
	
	if usePosTag:
	    #posTagging(X_all)
	    X_pos=pd.read_csv('../stumbled_upon/data/postagged.csv', sep=",", na_values=['?'], index_col=0)
	    X_svd=pd.concat([X_svd,X_pos], axis=1)
	    #new postag
	if usePosTagNew:
	    X_pos=pd.read_csv('../stumbled_upon/data/postagfeats.csv', sep=",", na_values=['?'], index_col=0)
	    X_svd=pd.concat([X_svd,X_pos], axis=1)
	
	#print "##X_svd##\n",X_svd
	X_all= X_all.drop(['boilerplate','url'], axis=1)
	X_all= X_all.drop(['hasDomainLink','framebased','news_front_page','embed_ratio'], axis=1)
	if useJson: 
	    X_rest= X_all.drop(['body','url2','title','alchemy_category'], axis=1)	    
	else:
	    X_rest= X_all.drop(['alchemy_category'], axis=1)
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
	   
	
	if useGreedyFilter:
	    print X_svd
	    #print X_svd.columns
	    #X_svd=X_svd.loc[:,[1,4,3,8,5,'linkwordscore',6,'char2',9,'url_contains_foodstuff',22,26,'MOD',33,'alchemy_category_score',24,45,'spelling_errors_ratio',43]]#Rgreedy
	    #X_svd=X_svd.loc[:,[1,4,3,8,5,'linkwordscore',6,'char2',9,'url_contains_foodstuff',22,26,'MOD',33,'alchemy_category_score',24,45,'spelling_errors_ratio',43,'frameTagRatio',19,21,25,0,'url_length',48,'TO','char5','url_contains_news','compression_ratio',37,'VD','twitter_ratio',49,'is_news','url_contains_sweetstuff',42,17,'url_contains_health',20,'char4',16,'DET',23,'commonlinkratio_2',41,'image_ratio',7,'wwwfacebook_ratio','char0']]
	    #X_svd=X_svd.loc[:,[1, 2, 4, u'url_contains_foodstuff', 9, 0, 8, u'CNJ', u'url_contains_recipe', 33, 6, u'non_markup_alphanum_characters', 10, u'body_length', 15, 5, 3, u'char2', 12, 11, 14, 21, 31, u'frameTagRatio', 7, 25, u'N', 22, 17, 16, 23, 19, 47, 18, u'linkwordscore', 29, 46, 30, u'V', 39, 32]]#rf feature importance sklearn
	    #X_svd=X_svd.loc[:,[1, 4, 9, 8, 0, u'url_contains_foodstuff', 2, 58, 67, 75, 71, u'CNJ', 44, 21, u'non_markup_alphanum_characters', 15, 42, u'linkwordscore', u'frameTagRatio', 36, u'logn_newline', 65, 47, 29, 33, 64, u'body_length', u'DET', 73, 56, 12, u'P', 14, 6, u'char0', 97, 37, 52, 83, 79, 17, u'avglinksize', u'char8', 22, 39, u'char4', u'wwwfacebook_ratio', u'url_length', u'ADJ', u'char1', 85, 30, 72, 62, 49, 28, 11, 59, 89, u'n_comment', 78, 55, 53, u'MOD', u'compression_ratio', 54, u'spelling_errors_ratio', u'commonlinkratio_1']]#GBM feature selection
	    X_svd=X_svd.loc[:,[1, 4, 9, 8, 0, 2, u'url_contains_foodstuff', 58, 67, 75, u'CNJ', u'linkwordscore', 21, 64, 15, u'non_markup_alphanum_characters', u'frameTagRatio', 44, 47, 42, u'P', 29, 36, 33, 73, u'DET', 97, 12, 56, 71, 52, 65, 59, 37, u'logn_newline', 6, 85, u'char0', 22, 83, u'url_length', u'body_length', 14, 17, 30, u'avglinksize', 62, u'compression_ratio', 39, u'ADJ', u'char1', 53, 78, 49, 11, 54, 79, 89, u'char4', u'char8']]#GBM feature selection
	    #X_svd=X_svd.loc[:,[1,4,3,8,5,u'linkwordscore']]
	    print X_svd
	
	
	print "##X_svd,final##\n",X_svd
	#X_rest=X_svd
	print "Dim: X_svd:",X_svd.shape    
	X_svd_train = X_svd[len(X_test.index):]
	X_svd_test = X_svd[:len(X_test.index)]
	return(X_svd_train,y,X_svd_test,X_test.index,X.index)
    else:
	if usePosTag:
	    #create new body with postag features!
	    #postagFeatures(X_all)
	    X_pos=pd.read_csv('../stumbled_upon/data/postagfeats.csv', sep=",", na_values=['?'], index_col=0)
	    
	    #vectorize data
	    #vectorizer = CountVectorizer(ngram_range=(1,1),analyzer='word',max_features=2**10)
	    vectorizer = HashingVectorizer(ngram_range=(1,1),analyzer='word',max_features=2**18,token_pattern=r'\w{1,}')
	    poscounts=vectorizer.fit_transform(X_pos['postagfeats'])  
	    #print vectorizer.get_feature_names()
	    print poscounts.shape
	    print type(body_counts)
	    print body_counts.shape
	    body_counts = sparse.hstack((body_counts,poscounts),format="csr")
	    
	if usewordtagSmoothing or usetagwordSmoothing:
	    #create tag-words and word-tag
	    #postagSmoothing(X_all)
	    X_smoothed=pd.read_csv('../stumbled_upon/data/postagsmoothed.csv', sep=",", na_values=['?'], index_col=0,encoding='utf-8')
	    print X_smoothed
	    
	    #word-tags
	    #vectorizer = CountVectorizer(ngram_range=(1,1),analyzer='word',max_features=2**10,token_pattern=r'\w{1,}')
	    #vectorizer = HashingVectorizer(ngram_range=(1,1),analyzer='word',max_features=2**18,token_pattern=r'\w{1,}')
	    vectorizer = TfidfVectorizer(min_df=3,  max_features=2**18, strip_accents='unicode',analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), sublinear_tf=True, norm=u'l2')
	    if usewordtagSmoothing:
		print "Using word-tag smoothing..." 
		col='wordtag'
		wordtagcounts=vectorizer.fit_transform(X_smoothed[col])
		print wordtagcounts.shape
		body_counts=wordtagcounts
		#body_counts = sparse.hstack((body_counts,wordtagcounts),format="csr")
		
	    if usetagwordSmoothing:
		print "Using tag-word smoothing..."
		col='tagword'
		tagwordcounts=vectorizer.fit_transform(X_smoothed[col])
		print tagwordcounts.shape
		body_counts = sparse.hstack((body_counts,tagwordcounts),format="csr")	
		
	    #print "Feature names:",vectorizer.get_feature_names()
	    print type(body_counts)
	    print body_counts.shape
	    
	    print "density:",density(body_counts)
           
           
           
	Xs = body_counts[len(X_test.index):]
	Xs_test = body_counts[:len(X_test.index)]
	#conversion to array necessary to work with integer indexing, .iloc does not work with this version
	return (Xs,y,Xs_test,X_test.index,X.index)
	
    

def one_hot_encoder(data, col, replace=False):
    """ Takes a dataframe and a list of columns that need to be encoded.
    Returns a 3-tuple comprising the data, the vectorized data,{
    and the fitted vectorizor.
    credits to https://gist.github.com/kljensen/5452382
    """
    vec=DictVectorizer()
    tmp=[]
    for row in data[col].itertuples():
	d=dict({'cat': row[1]})
	tmp.append(d)  
    tmp=vec.fit_transform(tmp).toarray()
    vecData = pd.DataFrame(tmp)
    vecData.columns = vec.get_feature_names()
    vecData.index = data.index
    if replace is True:
	data = data.drop(col, axis=1)
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
    ly = np.asarray(ly)
    print "Model evaluation..."
    folds=8
    #parameters=np.logspace(-14, -7, num=8, base=2.0)#SDG
    #parameters=np.logspace(-7, 0, num=8, base=2.0)#LG
    #parameters=[250,500,1000,2000]#rf
    parameters=[100.0,10.0,8.0,5.0,2.0,1.5,1.2]#chi2
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
	#if isinstance(lmodel,RandomForestClassifier) :
	#    lmodel.set_params(max_features=p)
	#if isinstance(lmodel,GradientBoostingClassifier):
	#    lmodel.set_params(max_depth=p)
        #print lmodel.get_params()
        cv = KFold(lXs.shape[0], n_folds=folds,indices=True, random_state=j)
	scores=np.zeros(folds)	
	for i, (train, test) in enumerate(cv):
	    #print("Extracting %s best features by a chi-squared test" % p)
	    #ch2 = SelectKBest(chi2, k=p)
	    #ch2 = SelectPercentile(chi2,percentile=p)
	    #Xtrain = ch2.fit_transform(lXs[train], ly[train])
	    #Xtest = ch2.transform(lXs[test]) 
	    Xtrain = lXs.iloc[train]
	    Xtest = lXs.iloc[test]
	    lmodel.fit(Xtrain, ly[train])
	    oobpreds[test,j] = lmodel.predict_proba(Xtest)[:,1]
	    scores[i]=roc_auc_score(ly[test],oobpreds[test,j])
	    #print "AUC: %0.2f " % (scores[i])
	    #save oobpredictions
	print "Iteration:",j," parameter:",p,
	print " <AUC>: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std()),
	print " AUC oob: %0.3f" %(roc_auc_score(ly,oobpreds[:,j]))
	#Filter data
	#predmod=lofFilter(oobpreds[:,j],p)
	#print " AUC calibrated: %0.3f" %(roc_auc_score(ly,predmod))
	
    scores=[roc_auc_score(ly,oobpreds[:,j]) for j in xrange(len(parameters))]
    plt.plot(parameters,scores,'ro')
    
def ensembleBuilding(lXs,ly):
    """
    train ensemble
    """
    print "Ensemble training..."
    folds=8
    parameters=np.logspace(-15, -10, num=200, base=2.0)
    #iterpos=[10,20,30,40,50]
    #regula=[0.001,0.1,1.0,10.0,100.0]
    parameters=nprnd.choice([0.1,10,1.0],10)
    loss=['l2','l2']
    #parameters=nprnd.choice(parameters, 20)
    classifiers = {}
    for p in parameters:
        l1ratio=nprnd.ranf()
        perc=85.0+nprnd.ranf()*15.0
        #iterations=choice(iterpos)
        l1l2=choice(loss)
	dic = {'LOG_C'+str(p)+'_perc'+str(perc)+'_loss'+str(l1l2): Pipeline([('filter', SelectPercentile(chi2, percentile=perc)), ('model', LogisticRegression(penalty=l1l2, tol=0.0001, C=p))])}
	#dic ={'SDG_alpha'+str(p)+'_L1'+str(l1ratio): SGDClassifier(alpha=p, n_iter=choice(iterpos),penalty='elasticnet',l1_ratio=l1ratio,shuffle=True,random_state=np.random.randint(0,100),loss='log')}
	#dic ={'PIP_SDG_iter'+str(p)+'_perc'+str(perc): Pipeline([('filter', SelectPercentile(chi2, percentile=perc)), ('model', SGDClassifier(alpha=0.00014, n_iter=p,shuffle=True,random_state=p,loss='log',penalty='elasticnet',l1_ratio=l1ratio))])}
	#dic ={'PIP_SDG_alpha'+str(p)+'_perc'+str(perc)+'_iter'+str(iterations): Pipeline([('filter', SelectPercentile(chi2, percentile=perc)),('model', SGDClassifier(alpha=p, n_iter=iterations,penalty='elasticnet',l1_ratio=l1ratio,shuffle=True,random_state=np.random.randint(0,100),loss='log'))])}
	classifiers.update(dic)
    #dic ={'NB': BernoulliNB(alpha=1.0)}
    #classifiers.update(dic)
    dic ={'LG1': LogisticRegression(penalty='l2', tol=0.0001, C=1.0)}
    classifiers.update(dic)
    dic ={'SDG1': SGDClassifier(alpha=0.0001, n_iter=50,shuffle=True,loss='log',penalty='l2')}
    classifiers.update(dic)
    dic ={'SDG2': SGDClassifier(alpha=0.00014, n_iter=50,shuffle=True,loss='log',penalty='elasticnet',l1_ratio=0.99)}
    classifiers.update(dic)
    #dic ={'LG2': LogisticRegression(penalty='l1', tol=0.0001, C=1.0,random_state=42)}
    #classifiers.update(dic)
    dic = {'NB2': Pipeline([('filter', SelectPercentile(f_classif, percentile=25)), ('model', BernoulliNB(alpha=0.1))])}
    classifiers.update(dic)    
    #dic ={'KNN': Pipeline([('filter', SelectPercentile(f_classif, percentile=25)), ('model', KNeighborsClassifier(n_neighbors=150))])}
    #classifiers.update(dic)  
    
    oobpreds=np.zeros((lXs.shape[0],len(classifiers)))
    for j,(key, lmodel) in enumerate(classifiers.iteritems()):
        #print lmodel.get_params()
        #cv = StratifiedKFold(ly, n_folds=folds, indices=True)
        cv = KFold(lXs.shape[0], n_folds=folds, indices=True,random_state=j,shuffle=True)
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
    
    #do another crossvalidation for weights and train blender...?
    #blender=LogisticRegression(penalty='l2', tol=0.0001, C=1.0)
    blender=AdaBoostClassifier(learning_rate=0.1,n_estimators=150,algorithm="SAMME.R")
    #blender=ExtraTreesRegressor(n_estimators=200,max_depth=None,n_jobs=1, max_features='auto',oob_score=False,random_state=42)
    #blender=ExtraTreesClassifier(n_estimators=200,max_depth=None,min_samples_leaf=10,n_jobs=1,criterion='entropy', max_features='auto',oob_score=False,random_state=42)
    cv = KFold(lXs.shape[0], n_folds=folds, indices=True,random_state=42)
    blend_scores=np.zeros(folds)
    blend_oob=np.zeros((lXs.shape[0]))
    for i, (train, test) in enumerate(cv):
	Xtrain = oobpreds[train]
	Xtest = oobpreds[test]
	blender.fit(Xtrain, ly[train])
	if hasattr(blender,'predict_proba'):
	    blend_oob[test] = blender.predict_proba(Xtest)[:,1]
	else:
	    blend_oob[test] = blender.predict(Xtest)
	blend_scores[i]=roc_auc_score(ly[test],blend_oob[test])
    print " <AUC>: %0.3f (+/- %0.3f)" % (blend_scores.mean(), blend_scores.std()),
    print " AUC oob after blending: %0.3f" %(roc_auc_score(ly,blend_oob))
    if hasattr(blender,'coef_'):
      print "Coefficients:",blender.coef_
    
    plt.plot(range(len(classifiers)),scores,'ro')
    return(classifiers,blender,oob_avg)
    

def ensemblePredictions(classifiers,blender,lXs,ly,lXs_test,lidx,lidx_train,oob_avg,filename):
    """   
    Makes prediction
    """ 
    print "Make final ensemble prediction..."
    #make prediction for each classifiers
    preds=np.zeros((lXs_test.shape[0],len(classifiers)))
    for j,(key, lmodel) in enumerate(classifiers.iteritems()):
        lmodel.fit(lXs,ly)
	preds[:,j]=lmodel.predict_proba(lXs_test)[:,1]
    #blend results
    finalpred=blender.predict_proba(preds)[:,1]   
    print "Saving predictions to: ",filename
    print "Final test dataframe:",lXs_test.shape
    oob_avg = pd.DataFrame(oob_avg,index=lidx_train,columns=['label'])
    pred_df = pd.DataFrame(finalpred, index=lidx, columns=['label'])
    pred_df = pd.concat([pred_df, oob_avg])
    pred_df.index.name='urlid'
    pred_df.to_csv(filename)
    
def pyGridSearch(lmodel,lXs,ly):  
    """   
    Grid search with sklearn internal tool
    """ 
    print "Grid search..."
    #parameters = {'C':[1000,10000,100], 'gamma':[0.001,0.0001]}
    #parameters = {'max_depth':[6,7], 'learning_rate':[0.010,0.008],'n_estimators':[300,500]}#gbm
    #parameters = {'max_depth':[2], 'learning_rate':[0.01,0.001],'n_estimators':[3000]}#gbm
    #parameters = {'n_estimators':[500], 'max_features':[5,10,15]}#rf
    #parameters = {'n_estimators':[250,100,50], 'learning_rate':[0.1,0.01,0.5]}#adaboost
    #parameters = {'n_estimators':[400,600,800], 'max_features':[10,15,20],'min_samples_leaf':[10,20]}#rf
    #parameters = {'C':[0.1,1,10]}#SVC
    #parameters = {'filter__percentile': [100,80,50,25] , 'model__alpha':[1.0,0.8,0.5,0.1]}#opt
    #parameters = {'filter__percentile': [16,15,14,13,12] , 'model__n_neighbors':[125,130,135,150,200]}#knn
    #parameters = {'n_neighbors':[1,2,3,5,8,10]}#knn
    #parameters = {'filter__percentile': [100,80,50], 'model__n_estimators': [200], 'model__max_features':['auto'], 'model__min_samples_leaf':[5] }#rf
    #parameters = {'filter__percentile': [100,95,80,70,60,50,25], 'model__C': [0.5,1.0, 10.0], 'model__intercept_scaling': [0.1,1.0,10,100,1000] }#pipeline
    parameters = {'filter__percentile': [100,95,80,70,60,50,25], 'model__C': [0.5,1.0, 10.0] }#pipeline
    clf_opt = grid_search.GridSearchCV(lmodel, parameters,cv=8,scoring='roc_auc',n_jobs=4,verbose=1)
    clf_opt.fit(lXs,ly)
    
    for params, mean_score, scores in clf_opt.grid_scores_:
        print("%0.3f (+/- %0.3f) for %r"
              % (mean_score.mean(), scores.std(), params))
    return(clf_opt.best_estimator_)
    
def buildModel(lmodel,lXs,ly,feature_names=None):
    """   
    Final model building part
    """ 
    print "Xvalidation..."
    scores = cross_validation.cross_val_score(lmodel, lXs, ly, cv=8, scoring='roc_auc',n_jobs=4)
    print "AUC: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std())
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
    if not hasattr(forest,'feature_importances_'): 
      print "Missing attribute feature_importances_"
      return
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
    #print "Xreduced_test:",Xreduced_test
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
	    #Xold.to_csv("../stumbled_upon/data/Xlarge_"+str(i)+".csv")
	    #Xold_test.to_csv("../stumbled_upon/data/XXlarge_test_"+str(i)+".csv")
	return(Xold,Xold_test)



def removeInstances(lXs,ly,preds,t,returnSD=True):
	#now remove examples from train
	if returnSD:
	    res=preds
	else:
	    res=np.abs(ly-preds)
	d={'abs_err' : pd.Series(res)}
	res=pd.DataFrame(d)
	res.index=lXs.index
	lXs_reduced=pd.concat([lXs,res], axis=1)
	boolindex=lXs_reduced['abs_err']<t
	lXs_reduced=lXs_reduced[boolindex]
	#print lXs_reduced.shape
	#ninst[i]=len(Xtrain.index)-len(lXs_reduced.index)
	lXs_reduced = lXs_reduced.drop(['abs_err'], axis=1)
	#print "New dim:",lXs_reduced.shape
	ly_reduced=ly[np.asarray(boolindex)]
	return (lXs_reduced,ly_reduced)

	
def getOOBCVPredictions(lmodel,lXs,lXs_test,ly,folds=8,repeats=1,returnSD=True):
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
	if returnSD:
	    oob_avg=np.std(oobpreds,axis=1)
	return(oob_avg)
	

def lofFilter(pred,threshhold=10.0,training=True):
	"""
	#filter data according to local outlier frequency.. bringt nichts...
	"""
	indices=[]
	global test_indices
	lof = pd.read_csv("../stumbled_upon/data/lof.csv", sep=",", index_col=0)
	lof = lof[len(test_indices):]
	avg=np.mean(pred)
	for i in xrange(len(lof.index)):
	    #print lof.iloc[i,0]
	    if lof.iloc[i,0]>threshhold:
		pred[i]=avg
		indices.append(i)
	#print indices
	print "threshhold:,",threshhold,"n,changed:",len(indices)," mean:",avg
	return pred
	
	
def filterClassNoise(lmodel,lXs,lXs_test,ly):
	"""
	Removes training samples which could be class noise
	Done in outer XVal loop
	precision: Wieviel falsche habe ich erwischt
	recall: wieviele richtige sind durch die Lappen gegangen
	"""
	threshhold=[0.045,0.04,0.035]
	folds=10
	print "Filter strongly misclassified classes..."
	#rdidx=random.sample(xrange(1000), 20)
	#print rdidx
	#lXs = lXs.iloc[rdidx]
	#ly = ly[rdidx]
	preds=getOOBCVPredictions(lmodel,Xs,Xs_test,y,folds,10)
	print preds
	plt.hist(preds,bins=20)
	plt.show()
	#print "stdev:",std
	#should be oob or cvalidated!!!!
	#preds = lmodel.predict_proba(lXs)[:,1]
	scores=np.zeros((folds,len(threshhold)))
	oobpreds=np.zeros((lXs.shape[0],folds))
	for j,t in enumerate(threshhold):
	    #XValidation
	    cv = KFold(lXs.shape[0], n_folds=folds, indices=True,random_state=j,shuffle=True)	    	    
	    ninst=np.zeros(folds)	    
	    for i, (train, test) in enumerate(cv):
		Xtrain = lXs.iloc[train]
		ytrain=  ly[train]		
		#now remove examples from train
		lXs_reduced,ly_reduced = removeInstances(Xtrain,ytrain,preds[train],t)
		ninst[i]=len(Xtrain.index)-len(lXs_reduced.index)
		lmodel.fit(lXs_reduced, ly_reduced)
		
		#testing data, not manipulated
		Xtest = lXs.iloc[test]
		oobpreds[test,j] = lmodel.predict_proba(Xtest)[:,1]
		
		scores[i,j]=roc_auc_score(ly[test],oobpreds[test,j])

	    print "Threshhold: %0.3f  <AUC>: %0.3f (+/- %0.3f) removed instances: %4.2f" % (t, scores[:,j].mean(), scores[:,j].std(), ninst.mean() ),
	    print " AUC oob: %0.3f" %(roc_auc_score(ly,oobpreds[:,j]))
	scores=np.mean(scores,axis=0)
	print scores
	plt.plot(threshhold,scores,'ro')
	top = np.argsort(scores)
	optt = threshhold[top[-1]]
	print "Optimum threshhold %4.2f index: %d with score: %4.4f" %(optt,top[-1],scores[top[-1]])
	lXs_reduced,ly_reduced = removeInstances(lXs,ly,preds,optt)	
	return(lXs_reduced,lXs_test,ly_reduced)


def showMisclass(lXs,lXs_test,ly,t=0.95):
    """
    Show bubble plot of strongest misclassifications...
    """
    folds=8
    repeats=3
    print "Show strongly misclassified classes..."
    preds=getOOBCVPredictions(model,Xs,Xs_test,y,folds,repeats,returnSD=False)
    res=np.abs(ly-preds)
    err=pd.DataFrame({'abs_err' : pd.Series(res)})
    ly=pd.DataFrame({'y' : pd.Series(y)})
    preds=pd.DataFrame({'preds' : pd.Series(preds)})
    lXs_plot=pd.concat([ly,preds,err], axis=1)
    lXs_plot.index=lXs.index
    lXs_plot=pd.concat([lXs_plot,lXs], axis=1)
    boolindex=lXs_plot['abs_err']>t
    lXs_plot=lXs_plot[boolindex]
    print "Number of instances left:",lXs_plot.shape[0]
    col1='1'
    col2='2'
    #bubblesizes=lXs_plot['non_markup_alphanum_characters']
    bubblesizes=lXs_plot['body_length']
    print bubblesizes
    #print lXs_plot.ix[3155]
    sct = plt.scatter(lXs_plot[col1], lXs_plot[col2],c=lXs_plot['y'],s=bubblesizes, linewidths=2, edgecolor='black')
    sct.set_alpha(0.75)
    for row_index, row in lXs_plot.iterrows():
	plt.text(row[col1], row[col2],row_index,size=10,horizontalalignment='center')
	print "INDEX:",row_index," Y:",row['y']," PRED:",row['preds']," body_length:",row['body_length']
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title("red:1 blue:0")
    plt.show()
    #now got to page interactively

def scaleData(lXs,lXs_test,cols=None):
    """
    standard+transformation scaling of data
    """
    print "Data scaling..."
    lX_all = pd.concat([lXs_test, lXs])   
    lX_all[cols].hist()   
    lX_all[cols] = (lX_all[cols] - lX_all[cols].min()+10e-10) 
    print lX_all[cols].describe()
    lX_all[cols]=lX_all[cols].apply(np.sqrt)
    lX_all[cols] = (lX_all[cols] - lX_all[cols].mean()) / (lX_all[cols].max() - lX_all[cols].min()) 
    print lX_all[cols].describe()
    lX_all[cols].hist()
    plt.show()
    
    #divide again
    lXs = lX_all[len(lXs_test.index):]
    lXs_test = lX_all[:len(lXs_test.index)]
    return (lXs,lXs_test)


if __name__=="__main__":
    """   
    MAIN PART
    """ 
    # Set a seed for consistant results
    t0 = time()
    np.random.seed(123)
    print "numpy:",np.__version__
    print "pandas:",pd.__version__
    #print pd.util.terminal.get_terminal_size()
    pd.set_printoptions(max_rows=300, max_columns=8)
    print "scipy:",sp.__version__
    #variables
    #(Xs,y,Xs_test,test_indices,train_indices) = prepareDatasets('tfidfV',useSVD=50,useJson=True,useHTMLtag=True,useAddFeatures=True,usePosTag=True,useAlcat=True,useGreedyFilter=False)#opt SVD=50
    #(Xs,y,Xs_test,test_indices,train_indices) = prepareDatasets('tfidfV_small',useSVD=50,useJson=True,useHTMLtag=False,useAddFeatures=True,usePosTag=False,useAlcat=False,useGreedyFilter=False)
    #Xs=pd.DataFrame(Xs.todense())
    #Xs_test=pd.DataFrame(Xs_test.todense())
    #(Xs,y,Xs_test,test_indices,train_indices) = prepareDatasets('hV',useSVD=10,useJson=False,useHTMLtag=False,useAddFeatures=False,usePosTag=False,useAlcat=True,useGreedyFilter=False,char_ngram=1,loadTemp=True)
    #(Xs,y,Xs_test,test_indices,train_indices) = prepareDatasets('tfidfV',useSVD=100,useJson=True,useHTMLtag=True,useAddFeatures=True,usePosTag=True,useAlcat=False,useGreedyFilter=False)#opt SVD=50
    #(Xs,y,Xs_test,test_indices,train_indices) = prepareDatasets('tfidfV',useSVD=300,useJson=True,useHTMLtag=True,useAddFeatures=True,usePosTag=True,useAlcat=True,useGreedyFilter=False)#opt SVD=50
    #(Xs,y,Xs_test,test_indices,train_indices) = prepareDatasets('tfidfV',useSVD=2,useJson=True,useHTMLtag=True,useAddFeatures=True,usePosTag=True,useAlcat=True,useGreedyFilter=False)#
    (Xs,y,Xs_test,test_indices,train_indices) = prepareDatasets('test',useSVD=0,useJson=True,usePosTag=False,usewordtagSmoothing=False,usetagwordSmoothing=True)
    #(Xs,y,Xs_test,test_indices,train_indices) = prepareDatasets('tfidfV_large',useSVD=0,useJson=True)
    #Xs.to_csv("../stumbled_upon/data/Xtemp.csv")
    #Xs_test.to_csv("../stumbled_upon/data/Xtemp_test.csv")
    #Xs.to_csv("../stumbled_upon/data/Xlarge.csv")
    #Xs_test.to_csv("../stumbled_upon/data/Xlarge_test.csv")

    #(Xs,y,Xs_test,test_indices) = prepareSimpleData()
    print "Dim X (training):",Xs.shape
    print "Type X:",type(Xs)
    print "Dim X (test):",Xs_test.shape
    # Fit a model and predict
    #model = SGDClassifier(alpha=.0001, n_iter=50,penalty='elasticnet',l1_ratio=0.2,shuffle=True,loss='log')
    #model = SGDClassifier(alpha=0.0005, n_iter=50,shuffle=True,loss='log',penalty='l2',n_jobs=4)#opt  
    #model = SGDClassifier(alpha=0.0001, n_iter=50,shuffle=True,loss='log',penalty='l2',n_jobs=4)#opt simple processing
    #model = SGDClassifier(alpha=0.00014, n_iter=50,shuffle=True,loss='log',penalty='elasticnet',l1_ratio=0.99)
    #model = LogisticRegression(penalty='l2', tol=0.0001, C=1.0)#opt
    model = Pipeline([('filter', SelectPercentile(chi2, percentile=50)), ('model', LogisticRegression(penalty='l2', tol=0.0001, C=0.5))])
    #model = Pipeline([('filter', SelectPercentile(chi2, percentile=70)), ('model', LogisticRegression(penalty='l2', tol=0.0001, C=1.0))])
    #model = Pipeline([('filter', SelectPercentile(f_classif, percentile=15)), ('model', KNeighborsClassifier(n_neighbors=150))])
    #model = Pipeline([('filter', SelectPercentile(chi2, percentile=20)), ('model', MultinomialNB(alpha=0.1))])
    #model = LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1.0, class_weight=None)#opt kaggle params
    #model = LogisticRegressionMod(penalty='l2', dual=False, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1.0, class_weight=None)#opt kaggle params
    #model = Pipeline([('filter', SelectPercentile(f_classif, percentile=100)), ('model', LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=10.0, class_weight=None))])
    #model = AdaBoostClassifier(base_estimator=LogisticRegressionMod(penalty='l2', dual=False, tol=0.0001, C=1, fit_intercept=True,intercept_scaling=1.0),learning_rate=0.1,n_estimators=50,algorithm="SAMME.R")
    #model = KNeighborsClassifier(n_neighbors=10)
    #model=SVC(C=0.3,kernel='linear',probability=True)
    #model=LinearSVC(penalty='l2', loss='l2', dual=True, tol=0.0001, C=1.0)#no proba
    #model = SVC(C=1, cache_size=200, class_weight='auto', gamma=0.0, kernel='linear', probability=True, shrinking=True,tol=0.001, verbose=False)
    #model=   RandomForestClassifier(n_estimators=500,max_depth=None,min_samples_leaf=10,n_jobs=1,criterion='entropy', max_features='auto',oob_score=False)
    #model = Pipeline([('filter', SelectPercentile(f_classif, percentile=25)), ('model', BernoulliNB(alpha=0.1))])#opt dense 0.855
    #model = Pipeline([('filter', SelectPercentile(f_classif, percentile=50)), ('model', BernoulliNB(alpha=0.1))])#opt sparse 0.849
    #model = RandomForestClassifier(n_estimators=500,max_depth=None,min_samples_leaf=12,n_jobs=1,criterion='entropy', max_features='auto',oob_score=False)#opt greedy approach
    #model = AdaBoostClassifier(n_estimators=500,learning_rate=0.1)

    #model = ExtraTreesClassifier(n_estimators=100,max_depth=None,min_samples_leaf=10,n_jobs=4,criterion='entropy', max_features=20,oob_score=False)#opt
    #model = AdaBoostClassifier(n_estimators=50,learning_rate=0.1)
    #model = GradientBoostingClassifier(loss='deviance', learning_rate=0.01, n_estimators=500, subsample=0.5, min_samples_split=6, min_samples_leaf=10, max_depth=5, init=None, random_state=123,verbose=False)#opt 0.883
    #model = SVC(C=1, cache_size=200, class_weight='auto', gamma=0.0, kernel='rbf', probability=True, shrinking=True,tol=0.001, verbose=False)  
    #modelEvaluation(model,Xs,y)
    model=pyGridSearch(model,Xs,y)
    #(gclassifiers,gblender,oob_avg)=ensembleBuilding(Xs,y)
    #ensemblePredictions(gclassifiers,gblender,Xs,y,Xs_test,test_indices,train_indices,oob_avg,'../stumbled_upon/data/lgblend.csv')
    #fit final model
    #(Xs,Xs_test)=scaleData(Xs,Xs_test,['body_length','linkwordscore','frameTagRatio','non_markup_alphanum_characters'])
    #Xs.hist()
    #print model
    #model = buildModel(model,Xs,y) 
    #print model.estimator_errors_
    #print model.estimator_weights_
    #showMisclass(Xs,Xs_test,y)
    #(Xs,Xs_test,y)=filterClassNoise(model,Xs,Xs_test,y)
    model = buildModel(model,Xs,y) 
    #(Xs,Xs_test)=iterativeFeatureSelection(model,Xs,Xs_test,y,1,1)
    #Xs.to_csv("../stumbled_upon/data/Xtemp.csv")
    #Xs_test.to_csv("../stumbled_upon/data/Xtemp_test.csv")
    #model = buildModel(model,Xs,y) 
    #lofFilter(y)
    #(Xs,Xs_test) = group_sparse(Xs,Xs_test)
    #print "Dim X (after grouping):",Xs.shape
    #makePredictions(model,Xs_test,test_indices,'../stumbled_upon/submissions/sub0410b.csv')	            
    print("Model building done in %fs" % (time() - t0))
    plt.show()
