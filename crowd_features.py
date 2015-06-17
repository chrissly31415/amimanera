#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""  crawl data
"""

from qsprLib import *

#TODO: right after tfidf, input 2 sparse matrics: def computeSimilarityFeatures(Xs_all,Xs_all_new)
#and for dense def computeSimilarityFeatures(Xall,Xall_new,nsplit)
#http://stackoverflow.com/questions/16597265/appending-to-an-empty-data-frame-in-pandas

def computeSimilarityFeatures(vectorizer=None,nsamples=-1,stop_words=None):
    
    Xtest,Xtrain,ytrain,idx = loadData(nsamples=nsamples)
    Xall = pd.concat([Xtest, Xtrain])
    
    if vectorizer is None: vectorizer = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode', analyzer='word',ngram_range=(1, 2), use_idf=True,smooth_idf=True,sublinear_tf=True,stop_words = stop_words,token_pattern=r'\w{1,}',norm='l2')
    #TODO: right after tfidf, input 2 sparse matrics: def computeSimilarityFeatures(Xs_all,Xs_all_new)
    
    Xs_all_new = vectorizer.transform(Xall[col])
  
    train_sim = np.diag(Xs_train.dot(Xs_train_new.T).todense())
    test_sim = np.diag(Xs_test.dot(Xs_test_new.T).todense())
    
    norm1 = np.linalg.norm(Xs_train.todense(),axis=1)	
    norm2 = np.linalg.norm(Xs_train_new.todense(),axis=1)
    
    train_sim = np.divide(train_sim,norm1)
    train_sim = np.nan_to_num(np.divide(train_sim,norm2))
    
    norm1 = np.linalg.norm(Xs_test.todense(),axis=1)	
    norm2 = np.linalg.norm(Xs_test_new.todense(),axis=1)
    
    test_sim = np.divide(test_sim,norm1)
    test_sim = np.nan_to_num(np.divide(test_sim,norm2))
    
    train_sim = train_sim.reshape((train_sim.shape[0],1))
    test_sim = test_sim.reshape((test_sim.shape[0],1))
    
    similarity = np.vstack((test_sim,train_sim))
    
    print similarity.shape
    print similarity

  
    return d.DataFrame(similarity,columns=['cosine_sim'])