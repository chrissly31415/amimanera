#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""  crawl data
"""

from qsprLib import *
from nltk.stem.porter import PorterStemmer
from bs4 import BeautifulSoup
import re
import difflib
from scipy.spatial.distance import cdist


#TODO: right after tfidf, input 2 sparse matrics: def computeSimilarityFeatures(Xs_all,Xs_all_new)
#and for dense def computeSimilarityFeatures(Xall,Xall_new,nsplit)
#http://stackoverflow.com/questions/16597265/appending-to-an-empty-data-frame-in-pandas


def computeSimilarityFeatures_old(vectorizer=None,nsamples=-1,stop_words=None):
    
    Xtest,Xtrain,ytrain,idx = loadData(nsamples=nsamples)
    #Xall = pd.concat([Xtest, Xtrain])
    
    if vectorizer is None: vectorizer = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode', analyzer='word',ngram_range=(1, 2), use_idf=True,smooth_idf=True,sublinear_tf=True,stop_words = stop_words,token_pattern=r'\w{1,}',norm='l2')
    #TODO: right after tfidf, input 2 sparse matrics: def computeSimilarityFeatures(Xs_all,Xs_all_new)
    
    Xs_train_new = vectorizer.fit_transform(Xtrain[col])
    Xs_test_new = vectorizer.transform(Xtest[col])
  
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

    return pd.DataFrame(similarity,columns=['cosine_sim'])

def computeCosineSimilarity_old(Xs1,Xs2):
    """
    Takes 2 sparse matrices and computes cosine similarity
    """
    tmp = Xs1.dot(Xs2.T)
    tmp = tmp.todense()
    X_sim = np.diag(tmp)
    norm1 = np.linalg.norm(Xs1.todense(),axis=1)
    norm2 = np.linalg.norm(Xs2.todense(),axis=1)
    X_sim = np.divide(X_sim,norm1)
    X_sim = np.nan_to_num(np.divide(X_sim,norm2))
    X_sim = X_sim.reshape((train_sim.shape[0],1))
    return pd.DataFrame(similarity,columns=['cosine_sim'])

  

def computeSimilarityFeatures(Xall,columns=['query','product_title'],verbose=False):
    print "Compute scipy similarity..."
    vectorizer = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode', analyzer='word',ngram_range=(1, 5), use_idf=True,smooth_idf=True,sublinear_tf=True,stop_words = None,token_pattern=r'\w{1,}')
    Xs1 = vectorizer.fit_transform(Xall[columns[0]])
    Xs2 = vectorizer.transform(Xall[columns[1]])
    #print "Xs1",Xs1.shape
    #print "Xs2",Xs2.shape
    #print Xall['query'].iloc[:5]
    #print Xall['product_title'].iloc[:5]
    sim = computeScipySimilarity(Xs1,Xs2)
    return sim



def computeScipySimilarity(Xs1,Xs2):
    Xs1 = Xs1.todense()
    Xs2 = Xs2.todense()
    Xall_new = np.zeros((Xs1.shape[0],6))
    for i,(a,b) in enumerate(zip(Xs1,Xs2)):
	dist = cdist(a,b,'cosine')
	Xall_new[i,0] = dist	
	dist = cdist(a,b,'euclidean')
	Xall_new[i,1] = dist
	dist = cdist(a,b,'hamming')
	Xall_new[i,2] = dist
	dist = cdist(a,b,'minkowski')
	Xall_new[i,3] = dist
	dist = cdist(a,b,'cityblock')
	Xall_new[i,4] = dist
	dist = cdist(a,b,'correlation')
	Xall_new[i,5] = dist
	
    Xall_new = pd.DataFrame(Xall_new,columns=['cosine','euclidean','hammming','minkowski','cityblock','correlation'])
    print "NA:",Xall_new.isnull().values.sum()
    Xall_new = Xall_new.fillna(0.0)
    print "NA:",Xall_new.isnull().values.sum()
    return Xall_new
	



    

#other features: len(description)
#total number of matches
#query id via label_encoder
def additionalFeatures(Xall,verbose=False):
    print "Computing additional features..."
    stemmer = PorterStemmer()
    Xall_new = np.zeros((Xall.shape[0],5))
    for i in range(Xall.shape[0]):
	query = Xall["query"].iloc[i].lower()
	title = Xall["product_title"].iloc[i].lower()
	
	query=re.sub("[^a-zA-Z0-9]"," ", query)
        query= (" ").join([stemmer.stem(z) for z in query.split(" ")])
        
        title=re.sub("[^a-zA-Z0-9]"," ", title)
        title= (" ").join([stemmer.stem(z) for z in title.split(" ")])
        
	nquery = len(query.split())
	ntitle = len(title.split())
	
	Xall_new[i,0] = nquery
	Xall_new[i,1] = ntitle
	Xall_new[i,2] = nquery / float(ntitle)
	
	s = difflib.SequenceMatcher(None,a=query,b=title).ratio()
	
	Xall_new[i,3] = s

	nmatches = 0
	for qword in query.split():
	    if qword in title:
		
		nmatches+=1
	nmatches = nmatches / float(nquery)
		
	Xall_new[i,4] = nmatches
	
	if i%5000==0:
	  print "i:",i
	
	if verbose:
	  print query
	  print nquery
	  print title
	  print ntitle
	  print "ratio:",Xall_new[i,2]
	  print "difflib ratio:",s
	  print "matches:",nmatches
	  raw_input()
	
    Xall_new = pd.DataFrame(Xall_new,columns=['query_length','title_length','length_ratio','difflibratio','simplematch']) 
    return Xall_new	
    

def genSimFeatures(Xall,verbose=True):
    print "Compute gensim features..."
    raw_input()
    

  
def useBenchmarkMethod_mod(X,verbose=False):  
     X = X.apply(lambda x:'q%s z%s' % (x['query'],x['product_title']),axis=1)
     return X
     
def useBenchmarkMethod(X,returnList=True,verbose=False):
    print "Create benchmark features..."
    X = X.fillna("")
    stemmer = PorterStemmer()
    s_data=[]
    for i in range(X.shape[0]):	
        s=(" ").join(["q"+ z for z in BeautifulSoup(X["query"].iloc[i]).get_text(" ").split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(X["product_title"].iloc[i]).get_text(" ").split(" ")]) + " " + BeautifulSoup(X["product_description"].iloc[i]).get_text(" ")      
        s=re.sub("[^a-zA-Z0-9]"," ", s)
        s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
        s_data.append(s)
        #s_labels.append(str(X["median_relevance"][i]))
        if verbose:
	  print "i:",i
	  print "query:",X["query"].iloc[i]
	  print "bs:",BeautifulSoup(X["query"].iloc[i]).get_text(" ")
	  print "title:",X["product_title"].iloc[i]
	  print "bs:",BeautifulSoup(X["product_title"].iloc[i]).get_text(" ")
	  print s
	  raw_input()

    if returnList:
      X = s_data
      X = pd.DataFrame(X,columns=['concate_all']) 
    else:
      X = np.asarray(s_data)
      X = X.reshape((X.shape[0],-1))
      #X = pd.DataFrame(X,columns=['concate_all']) 
    
    print "Finished.."
    print X
    #print type(X[0])
    
    return X
    
    
    