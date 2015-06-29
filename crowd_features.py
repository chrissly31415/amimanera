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

import itertools
import math

import gensim
from nltk.corpus import wordnet as wn
from nltk.corpus import brown
from nltk import Text

from kaggle_distance import *
import pickle

#TODO: right after tfidf, input 2 sparse matrics: def computeSimilarityFeatures(Xs_all,Xs_all_new)
#and for dense def computeSimilarityFeatures(Xall,Xall_new,nsplit)
#http://stackoverflow.com/questions/16597265/appending-to-an-empty-data-frame-in-pandas

 
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
    Xall_new = np.zeros((Xs1.shape[0],4))
    for i,(a,b) in enumerate(zip(Xs1,Xs2)):
	dist = cdist(a,b,'cosine')
	Xall_new[i,0] = dist	
	dist = cdist(a,b,'euclidean')
	Xall_new[i,1] = dist
	dist = cdist(a,b,'hamming')
	Xall_new[i,2] = dist
	#dist = cdist(a,b,'minkowski')
	#Xall_new[i,3] = dist
	dist = cdist(a,b,'cityblock')
	Xall_new[i,3] = dist
	#dist = cdist(a,b,'correlation')
	#Xall_new[i,5] = dist
	#dist = cdist(a,b,'jaccard')
	#Xall_new[i,6] = dist
	
    Xall_new = pd.DataFrame(Xall_new,columns=['cosine','euclidean','hammming','cityblock'])
    print "NA:",Xall_new.isnull().values.sum()
    Xall_new = Xall_new.fillna(0.0)
    print "NA:",Xall_new.isnull().values.sum()
    print Xall_new.corr(method='spearman')
    return Xall_new


def getSynonyms(word,stemmer):
    #synonyms=[word]
    try:
      synonyms = [l.lemma_names() for l in wn.synsets(word)]
    except:
      pass 
    synonyms.append([word])
    synonyms = list(itertools.chain(*synonyms))
    synonyms = [stemmer.stem(l.lower()) for l in synonyms]
    synonyms = set(synonyms)
    return(synonyms)

def makeQuerySynonyms(Xall):
    print "Creating synonyma for query..."
    stemmer = PorterStemmer()
    for i in range(Xall.shape[0]):
	query = Xall["query"].iloc[i].lower()
	qsynonyms = []
	for word in query.split():
	    #print "word:",word
	    qsynonyms.extend(getSynonyms(word,stemmer))
	    
	qsynonyms = (" ").join(z.replace("_"," ") for z in qsynonyms)
	#print qsynonyms
	Xall["query"].iloc[i]=qsynonyms
	#raw_input()
	if i%5000==0:
	  print "i:",i
    return Xall


def information_entropy(text):
    log2=lambda x:math.log(x)/math.log(2)
    exr={}
    infoc=0
    for each in text:
        try:
            exr[each]+=1
        except:
            exr[each]=1
    textlen=len(text)
    for k,v in exr.items():
        freq  =  1.0*v/textlen
        infoc+=freq*log2(freq)
    infoc*=-1
    return infoc

#query id via label_encoder
#max similarity with difflib
#use kaggle distance??
#closed distance 
#
#text.similar('woman')
def additionalFeatures(Xall,verbose=False,dropList=['bestmatch']):
    #dropList=['bestmatch','S_title','S_query','checksynonyma']
    print "Computing additional features..."
    text = Text(word.lower() for word in brown.words())
    stemmer = PorterStemmer()
    Xall_new = np.zeros((Xall.shape[0],13))
    for i in range(Xall.shape[0]):
	query = Xall["query"].iloc[i].lower()
	title = Xall["product_title"].iloc[i].lower()
	desc = Xall["product_description"].iloc[i].lower()
	
	#here we should get similars...
	similar_words = [getSynonyms(q,stemmer) for q in query.split()]
	similar_words = set(itertools.chain(*similar_words))
	
	query=re.sub("[^a-zA-Z0-9]"," ", query)
	query= (" ").join([stemmer.stem(z) for z in query.split()])
        
        title=re.sub("[^a-zA-Z0-9]"," ", title)
        title= (" ").join([stemmer.stem(z) for z in title.split()])
        
        desc=re.sub("[^a-zA-Z0-9]"," ", desc)
        desc= (" ").join([stemmer.stem(z) for z in desc.split()])
        
        nquery = len(query.split())
	ntitle = len(title.split())
	ndesc = len(desc.split())
	
	Xall_new[i,0] = nquery
	Xall_new[i,1] = ntitle
	Xall_new[i,2] = nquery / float(ntitle)
	Xall_new[i,3] = ndesc+1
	Xall_new[i,4] = nquery / float(ndesc+1)
	
	s = difflib.SequenceMatcher(None,a=query,b=title).ratio()
	
	Xall_new[i,5] = s

	nmatches = 0
	avgsim = 0.0
	lastsim = 0.0
	firstsim = 0.0
	checksynonyma = 0.0
	
	for qword in query.split():
	    if qword in title:
		nmatches+=1
		avgsim = avgsim + 1.0
		if qword == query.split()[-1]:
		  lastsim+=1
		if qword == query.split()[0]:
		  firstsim+=1
		
	    else:
	      bestvalue=0.0
	      for tword in title.split():
		s = difflib.SequenceMatcher(None,a=qword,b=tword).ratio()
		if s>bestvalue:
		    bestvalue=s
	      avgsim = avgsim + bestvalue
	      if qword == query.split()[-1]:
		  lastsim = lastsim + bestvalue
	      if qword == query.split()[0]:
		  firstsim = firstsim + bestvalue
	
	    #check similar
	    #print "qword:",qword
	    
	    #if similar_words is not None:
	      for simword in similar_words:
		  if simword in title:
		      checksynonyma+=1		  
		
	Xall_new[i,6] = nmatches / float(nquery)	
	Xall_new[i,7] = avgsim / float(nquery)	
	Xall_new[i,8] = information_entropy(query)
	Xall_new[i,9] = information_entropy(title)
	Xall_new[i,10] = lastsim
	Xall_new[i,11] = firstsim
	Xall_new[i,12] = checksynonyma / float(nquery)
	
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
	
    Xall_new = pd.DataFrame(Xall_new,columns=['query_length','title_length','query_title_ratio','desc_length','query_desc_ratio','difflibratio','bestmatch','averagematch','S_query','S_title','last_sim','first_sim','checksynonyma',]) 
    Xall_new = Xall_new.drop(dropList, axis=1)
    print Xall_new.corr(method='spearman')
    return Xall_new	
    
#make top 5 most similar in query and check again...
def genSimFeatures(Xall,verbose=True):
    print "Compute gensim features..."
    b = gensim.models.Word2Vec(brown.sents())
    #model = gensim.models.Word2Vec.load_word2vec_format('/home/chris/Downloads/GoogleNews-vectors-negative300.bin.gz', binary=True)
    print b.most_similar('money', topn=5)
    #print b.most_similar('playstation', topn=5)
    
    raw_input()
    
def createKaggleDist(Xall,verbose=True):
    print "Kaggle distance..."
    #dic = index_corpus()
    #with open("dic2.pkl", "w") as f: pickle.dump(dic, f) #dic2 encoded without median relevance
    with open("dic2.pkl", "r") as f: dic = pickle.load(f)
    #print "nkd:",nkd('apple','iphone',d)
    #print "nkd:",nkd('apple','peach',d)    
    stemmer = PorterStemmer()
    Xall_new = np.zeros((Xall.shape[0],1))
    for i in range(Xall.shape[0]):
	query = Xall["query"].iloc[i].lower()
	title = Xall["product_title"].iloc[i].lower()
	title=re.sub("[^a-zA-Z0-9]"," ", title)
	nquery = len(query.split())
	
	#topics = ["notebook","computer","movies","clothes","media","shoes","kitchen","car","bike","toys","phone","food"]
	topics = title.split()
	#desc = Xall["product_description"].iloc[i].lower()
	dist_total = 0.0
	for qword in query.split():	      
	      if not qword in topics:
		bestvalue=2.0
		for tword in topics:
		    print "qword:",qword
		    print "tword:",tword
		    dist = nkd(qword,tword,dic)
		    print "nkd:",dist
		    if dist<bestvalue:
		      bestvalue=dist
		dist_total += bestvalue
	      print "nkd-best:",dist_total
	      print "nkd_total",dist_total
	      
	Xall_new[i,0] = dist_total / float(nquery)	
       
    Xall_new = pd.DataFrame(Xall_new,columns=['avg_nkd'])
    print Xall_new.describe()
    #topics = ["Laptop","Children","Movies"]
    
    #print topic_modeling(dic,topics)
    #raw_input()
    
    print "finished"
    return Xall_new
    
  
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
    
    
    