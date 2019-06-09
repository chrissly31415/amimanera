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


import csv
from collections import defaultdict
import math
import json


#TODO: right after tfidf, input 2 sparse matrics: def computeSimilarityFeatures(Xs_all,Xs_all_new)
#and for dense def computeSimilarityFeatures(Xall,Xall_new,nsplit)
#http://stackoverflow.com/questions/16597265/appending-to-an-empty-data-frame-in-pandas


def computeSimilarityFeatures(Xall,columns=['query','product_title'],verbose=False,useOnlyTrain=False,startidx=0,stop_words=None,doSVD=261):
    print("Compute scipy similarity...")
    vectorizer = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode', analyzer='word',ngram_range=(1, 5), use_idf=True,smooth_idf=True,sublinear_tf=True,stop_words = stop_words,token_pattern=r'\w{1,}')
    if useOnlyTrain:
        print("Using train only for TFIDF...")
        Xtrain = Xall[startidx:]
        Xs1 = vectorizer.fit(Xtrain[columns[0]])
    else:
        Xs1 = vectorizer.fit(Xall[columns[0]])

    Xs1 = vectorizer.transform(Xall[columns[0]])
    Xs2 = vectorizer.transform(Xall[columns[1]])
    #print "Xs1",Xs1.shape
    #print "Xs2",Xs2.shape
    #print Xall['query'].iloc[:5]
    #print Xall['product_title'].iloc[:5]
    sparse=True
    if doSVD is not None:
        print("Similiarity with SVD, n_components:",doSVD)
        reducer=TruncatedSVD(n_components=doSVD, algorithm='randomized', n_iter=5, tol=0.0)
        Xs1 = reducer.fit_transform(Xs1)
        Xs2 = reducer.transform(Xs2)
        sparse=False

    sim = computeScipySimilarity(Xs1,Xs2,sparse=sparse)
    return sim

def computeScipySimilarity(Xs1,Xs2,sparse=False):
    if sparse:
        Xs1 = np.asarray(Xs1.todense())
        Xs2 = np.asarray(Xs2.todense())
    Xall_new = np.zeros((Xs1.shape[0],2))
    for i,(a,b) in enumerate(zip(Xs1,Xs2)):
        a = a.reshape(-1,a.shape[0])
        b = b.reshape(-1,b.shape[0])
        #print a.shape
        #print type(a)
        dist = cdist(a,b,'cosine')
        #print dist
        #print type(dist)

        Xall_new[i,0] = dist

        #dist = cdist(a,b,'minkowski')
        #Xall_new[i,3] = dist
        dist = cdist(a,b,'cityblock')
        Xall_new[i,1] = dist
        #dist = cdist(a,b,'hamming')
        #Xall_new[i,2] = dist
        #dist = cdist(a,b,'euclidean')
        #Xall_new[i,3] = dist
        #dist = cdist(a,b,'correlation')
        #Xall_new[i,5] = dist
        #dist = cdist(a,b,'jaccard')
        #Xall_new[i,3] = dist

    Xall_new = pd.DataFrame(Xall_new,columns=['cosine','cityblock'])
    print("NA:",Xall_new.isnull().values.sum())
    Xall_new = Xall_new.fillna(0.0)
    print("NA:",Xall_new.isnull().values.sum())
    print(Xall_new.corr(method='spearman'))
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

def makeQuerySynonyms(Xall,construct_map=False):
    query_map={}
    if construct_map:
        print("Creating synonyma for query...")
        model = gensim.models.Word2Vec.load_word2vec_format('/home/loschen/Downloads/GoogleNews-vectors-negative300.bin.gz', binary=True)
        X_temp = Xall.drop_duplicates('query')
        print(X_temp.describe())

        for i in range(X_temp.shape[0]):
            query = X_temp["query"].iloc[i].lower()
            qsynonyms = query.split()
            for word in query.split():
                #print "word:",word
                try:
                    s = model.most_similar(word, topn=3)
                    qlist = []
                    for item,sim in s:
                        if sim>0.6:
                            qlist.append(item.lower())

                    #print "word: %s synonyma: %r"%(word,qlist)
                    qsynonyms = qsynonyms+qlist
                except:
                    pass

            #print qsynonyms
            qsynonyms = (" ").join(z.replace("_"," ") for z in qsynonyms)
            #print qsynonyms
            #Xall["query"].iloc[i]=qsynonyms
            query_map[query]=qsynonyms
            #raw_input()
            if i%20==0:
                print("i:",i)

        with open("w2v_querymap.pkl", "w") as f: pickle.dump(query_map, f)

    with open("w2v_querymap.pkl", "r") as f: query_map = pickle.load(f)


    print("Mapping synonyma to query...")
    for i in range(Xall.shape[0]):
        query = Xall["query"].iloc[i].lower()
        Xall["query"].iloc[i]=query_map[query]
        if i%5000==0:
            print("i:",i)

    print(Xall['query'].iloc[:10])
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
    for k,v in list(exr.items()):
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
    print("Computing additional features...")
    #text = Text(word.lower() for word in brown.words())
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
            print("i:",i)

        if verbose:
            print(query)
            print(nquery)
            print(title)
            print(ntitle)
            print("ratio:",Xall_new[i,2])
            print("difflib ratio:",s)
            print("matches:",nmatches)
            input()

    Xall_new = pd.DataFrame(Xall_new,columns=['query_length','title_length','query_title_ratio','desc_length','query_desc_ratio','difflibratio','bestmatch','averagematch','S_query','S_title','last_sim','first_sim','checksynonyma',])
    Xall_new = Xall_new.drop(dropList, axis=1)
    print(Xall_new.corr(method='spearman'))
    return Xall_new

def cleanse_data(Xall):
    print("Cleansing the data...")
    with open("query_map.pkl", "r") as f: query_map = pickle.load(f)#key query value corrected value

    ablist=[]
    ablist.append((['ps','p.s.','play station','ps2','ps3','ps4'],'playstation'))
    ablist.append((['ny','n.y.'],'new york'))
    ablist.append((['tb','tera byte'],'gigabyte'))
    ablist.append((['gb','giga byte'],'gigabyte'))
    ablist.append((['t.v.','tv'],'television'))
    ablist.append((['mb','mega byte'],'megabyte'))
    ablist.append((['d.r.','dr'],'doctor'))
    ablist.append((['phillips'],'philips'))

    for i in range(Xall.shape[0]):
        query = Xall["query"].iloc[i].lower()

        #correct typos
        if query in list(query_map.keys()):
            query = query_map[query]

        #correct abbreviations query
        new_query =[]
        for qword in query.split():
            for ab,full in ablist:
                if qword in ab:
                    qword = full
            new_query.append(qword)

        new_query = (" ").join(new_query)
        Xall["query"].iloc[i] = new_query

        title = Xall["product_title"].iloc[i].lower()

        #correct abbreviations title
        new_title=[]
        for qword in title.split():
            for ab,full in ablist:
                if qword in ab:
                    qword = full
            new_title.append(qword)
        new_title = (" ").join(new_title)
        Xall["product_title"].iloc[i] = new_title


        if i%5000==0:
            print("i:",i)

    print("Finished")
    return Xall


#make top 5 most similar in query and check again...
def genWord2VecFeatures(Xall,verbose=True,dropList=[]):
    print("Compute word2vec features...")
    #print Xall['query'].tolist()
    #print brown.sents()
    #b = gensim.models.Word2Vec(brown.sents())
    model = gensim.models.Word2Vec.load_word2vec_format('/home/loschen/Downloads/GoogleNews-vectors-negative300.bin.gz', binary=True)
    Xall_new = np.zeros((Xall.shape[0],5))
    for i in range(Xall.shape[0]):
        query = Xall["query"].iloc[i].lower()
        title = Xall["product_title"].iloc[i].lower()

        query=re.sub("[^a-zA-Z0-9]"," ", query)
        nquery = len(query.split())

        title=re.sub("[^a-zA-Z0-9]"," ", title)
        ntitle = len(title.split())

        bestsim = 0.0
        lastsim = 0.0
        firstsim = 0.0
        avgsim = 0.0

        #print "Query:",query
        #print "Title:",title
        for qword in query.split():
            if qword in title:
                bestsim = bestsim + 1.0
                avgsim = avgsim +1.0
                if qword == query.split()[-1]:
                    lastsim+=1
                if qword == query.split()[0]:
                    firstsim+=1
            else:
                bestvalue=0.0
                for tword in title.split():
                    try:
                        s = model.similarity(qword,tword)
                        #print "query: %s title: %s  sim: %4.2f"%(qword,tword,s)
                        #print model.most_similar(qword, topn=5)
                        #print model.most_similar(tword, topn=5)
                    except:
                        s = 0.0
                    avgsim = avgsim + s
                    if s>bestvalue:
                        bestvalue=s

                bestsim = bestsim + bestvalue
                #print "bestvalue: %4.2f avg: %4.2f"%(bestvalue,avgsim)

                if qword == query.split()[-1]:
                    lastsim = bestvalue
                if qword == query.split()[0]:
                    firstsim = bestvalue

        if i%5000==0:
            print("i:",i)

        Xall_new[i,0] = bestsim / float(nquery)
        Xall_new[i,1] = lastsim
        Xall_new[i,2] = firstsim
        Xall_new[i,3] = avgsim / float(ntitle)
        Xall_new[i,4] = avgsim

        #raw_input()
    Xall_new = pd.DataFrame(Xall_new,columns=['w2v_bestsim','w2v_lastsim','w2v_firstsim','w2v_avgsim','w2v_totalsim'])
    Xall_new = Xall_new.drop(dropList, axis=1)
    print(Xall_new.corr(method='spearman'))
    return Xall_new

def createKaggleDist(Xall,general_topics=["notebook","computer","movie","clothes","media","shoe","kitchen","car","bike","toy","phone","food","sport"], verbose=True):
    print("Kaggle distance...")
    #dic = index_corpus()
    #with open("dic2.pkl", "w") as f: pickle.dump(dic, f) #dic2 encoded without median relevance
    #with open("dic3.pkl", "w") as f: pickle.dump(dic, f) #only train dic2 encoded without median relevance
    with open("dic3.pkl", "r") as f: dic = pickle.load(f)
    #print "nkd:",nkd('apple','iphone',d)
    #print "nkd:",nkd('apple','peach',d)
    # = ["notebook","computer","movie","clothes","media","shoe","kitchen","car","bike","toy","phone","food","sport"]
    stemmer = PorterStemmer()

    if general_topics is None:
        n = 1
    else:
        n = len(general_topics)+1

    Xall_new = np.zeros((Xall.shape[0],n))
    for i in range(Xall.shape[0]):
        query = Xall["query"].iloc[i].lower()
        title = Xall["product_title"].iloc[i].lower()
        title=re.sub("[^a-zA-Z0-9]"," ", title)
        nquery = len(query.split())

        topics = title.split()

        #print "query:",query
        #print "title:",title
        dist_total = 0.0
        for qword in query.split():
            #print "qword:",qword
            if not qword in topics:
                bestvalue=2.0
                for tword in topics:
                    #print "qword:",qword
                    #print "tword:",tword
                    dist = nkd(qword,tword,dic)
                    #print "nkd:",dist
                    if dist<bestvalue:
                        bestvalue=dist
                dist_total += bestvalue
            #print "nkd-best:",dist_total
            #print "nkd_total",dist_total
            if general_topics is not None:
                for j,topic in enumerate(general_topics):
                    dist = nkd(qword,topic,dic)
                    Xall_new[i,1+j] = Xall_new[i,1+j] + dist/float(nquery)
                    #print "qword:%s topic:%s nkd:%4.2f nkd-avg: %4.2f"%(qword,topic,dist,Xall_new[i,1+j])
                #raw_input()

        Xall_new[i,0] = dist_total / float(nquery)

    if general_topics is None:
        Xall_new = pd.DataFrame(Xall_new,columns=['avg_nkd'])
    else:
        Xall_new = pd.DataFrame(Xall_new,columns=['avg_nkd']+general_topics)
    print(Xall_new.describe())
    #print topic_modeling(dic,topics)
    #raw_input()

    print("finished")
    return Xall_new


def useBenchmarkMethod(X,returnList=True,verbose=False):
    print("Create benchmark features...")
    X = X.fillna("")
    stemmer = PorterStemmer()
    s_data=[]
    for i in range(X.shape[0]):
        s=(" ").join(["q"+ z for z in BeautifulSoup(X["query"].iloc[i]).get_text(" ").split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(X["product_title"].iloc[i]).get_text(" ").split(" ")]) + " " + BeautifulSoup(X["product_description"].iloc[i]).get_text(" ")
        s=re.sub("[^a-zA-Z0-9]"," ", s)
        s= (" ").join([stemmer.stem(z) for z in s.split()])
        s_data.append(s.lower())
        if i%5000==0:
            print("i:",i)
    if returnList:
        X = s_data
        X = pd.DataFrame(X,columns=['query'])
    else:
        X = np.asarray(s_data)
        X = X.reshape((X.shape[0],-1))
        X = pd.DataFrame(X,columns=['concate_all'])

    print("Finished..")
    #print X
    #print type(X[0])

    return X

# Use Pandas to read in the training and test data
#train = pd.read_csv("../input/train.csv").fillna("")
#test  = pd.read_csv("../input/test.csv").fillna("")

def build_query_correction_map(print_different=True):
    train = pd.read_csv('./data/train.csv').fillna("")
    test  = pd.read_csv("./data/test.csv").fillna("")
    # get all query
    queries = set(train['query'].values)
    correct_map = {}
    if print_different:
        print(("%30s \t %30s"%('original query','corrected query')))
    for q in queries:
        corrected_q = autocorrect_query(q,train=train,test=test,warning_on=False)
        if print_different and q != corrected_q:
            print(("%30s \t %30s"%(q,corrected_q)))
        correct_map[q] = corrected_q
    return correct_map

def autocorrect_query(query,train=None,test=None,cutoff=0.8,warning_on=True):
    """
    autocorrect a query based on the training set
    """
    if train is None:
        train = pd.read_csv('./data/train.csv').fillna('')
    if test is None:
        test = pd.read_csv('./data/test.csv').fillna('')
    train_data = train.values[train['query'].values==query,:]
    test_data = test.values[test['query'].values==query,:]
    s = ""
    for r in train_data:
        #print "----->r2:",r[2]
        #print "r3:",r[3]

        s = "%s %s %s"%(s,BeautifulSoup(r[2]).get_text(" ",strip=True),BeautifulSoup(r[3]).get_text(" ",strip=True))
        #print "s:",s
        #raw_input()
    for r in test_data:
        s = "%s %s %s"%(s,BeautifulSoup(r[2]).get_text(" ",strip=True),BeautifulSoup(r[3]).get_text(" ",strip=True))
    s = re.findall(r'[\'\"\w]+',s.lower())
    #print s
    s_bigram = [' '.join(i) for i in bigrams(s)]
    #print s_bigram
    #raw_input()
    s.extend(s_bigram)
    corrected_query = []
    for q in query.lower().split():
        #print "q:",q
        if len(q)<=2:
            corrected_query.append(q)
            continue
        corrected_word = difflib.get_close_matches(q, s,n=1,cutoff=cutoff)
        #print "correction:",corrected_word
        if len(corrected_word) >0:
            corrected_query.append(corrected_word[0])
        else :
            if warning_on:
                print(("WARNING: cannot find matched word for '%s' -> used the original word"%(q)))
            corrected_query.append(q)
        #print "corrected_query:",corrected_query
        #raw_input()
    return ' '.join(corrected_query)


def autocorrect():
    query_map = build_query_correction_map()
    with open("query_map.pkl", "w") as f: pickle.dump(query_map, f)

"""
	File: kagge_distance.py

	__author__: 	 Triskelion
	__description__: Normalized Kaggle Distance for visualization, topic modeling and semantic 
					 knowledge base creation.
	
					 Based on Normalized Google Distance, which in turn is based on Normalized 
					 Compression Distance, which in turn is based on Information Distance, 
					 which in turn is based on Kolmogorov Complexity.
					 
					 A bit messy to mix HTML with Python, but that's how it goes.
					 
					 Ref: http://homepages.cwi.nl/~paulv/papers/crc08.pdf
"""

def clean(s):
	# Returns unique token-sorted cleaned lowercased text
	return " ".join(sorted(set(re.findall(r'\w+', s,flags = re.UNICODE | re.LOCALE)))).lower()

def index_document(s,d):
	# Creates half the matrix of pairwise tokens
	# This fits into memory, else we have to choose a Count-min Sketch probabilistic counter 
	tokens = s.split()
	for x in range(len(tokens)):
		d[tokens[x]] += 1
		for y in range(x+1,len(tokens)):
			d[tokens[x]+"_X_"+tokens[y]] += 1
	return d

def index_corpus():	
	# Create our count dictionary and fill it with train and test set (pairwise) token counts	
	d = defaultdict(int)
	#for e, row in enumerate( csv.DictReader(open("./data/train.csv",'r', newline='', encoding='utf8'))):
	for e, row in enumerate( csv.DictReader(open("./data/train.csv",'r'))):
		#s = clean("medianrellabel%s %s %s"%(row["median_relevance"], row["product_description"],row["product_title"]))
		s = clean("%s %s"%(row["product_description"],row["product_title"]))
		d = index_document(s,d)
	#for e, row in enumerate( csv.DictReader(open("./data/test.csv",'r', newline='', encoding='utf8'))):
	for e, row in enumerate( csv.DictReader(open("./data/test.csv",'r'))):
		s = clean("%s %s"%(row["product_description"],row["product_title"]))
		d = index_document(s,d)
	return d	
	
def nkd(token1, token2, d):
	# Returns the semantic Normalized Kaggle Distance between two tokens
	sorted_tokens = sorted([clean(token1), clean(token2)])
	token_x = sorted_tokens[0]
	token_y = sorted_tokens[1]
	if d[token_x] == 0 or d[token_y] == 0 or d[token_x+"_X_"+token_y] == 0:
		return 2.
	else:
		#print d[token_x], d[token_y], d[token_x+"_X_"+token_y], token_x+"_X_"+token_y
		logcount_x = math.log(d[token_x])
		logcount_y = math.log(d[token_y])
		logcount_xy = math.log(d[token_x+"_X_"+token_y])
		log_index_size = math.log(100000) # fixed guesstimate
		nkd = (max(logcount_x,logcount_y)-logcount_xy) / (log_index_size-min(logcount_x,logcount_y))	
		return nkd

def generate_json_graph(targets,d):
	# From a comma seperated string this creates the JSON to build a force-directed graph in D3.js
	targets = targets.split(",")
	result = defaultdict(list)
	
	for i in range(len(targets)):
		result["nodes"].append({"s": targets[i], "y": d[targets[i]] })
		for j in range(i+1,len(targets)):
			result["links"].append({"source": i, "target":  j, "strength": nkd(targets[i], targets[j], d)})
	return json.dumps(result)	

def multiple_choice(d,question,anchor,choices):
	# Answers a multiple choice question in HTML where 'anchor' is the keyword
	q = """<li class="pane"><h3>%s</h3><ul>%s</ul></li>"""%(question,
			"".join(["<li><span>%s</span>%s</li>"%(round(w[0],3),w[1]) for w in sorted([(nkd(f,anchor,d),f) for f in choices])]))
	return q
	
def topic_modeling(d,labeled_topics):
	# Labeled_topics is a list of topics you want to create
	# Uses only words in train set
	v = {}
	for topic in labeled_topics:
		v[topic] = []
	for e, row in enumerate( csv.DictReader(open("./data/train.csv",'r'))):
		words = clean("%s %s"%(row["product_description"],row["product_title"])).split()
		for word in words:
			for k in v:
				v[k].append( (nkd(word,k,d),word) )
	out = ""
	for k in v:
		out += "<h3>Topic: %s</h3><p>"%k
		l = []
		for t in sorted(set(v[k]))[:25]:
			l.append(t[1])
		out += ", ".join(l)+ "</p>"
	return out
	
def edge_bundeling(d):
    # Create edge bundeling visualization
	# Uses only words in train set	
	w = []
	for e, row in enumerate( csv.DictReader(open("./data/train.csv",'r', newline='', encoding='utf8'))):
		w += clean("%s %s"%(row["product_description"],row["product_title"])).split()
	w = list(set(w))	
	
	# Find 115 words closest to LABEL_4
	c = []
	for word in w:
		c.append((nkd(word,"medianrellabel4",d),word))
	c = sorted(c)[:115]
	
	edge = defaultdict(list)
	# Find 5 closest words for every closest word
	for e, (distance,word) in enumerate(c):
		for anchor_distance, anchor_word in c:
			edge[(e,word)].append((nkd(word,anchor_word,d),anchor_word))
		edge[(e,word)] = [f[1] for f in sorted(edge[(e,word)])[:4]]
	
	# Format json
	result = []
	for k in sorted(edge):
		result.append({"name":k[1],"imports":edge[k]})
	return json.dumps(result)
	
if __name__ == "__main__":
	d = index_corpus()

	manufacturers = ["Amazon","Apple","Google","Microsoft","Motorola"]
	html1 = multiple_choice(d,"Who created the iPhone phone?","iPhone",manufacturers)
	html2 = multiple_choice(d,"Who created the Nexus phone?","Nexus",manufacturers)
	html3 = multiple_choice(d,"Who created the Moto phone?","Moto",manufacturers)
	html4 = multiple_choice(d,"Who created the Fire phone?","Fire",manufacturers)
	
	film_studios = ["Disney","20th","Fox","Paramount","Sony","Colombia","Goldwyn","Universal","Warner"]
	html5 = multiple_choice(d,"Which major film studio produced the Batman films?","Batman",film_studios)
	html6 = multiple_choice(d,"Which major film studio produced the film Frozen?","Frozen",film_studios)
	
	topics = ["Laptop","Children","Movies"]
	html7 = topic_modeling(d,topics)
	
	targets = "one,two,white,green,pc,laptop,desktop,mouse,red"
	json1 = generate_json_graph(targets,d)
	
	json2 = edge_bundeling(d)
	
	with open("output.html","wb") as outfile:
		html = """<!DOCTYPE html>
<html>
	<head>
		<meta charset="utf-8">
		<title>Triskelion &bull; Normalized Kaggle Distance</title>
		<meta name="robots" content="noindex,nofollow,noodp,nonothing,goaway">
		<link href='https://fonts.googleapis.com/css?family=Open+Sans:400,800' rel='stylesheet' type='text/css'>
		<style>
			* { margin:0; padding: 0;}
			body { font-family: "Open Sans",Verdana,sans-serif; margin: 30px auto; width: 700px; }
			p { font-size: 18px; line-height: 27px; padding-bottom: 27px; color: #111; }
			h1 { font-weight: 800; font-size: 33px; color: #2C3E50; padding-bottom: 10px;}
			h2 { font-weight: 800; font-size: 22px; color: #2C3E50; padding-bottom: 10px; }
			h3 { font-weight: 800; font-size: 18px; color: #34495E; padding-bottom: 10px; }
			small { color: #7F8C8D; }
			ul.panes { display: block; overflow: hidden; list-style-type: none; padding: 10px 0px;}
			li.pane { float: left; width: 300px; margin-right: 40px; padding-bottom: 40px; }
			ul { list-style-type: none; }
			span { display: block; width: 50px; padding-right: 15px; text-align: right; height: 20px; float: left; }
			li ul li { color: #7F8C8D; }
			li ul li:first-child { color: #111; }
			.node { font: 400 14px "Open Sans", Verdana, sans-serif; fill: #333; cursor:pointer;}
			.node:hover {fill: #000;}
			.link {stroke: steelblue; stroke-opacity:.4;fill: none; pointer-events: none;}
			.node:hover,.node--source,.node--target { font-weight: 700;}
			.node--source { fill: #2ca02c;}
			.node--target { fill: #d62728;}
			.link--source,.link--target { stroke-opacity: 1; stroke-width: 2px;}
			.link--source { stroke: #d62728;}
			.link--target { stroke: #2ca02c;}
		</style>
	</head>
	<body>
	    <noscript>Uses JavaScript for visualizations (d3.js)</noscript>
		<h1>Normalized Kaggle Distance</h2>
		<p>Normalized Kaggle Distance (NKD) uses a search engine index as a compressor by looking at page count statistics. Semantically related words get a closer distance. This allows us to do all sorts of fun stuff.</p>
		<p>NKD is based on <a href="http://en.wikipedia.org/wiki/Normalized_Google_distance">Normalized Google Distance</a> which uses the Google index. We show that, even with a relatively small corpus, we can still extract useful semantic information.</p>
		
		<h2>Semantic Knowledge Base</h2>
		
		<ul class="panes">
			%s %s %s %s %s %s
		</ul>
		
		<h2>Topic Modeling</h2>
		%s
		
		<h2>Force-Directed Clustering <small>(drag to interact)</small></h2>
		
		<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js"></script>
		<script>
		function draw_cluster_graph(){
			var json = %s;	

			var width = 700,
				height = 600;

			var svg = d3.select("body").append("svg")
				.attr("width", width)
				.attr("height", height);

			var force = d3.layout.force()
						.gravity(.05)
						.distance(800)
						.charge(-800)
						.size([width, height]);

			force
				.nodes(json.nodes)
				.links(json.links)
				.linkDistance(function(d) { return  300 - ((1 - d.strength) * 300); })
				.start();

			var link = svg.selectAll(".link")
						.data(json.links)
						.enter().append("line")
						.attr("class", "link")
						.style("stroke", "#2C3E50")
						.style("stroke-opacity", function(d) { return 0.2 - (d.strength/4); })
						.style("stroke-width", 1)
						  ;
			var node = svg.selectAll(".node")
						.data(json.nodes)
						.enter().append("g")
						.attr("class", "node")
						.call(force.drag);
						  
			node.append("circle")
				.attr("r", function(d) { return (Math.log(d.y) * 5)+3; })
				.style("fill", "#3498DB")
				.style("fill-opacity", 0.23);

			node.append("text")
				.attr("dx", 12)
				.attr("class", "text")
				.attr("dy", ".35em")
				.style("font-family", "Open Sans")
				.style("font-weight", "400")
				.style("color", "#2C3E50")
				.text(function(d) { return d.s });

			force.on("tick", function() {
				link.attr("x1", function(d) { return d.source.x; })
					.attr("y1", function(d) { return d.source.y; })
					.attr("x2", function(d) { return d.target.x; })
					.attr("y2", function(d) { return d.target.y; });

				node.attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; });
			});

		};
		draw_cluster_graph();

		d3.select("body").append("h2").text("Hierarchical edge bundling");
		var diameter = 700,
			radius = diameter / 2,
			innerRadius = radius - 120;

		var cluster = d3.layout.cluster()
			.size([360, innerRadius])
			.sort(null)
			.value(function(d) { return d.size; });

		var bundle = d3.layout.bundle();

		var line = d3.svg.line.radial()
			.interpolate("bundle")
			.tension(.85)
			.radius(function(d) { return d.y; })
			.angle(function(d) { return d.x / 180 * Math.PI; });

		var svg = d3.select("body").append("svg")
			.attr("width", diameter)
			.attr("height", diameter)
		  .append("g")
			.attr("transform", "translate(" + radius + "," + radius + ")");

		var link = svg.append("g").selectAll(".link"),
			node = svg.append("g").selectAll(".node");

		var classes = %s;	

		  var nodes = cluster.nodes(packageHierarchy(classes)),
			  links = packageImports(nodes);

		  link = link
			  .data(bundle(links))
			.enter().append("path")
			  .each(function(d) { d.source = d[0], d.target = d[d.length - 1]; })
			  .attr("class", "link")
			  .attr("d", line);

		  node = node
			  .data(nodes.filter(function(n) { return !n.children; }))
			.enter().append("text")
			  .attr("class", "node")
			  .attr("dy", ".31em")
			  .attr("transform", function(d) { return "rotate(" + (d.x - 90) + ")translate(" + (d.y + 8) + ",0)" + (d.x < 180 ? "" : "rotate(180)"); })
			  .style("text-anchor", function(d) { return d.x < 180 ? "start" : "end"; })
			  .text(function(d) { return d.key; })
			  .on("mouseover", mouseovered)
			  .on("mouseout", mouseouted);


		function mouseovered(d) {
		  node
			  .each(function(n) { n.target = n.source = false; });

		  link
			  .classed("link--target", function(l) { if (l.target === d) return l.source.source = true; })
			  .classed("link--source", function(l) { if (l.source === d) return l.target.target = true; })
			.filter(function(l) { return l.target === d || l.source === d; })
			  .each(function() { this.parentNode.appendChild(this); });

		  node
			  .classed("node--target", function(n) { return n.target; })
			  .classed("node--source", function(n) { return n.source; });
		}

		function mouseouted(d) {
		  link
			  .classed("link--target", false)
			  .classed("link--source", false);

		  node
			  .classed("node--target", false)
			  .classed("node--source", false);
		}

		d3.select(self.frameElement).style("height", diameter + "px");

		// Lazily construct the package hierarchy from class names.
		function packageHierarchy(classes) {
		  var map = {};

		  function find(name, data) {
			var node = map[name], i;
			if (!node) {
			  node = map[name] = data || {name: name, children: []};
			  if (name.length) {
				node.parent = find(name.substring(0, i = name.lastIndexOf(".")));
				node.parent.children.push(node);
				node.key = name.substring(i + 1);
			  }
			}
			return node;
		  }

		  classes.forEach(function(d) {
			find(d.name, d);
		  });

		  return map[""];
		}

		// Return a list of imports for the given array of nodes.
		function packageImports(nodes) {
		  var map = {},
			  imports = [];

		  // Compute a map from name to node.
		  nodes.forEach(function(d) {
			map[d.name] = d;
		  });

		  // For each import, construct a link from the source to target node.
		  nodes.forEach(function(d) {
			if (d.imports) d.imports.forEach(function(i) {
			  imports.push({source: map[d.name], target: map[i]});
			});
		  });

		  return imports;
		}
		
		
		
		</script>
	</body>
</html>"""%(html1,html2,html3,html4,html5,html6,html7,json1,json2)
		outfile.write(html.encode('utf-8'))




