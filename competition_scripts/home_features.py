#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NLP features chrisslz

Recycled from crowd flower competition
"""

from qsprLib import *
from nltk.stem.porter import PorterStemmer

import re
import difflib
from scipy.spatial.distance import cdist

import itertools
import math

from nltk.corpus import wordnet as wn

stemmer = PorterStemmer() # faster

#TODO: right after tfidf, input 2 sparse matrics: def computeSimilarityFeatures(Xs_all,Xs_all_new)
#and for dense def computeSimilarityFeatures(Xall,Xall_new,nsplit)
#http://stackoverflow.com/questions/16597265/appending-to-an-empty-data-frame-in-pandas


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

def computeSimilarityFeatures(Xall,columns=['query','product_title'],verbose=False,useOnlyTrain=False,startidx=0,stop_words=None,doSVD=261,vectorizer=None):
    print "Compute scipy similarity..."
    if vectorizer is None:
        vectorizer = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode', analyzer='word',ngram_range=(1, 5), use_idf=True,smooth_idf=True,sublinear_tf=True,stop_words = stop_words,token_pattern=r'\w{1,}')
    print vectorizer
    if useOnlyTrain:
        print "Using train only for TFIDF..."
        Xtrain = Xall[startidx:]
        Xs1 = vectorizer.fit(Xtrain[columns[0]])
    else:
        Xs1 = vectorizer.fit(Xall[columns[0]])

    Xs1 = vectorizer.transform(Xall[columns[0]])
    Xs2 = vectorizer.transform(Xall[columns[1]])
    sparse=True
    if doSVD is not None:
        print "Similiarity with SVD, n_components:",doSVD
        reducer=TruncatedSVD(n_components=doSVD, algorithm='randomized', n_iter=5)
        Xs1 = reducer.fit_transform(Xs1)
        print "Variance explaind:",np.sum(reducer.explained_variance_ratio_)
        Xs2 = reducer.transform(Xs2)
        sparse=False

    sim = computeScipySimilarity(Xs1,Xs2,sparse=sparse)
    return sim

def computeScipySimilarity(Xs1,Xs2,sparse=False):
    Xall_new = np.zeros((Xs1.shape[0],4))

    if sparse:
        print Xs1.shape
        print Xs2.shape
        Xs1 = np.asarray(Xs1.todense())
        Xs2 = np.asarray(Xs2.todense())

    for i,(a,b) in enumerate(zip(Xs1,Xs2)):
        a = a.reshape(-1,a.shape[0])
        b = b.reshape(-1,b.shape[0])
        #print a.shape
        #print type(a)
        dist = cdist(a,b,'cosine')
        Xall_new[i,0] = dist
        #Xall_new[i,3] = dist
        dist = cdist(a,b,'cityblock')
        Xall_new[i,1] = dist
        dist = cdist(a,b,'hamming')
        Xall_new[i,2] = dist
        dist = cdist(a,b,'euclidean')
        Xall_new[i,3] = dist

    Xall_new = pd.DataFrame(Xall_new,columns=['cosine','cityblock','hamming','euclidean'])

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

def additionalFeatures(Xall,verbose=False,dropList=['bestmatch']):
    print "Computing additional features..."
    stemmer = PorterStemmer()
    Xall_new = np.zeros((Xall.shape[0],13))
    for i in range(Xall.shape[0]):
        query = Xall["search_term"].iloc[i].lower()
        title = Xall["product_title"].iloc[i].lower()
        desc = Xall["product_description"].iloc[i].lower()

        #here we should get similars...
        similar_words = [getSynonyms(q,stemmer) for q in query.split()]
        similar_words = set(itertools.chain(*similar_words))

        #is it necessary???
        query=re.sub("[^a-zA-Z0-9]"," ", query)
        query= (" ").join([stemmer.stem(z) for z in query.split()])

        title=re.sub("[^a-zA-Z0-9]"," ", title)
        title= (" ").join([stemmer.stem(z) for z in title.split()])

        desc=re.sub("[^a-zA-Z0-9]"," ", desc)
        desc= (" ").join([stemmer.stem(z) for z in desc.split()])

        #start here
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

        #find number of matches, split in first and last word of serach term
        for qword in query.split():
            # in case we have a perfect match
            if qword in title:
                nmatches+=1
                avgsim = avgsim + 1.0
                if qword == query.split()[-1]:
                    lastsim+=1
                if qword == query.split()[0]:
                    firstsim+=1
            # otherwise get string similarity
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


def cleanse_text(s):
    """
    https://www.kaggle.com/vabatista/home-depot-product-search-relevance/test-script-1/code
    """

    if isinstance(s, str) or isinstance(s, unicode):
        #print "before:",s
        s = s.lower()
        s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s) ##'desgruda' palavras que estão juntas

        s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)

        s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)

        s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)

        s = s.replace(" x "," xby ")
        s = s.replace("*"," xby ")
        s = s.replace(" by "," xby")
        s = s.replace("x0"," xby 0")
        s = s.replace("x1"," xby 1")
        s = s.replace("x2"," xby 2")
        s = s.replace("x3"," xby 3")
        s = s.replace("x4"," xby 4")
        s = s.replace("x5"," xby 5")
        s = s.replace("x6"," xby 6")
        s = s.replace("x7"," xby 7")
        s = s.replace("x8"," xby 8")
        s = s.replace("x9"," xby 9")
        s = s.replace("0x","0 xby ")
        s = s.replace("1x","1 xby ")
        s = s.replace("2x","2 xby ")
        s = s.replace("3x","3 xby ")
        s = s.replace("4x","4 xby ")
        s = s.replace("5x","5 xby ")
        s = s.replace("6x","6 xby ")
        s = s.replace("7x","7 xby ")
        s = s.replace("8x","8 xby ")
        s = s.replace("9x","9 xby ")

        s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)

        s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)

        s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)

        s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)

        s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)

        s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)

        s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)

        s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)

        s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)

        s = s.replace("whirpool","whirlpool")
        s = s.replace("whirlpoolga", "whirlpool")
        s = s.replace("whirlpoolstainless","whirlpool stainless")

        s = s.replace("  "," ")
        s = (" ").join([stemmer.stem(z) for z in s.split(" ")])
        #print "after:",s
        #raw_input()
        return s.lower()
    else:
        return "null"

def str_stemmer(s):
    liste = [stemmer.stem(word) for word in s.lower().split()]
    liste = " ".join(liste)
    return(liste)


def str_common_word(str1, str2):
	return sum(int(str2.find(word)>=0) for word in str1.split())


def createCommonWords(Xall):
        """
        See https://www.kaggle.com/wenxuanchen/home-depot-product-search-relevance/sklearn-random-forest/code
        also: https://www.kaggle.com/the1owl/home-depot-product-search-relevance/rf-mean-squared-error/code
        """
        print "Create common words..."
        Xall['len_of_query'] = Xall['search_term'].map(lambda x:len(x.split())).astype(np.int64)
        Xall['len_of_title'] = Xall['product_title'].map(lambda x:len(x.split())).astype(np.int64)
        Xall['len_of_description'] = Xall['product_description'].map(lambda x:len(x.split())).astype(np.int64)

        Xall['product_info'] = Xall['search_term']+"\t"+Xall['product_title']+"\t"+Xall['product_description']

        Xall['word_in_title'] = Xall['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
        Xall['word_in_description'] = Xall['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))

        print Xall.head(10)
        return Xall