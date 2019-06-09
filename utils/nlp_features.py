#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""  NLP features
"""

from qsprLib import *
from nltk.stem.porter import PorterStemmer
from bs4 import BeautifulSoup
import re
import difflib
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import pairwise_distances

from collections import Counter
import itertools
import math

import gensim
from nltk.corpus import wordnet as wn
from nltk import bigrams
from nltk.corpus import brown
from nltk import Text
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction import text
from nltk import corpus

import pickle



#TODO: right after tfidf, input 2 sparse matrics: def computeSimilarityFeatures(Xs_all,Xs_all_new)
#chwck http://research.microsoft.com/en-us/projects/mslr/feature.aspx
#and for dense def computeSimilarityFeatures(Xall,Xall_new,nsplit)
#2nd place solution home depot: https://www.kaggle.com/c/home-depot-product-search-relevance/forums/t/20427/congrats-to-the-winners
#http://stackoverflow.com/questions/16597265/appending-to-an-empty-data-frame-in-pandas

stemmer = PorterStemmer() # faster

stop_words = text.ENGLISH_STOP_WORDS.union(corpus.stopwords.words('english'))

def str_stemmer(s):
    liste = [stemmer.stem(word) for word in s.lower().split()]
    liste = " ".join(liste)
    return(liste)


def str_common_word(str1, str2):
	return sum(int(str2.find(word)>=0) for word in str1.split())

def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row[0]).lower().split():
        if word not in stop_words:
            q1words[word] = 1.0
    for word in str(row[1]).lower().split():
        if word not in stop_words:
            q2words[word] = 1.0
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in list(q1words.keys()) if w in q2words]
    shared_words_in_q2 = [w for w in list(q2words.keys()) if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/float(len(q1words) + len(q2words))
    return R

# If a word appears only once, we ignore it completely (likely a typo)
# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller
def get_weight(count, eps=10000.0, min_count=2):
    if count < min_count:
        return 0.0
    else:
        return 1.0 / float(count + eps)

def tfidf_word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row[0]).lower().split():
        if word not in stop_words:
            q1words[word] = 1
    for word in str(row[1]).lower().split():
        if word not in stop_words:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0

    shared_weights = [weights.get(w, 0) for w in list(q1words.keys()) if w in q2words] + [weights.get(w, 0) for w in
                                                                                    list(q2words.keys()) if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

    R = np.sum(shared_weights) / float(np.sum(total_weights))
    return R

weights=0
def get_tfidf_share(X):
    global weights
    eps = 5000.0
    train_qs = pd.Series(X['question1'].tolist() + X['question2'].tolist()).astype(str)
    words = (" ".join(train_qs)).lower().split()
    counts = Counter(words)
    weights = {word: get_weight(count) for word, count in list(counts.items())}
    return X.apply(tfidf_word_match_share, axis=1, raw=True)


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

def computeSimilarityFeatures(Xall,columns=['question1','question2'],verbose=False,useOnlyTrain=False,startidx=0,stop_words=None,doSVD=261,vectorizer=None,reducer=None):
    print("Compute scipy similarity...for :",columns)
    if vectorizer is None:
        vectorizer = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode', analyzer='word',ngram_range=(1, 5), use_idf=True,smooth_idf=True,sublinear_tf=True,stop_words = stop_words,token_pattern=r'\w{1,}')
    print(vectorizer)
    if useOnlyTrain:
        print("Using train only for TFIDF...")
        Xtrain = Xall[startidx:]
        Xs1 = vectorizer.fit(Xtrain[columns[0]])
    else:
        Xs1 = vectorizer.fit(Xall[columns[0]])

    Xs1 = vectorizer.transform(Xall[columns[0]])
    Xs2 = vectorizer.transform(Xall[columns[1]])
    sparse=True
    if doSVD is not None:
        print("Similiarity with SVD, n_components:",doSVD)
        if reducer is None:
            reducer=TruncatedSVD(n_components=doSVD, algorithm='randomized', n_iter=5)
        print(reducer)
        Xs1 = reducer.fit_transform(Xs1)
        #print "Variance explaind:",np.sum(reducer.explained_variance_ratio_)
        Xs2 = reducer.transform(Xs2)
        sparse=False

    sim = computeScipySimilarity(Xs1,Xs2,sparse=sparse)
    return sim

def computeScipySimilarity(Xs1,Xs2,sparse=False):
    Xall_new = np.zeros((Xs1.shape[0],4))

    if sparse:
        print(Xs1.shape)
        print(Xs2.shape)
        Xs1 = np.asarray(Xs1.todense())
        Xs2 = np.asarray(Xs2.todense())
        #Xall_new[:,0] = pairwise_distances(Xs1,Xs2,metric='cosine')
        #Xall_new[:,1] = pairwise_distances(Xs1,Xs2,metric='cityblock')
        #Xall_new[:,2] = pairwise_distances(Xs1,Xs2,metric='hamming')
        #Xall_new[:,3] = pairwise_distances(Xs1,Xs2,metric='euclidean')
        #Xall_new = pd.DataFrame(Xall_new,columns=['cosine','cityblock','hamming','euclidean'])
        #print Xall_new.head(30)

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
        #dist = cdist(a,b,'correlation')
        #Xall_new[i,5] = dist
        #dist = cdist(a,b,'jaccard')
        #Xall_new[i,4] = dist
    Xall_new = pd.DataFrame(Xall_new,columns=['cosine','cityblock','hamming','euclidean'])

    print("NA,before:",Xall_new.isnull().values.sum())
    Xall_new = Xall_new.fillna(0.0)
    print("NA,after:",Xall_new.isnull().values.sum())
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


def additionalFeatures(Xall,verbose=False,dropList=['bestmatch']):
    print("Computing additional features...")
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


def additionalFeatures_new(Xall,verbose=False,dropList=['bestmatch']):
    print("Computing additional features new...")
    stemmer = PorterStemmer()
    Xall_new = np.zeros((Xall.shape[0],11))
    for i in range(Xall.shape[0]):
        query = Xall["search_term"].iloc[i].lower()
        title = Xall["product_info"].iloc[i].lower()

        #here we should get similars...
        similar_words = [getSynonyms(q,stemmer) for q in query.split()]
        similar_words = set(itertools.chain(*similar_words))

        #is it necessary???
        query=re.sub("[^a-zA-Z0-9]"," ", query)
        query= (" ").join([stemmer.stem(z) for z in query.split()])

        title=re.sub("[^a-zA-Z0-9]"," ", title)
        title= (" ").join([stemmer.stem(z) for z in title.split()])

        #start here
        nquery = len(query.split())
        ntitle = len(title.split())

        Xall_new[i,0] = nquery
        Xall_new[i,1] = ntitle
        Xall_new[i,2] = nquery / float(ntitle)

        s = difflib.SequenceMatcher(None,a=query,b=title).ratio()

        Xall_new[i,3] = s

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

            #check similar
            #print "qword:",qword

            #if similar_words is not None:
                for simword in similar_words:
                    if simword in title:
                        checksynonyma+=1

        Xall_new[i,4] = nmatches / float(nquery)
        Xall_new[i,5] = avgsim / float(nquery)
        Xall_new[i,6] = information_entropy(query)
        Xall_new[i,7] = information_entropy(title)
        Xall_new[i,8] = lastsim
        Xall_new[i,9] = firstsim
        Xall_new[i,10] = checksynonyma / float(nquery)

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

    Xall_new = pd.DataFrame(Xall_new,columns=['query_length','title_length','query_title_ratio','difflibratio','bestmatch','averagematch','S_query','S_title','last_sim','first_sim','checksynonyma',])
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


def genWord2VecFeatures_new(Xall,verbose=False):
    model = gensim.models.Word2Vec.load_word2vec_format('/home/loschen/Downloads/GoogleNews-vectors-negative300.bin.gz', binary=True)
    for i in range(Xall.shape[0]):
        query = Xall["search_term"].iloc[i].lower()
        title = Xall["product_title"].iloc[i].lower()

        query=re.sub("[^a-zA-Z0-9]"," ", query)
        title=re.sub("[^a-zA-Z0-9]"," ", title)

        for qword in query.split():
            for tword in title.split():
                s = model.similarity(qword,tword)
                print("query: %s title: %s  sim: %4.2f"%(qword,tword,s))
                print(model.most_similar(qword, topn=5))
                print(model.most_similar(tword, topn=5))
                input()


#make top 5 most similar in query and check again...
def genWord2VecFeatures(Xall,verbose=True,dropList=[]):
    print("Compute word2vec features...")
    #print Xall['query'].tolist()
    #b = gensim.models.Word2Vec(brown.sents())
    model = gensim.models.Word2Vec.load_word2vec_format('/home/loschen/Downloads/GoogleNews-vectors-negative300.bin.gz', binary=True)
    Xall_new = np.zeros((Xall.shape[0],5))
    for i in range(Xall.shape[0]):
        query = Xall["search_term"].iloc[i].lower()
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

def build_query_correction_map(train,test,print_different=True):

    # get all query
    queries = set(train['search_term'].values)
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
    https://www.kaggle.com/hiendang/crowdflower-search-relevance/auto-correct-query
    autocorrect a query based on the training set
    """
    if train is None:
        train = pd.read_csv('./data/train.csv').fillna('')
    if test is None:
        test = pd.read_csv('./data/test.csv').fillna('')
    train_data = train.values[train['search_term'].values==query,:]
    test_data = test.values[test['search_term'].values==query,:]
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


