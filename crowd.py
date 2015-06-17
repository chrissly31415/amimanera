#!/usr/bin/python 
# coding: utf-8

from qsprLib import *

from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition, pipeline, metrics, grid_search

from sklearn.feature_extraction import text

import re

from nltk import word_tokenize,sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import EnglishStemmer

from crowd_features import *

class MyTokenizer(object):
    """
    http://scikit-learn.org/stable/modules/feature_extraction.html
    http://stackoverflow.com/questions/15547409/how-to-get-rid-of-punctuation-using-nltk-tokenizer
    http://nltk.org/api/nltk.tokenize.html
    """
    def __init__(self,stemmer=None,stop_words=None):
      print "Using special tokenizer, stemmer:",stemmer," stop_words:",stop_words
      #self.wnl = LancasterStemmer()
      #self.wnl = PorterStemmer()#best so far
      self.wnl = stemmer
      self.stop_words = stop_words
      #self.doStem = doStem
      #self.wnl = EnglishStemmer(ignore_stopwords=True)
      #self.wnl = WordNetLemmatizer()
    def __call__(self, doc): 
      #words=[word_tokenize(t) for t in sent_tokenize(doc)]
      #words=[item for sublist in words for item in sublist]
      #simple tokenizer 2 e.g. scikit learn, preprocessing is done beforehand
      token_pattern = re.compile(r'\w{1,}')
      words = token_pattern.findall(doc)
      #print words
      #raw_input()
      #print "n words, after:",len(words)
      
      if hasattr(self.wnl,'stem'):
	  words=[self.wnl.stem(t) for t in words]
      #else:
      #	  words=[self.wnl.lemmatize(t) for t in words]
      return words

def loadData(nsamples=-1):
    # Load the training file
    Xtrain = pd.read_csv('data/train.csv')
    Xtest = pd.read_csv('data/test.csv')
    
    # we dont need ID columns
    idx = Xtest.id.values.astype(int)
    Xtrain = Xtrain.drop('id', axis=1)
    Xtest = Xtest.drop('id', axis=1)
    
    # create labels. drop useless columns
    ytrain = Xtrain.median_relevance.values
    Xtrain = Xtrain.drop(['median_relevance', 'relevance_variance'], axis=1)
    
    if nsamples != -1:
      if isinstance(nsamples,str) and 'shuffle' in nsamples:
	  print "Shuffle train data..."
	  rows = np.random.choice(len(Xtrain.index), size=len(Xtrain.index),replace=False)
      else:
	  rows = np.random.randint(0,len(Xtrain.index), nsamples)
      print "unique: %6.2f"%(float(np.unique(rows).shape[0])/float(rows.shape[0]))
      Xtrain = Xtrain.iloc[rows,:]
      ytrain = ytrain[rows]
    
    return Xtest,Xtrain,ytrain,idx


def prepareDataset(seed=123,nsamples=-1,preprocess=False,doSVD=None,standardize=False,doSeparateTFID=None,doSVDseparate=False,computeSim=False,stop_words=text.ENGLISH_STOP_WORDS,doTFID=False,concat=False,useAll=False,concatAll=False,doKmeans=False):
    np.random.seed(seed)
    
    Xtest,Xtrain,ytrain,idx = loadData(nsamples=nsamples)
    Xall = pd.concat([Xtest, Xtrain])
    print "Original shape:",Xall.shape

    if preprocess:
	pass
      
    if doSeparateTFID is not None:
	analyze=False
	print "Vectorize separately...",

	tokenizer = MyTokenizer(stemmer=PorterStemmer())
	
	#vectorizer = HashingVectorizer(stop_words=stop_words,ngram_range=(1,2),analyzer="char", non_negative=True, norm='l2', n_features=2**18)
	#vectorizer = CountVectorizer(min_df=3,  max_features=None, lowercase=True,analyzer="word",ngram_range=(1,2),stop_words=stop_words,strip_accents='unicode')
	vectorizer = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode', analyzer='word',ngram_range=(1, 2), use_idf=True,smooth_idf=True,sublinear_tf=True,stop_words = stop_words,token_pattern=r'\w{1,}',norm='l2')
	#vectorizer = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode', analyzer='word',ngram_range=(1, 5), use_idf=True,smooth_idf=True,sublinear_tf=True,stop_words = stop_words,token_pattern=r'\w{1,}',norm='l2')
	
	Xtrain = Xall[len(Xtest.index):]
	Xtest_t = Xall[:len(Xtest.index)]
	
	for i,col in enumerate(doSeparateTFID):
	    print "Vectorizing: ",col
	    
	    Xtrain[col] = Xtrain[col].fillna("no description")
	    Xtest_t[col] = Xtest_t[col].fillna("no description")
	    
	    print "Is null:",Xtrain[col].isnull().sum()
	    print Xtrain[col].describe()
	    
	    if i>0:
		vectorizer.fit(Xall[col])#should reduce overfitting?
		Xs_all_new = vectorizer.transform(Xall[col])
		#vectorizer.fit(Xtrain[col])
		#Xs_train_new = vectorizer.transform(Xtrain[col])
		#Xs_test_new = vectorizer.transform(Xtest_t[col])		
		
		if doSVDseparate:
		    reducer=TruncatedSVD(n_components=doSVDseparate, algorithm='randomized', n_iter=5, tol=0.0)
		    #Xs_all_new=sparse.vstack((Xs_test_new,Xs_train_new))
		    Xs_all_new=reducer.fit_transform(Xs_all_new)
		    Xs_all = pd.DataFrame(np.hstack((Xs_all,Xs_all_new)))
		    print "Shape Xs_all after SVD:",Xs_all.shape
		    
		    #Xs_train_new=reducer.fit_transform(Xs_train_new)
		    #Xs_test_new=reducer.transform(Xs_test_new)
		    #Xs_train = pd.DataFrame(np.hstack((Xs_train,Xs_train_new)))
		    #Xs_test =  pd.DataFrame(np.hstack((Xs_test,Xs_test_new)))
		    #print "New shape Xs_train:",Xs_train.shape
		    #print "New shape Xs_test:",Xs_test.shape
		    
		    
		else:
		    Xs_train = sparse.hstack((Xs_train,Xs_train_new),format="csr")
		    Xs_test =  sparse.hstack((Xs_test,Xs_test_new),format="csr")
		
	    else:
		#we fit first column on all data!!!
		vectorizer.fit(Xall[col])#should reduce overfitting?
		Xs_all = vectorizer.transform(Xall[col])
		#vectorizer.fit(Xtrain[col])
		#Xs_train = vectorizer.transform(Xtrain[col])
		#Xs_test = vectorizer.transform(Xtest_t[col])	
		print "Shape Xs_all after vectorization:",Xs_all.shape
		
		if doSVDseparate:
		    reducer=TruncatedSVD(n_components=doSVDseparate, algorithm='randomized', n_iter=5, tol=0.0)
		    #Xs_all=sparse.vstack((Xs_test,Xs_train))
		    Xs_all=reducer.fit_transform(Xs_all)
		    print "Shape Xs_all after SVD:",Xs_all.shape
		    #Xs_train=reducer.fit_transform(Xs_train)
		    #Xs_test=reducer.transform(Xs_test)		    
		    #print "New shape Xs_train:",Xs_train.shape
		    #print "New shape Xs_test:",Xs_test.shape
	
	    if analyze:
		analyzer(vectorizer,Xs_train, ytrain)
	
	
	if doSVDseparate:
	    #Xall = pd.concat([Xs_test, Xs_train])
	    Xall = Xs_all
	else:
	    Xall = sparse.vstack((Xs_test,Xs_train),format="csr")
	
	print type(Xall)
	print " shape:",Xall.shape
	#vectorizer.get_features_names()[0:2]

    if concat:
	print "Concatenating query+title, discarding description..."
	# do some lambda magic on text columns
	#data = list(Xall.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))
	#print data[0]
	Xall['query'] = Xall.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1)
	Xall = Xall.drop(['product_title','product_description'], axis=1)
	print Xall.head(10)
    
    if concatAll:
	print "Concatenating query+title+description..."
	Xall['query'] = Xall.apply(lambda x:'%s %s %s' % (x['query'],x['product_title'],['product_description']),axis=1)
	Xall = Xall.drop(['product_title','product_description'], axis=1)
	print Xall.head(10)

    if doTFID:
	print "Fit TFIDF..."
	
	#stop_words = ['http','www','img','border','0','1','2','3','4','5','6','7','8','9','a','the']
	#
	#stop_words = text.ENGLISH_STOP_WORDS.union(stop_words)
	stop_words = text.ENGLISH_STOP_WORDS
	#tokenizer = MyTokenizer(stemmer=PorterStemmer(),stop_words = None)
	#tokenizer = MyTokenizer(stemmer=EnglishStemmer(ignore_stopwords=True),stop_words = stop_words)
	
	#tokenizer = None
	#stop_words = None
	#print "stop_words:",stop_words
	
	#tfv = TfidfVectorizer(min_df=3,  max_features=None, 
	#	strip_accents='unicode', analyzer='word',
	#	ngram_range=(1, 5), use_idf=True,smooth_idf=True,sublinear_tf=True,
	#	stop_words = stop_words,tokenizer=tokenizer)
	
	vectorizer = TfidfVectorizer(min_df=3,  max_features=None, 
		strip_accents='unicode', analyzer='word',
		ngram_range=(1, 5), use_idf=True,smooth_idf=True,sublinear_tf=True,
		stop_words = stop_words,token_pattern=r'\w{1,}',norm='l2')#token_pattern=r"(?u)\b\w\w+\b"
	 
	#tfv = TfidfVectorizer(min_df=3,  max_features=None, 
	#	strip_accents='unicode', analyzer='word',
#		ngram_range=(1, 5), use_idf=True,smooth_idf=True,sublinear_tf=True,tokenizer=tokenizer)
	
	print vectorizer
	if useAll:
	    Xall =  vectorizer.fit_transform(Xall['query'])
	    print Xall
	    
	else:
	    Xtrain = Xall[len(Xtest.index):]
	    Xtest_t = Xall[:len(Xtest.index)]
	    Xtrain=vectorizer.fit_transform(Xtrain['query'])
	    Xtest_t=vectorizer.transform(Xtest_t['query'])
	    Xall = sparse.vstack((Xtest_t,Xtrain),format="csr")
	
	#print vectorizer.get_features_names()
	
	print "Xall:"
	print Xall.shape

    reducer=None
    if doKmeans is not None and doKmeans is not False:
	print "Kmeans components..."
	reducer = MiniBatchKMeans(init='k-means++', n_clusters=doKmeans, n_init=3,batch_size=400)
    
    if doSVD is not None and doSVD is not False:
	print "SVD...components:",doSVD
	reducer=TruncatedSVD(n_components=doSVD, algorithm='randomized', n_iter=5, tol=0.0)
	#tsvd = RandomizedPCA(n_components=doSVD, whiten=True)
    if reducer is not None:
	print reducer
	
	if useAll:
	  Xall = reducer.fit_transform(Xall)
	  
	else:
	  Xtrain = Xall[len(Xtest.index):]
	  Xtest_t = Xall[:len(Xtest.index)]
	  Xtrain=reducer.fit_transform(Xtrain)
	  Xtest_t=reducer.transform(Xtest_t)
	  Xall = np.vstack((Xtest_t,Xtrain))
	
	Xall = pd.DataFrame(Xall)
	#print Xall.describe()
	#df_info(Xall)
    
    if computeSim:
	Xsim = computeSimilarityFeatures(vectorizer=None,nsamples=nsamples,stop_words=stop_words)
      
	if not isinstance(Xall,pd.DataFrame):
	    print "X is not a DataFrame, converting from,",type(Xall)
	    Xall = pd.DataFrame(Xall.todense())
	
	Xall = pd.concat([Xall,Xsim], axis=1)
	print Xall.describe()
	print Xall.columns[-5:]
	
	
    
    if standardize:
	if not isinstance(Xall,pd.DataFrame):
	    print "X is not a DataFrame, converting from,",type(Xall)
	    Xall = pd.DataFrame(Xall.todense())
	Xall = scaleData(lXs=Xall,lXs_test=None)
    
    Xtrain = Xall[len(Xtest.index):]
    Xtest = Xall[:len(Xtest.index)]
    
    print "#Xtrain:",Xtrain.shape
    print "#Xtest:",Xtest.shape
    
    #print type(ytrain)
    print "#ytrain:",ytrain.shape
    print ytrain
    
    return(Xtrain,ytrain,Xtest,idx)



def analyzer(vectorizer,Xs_train, ytrain):
    if isinstance(vectorizer,TfidfVectorizer):
		indices = np.argsort(vectorizer.idf_)[::-1]
		features = vectorizer.get_feature_names()
		top_n = 20
		top_features = [features[i] for i in indices[:top_n]]
		print top_features
	    
    else:
	#indices = np.argsort(vectorizer.idf_)[::-1]
	features = vectorizer.get_feature_names()
	ch2 = SelectKBest(chi2, "all")
	X_train = ch2.fit_transform(Xs_train, ytrain)
	indices = np.argsort(ch2.scores_)[::-1]
	
	for idx in indices:
	    print "idx: %10d Key: %32s  score: %20d"%(idx,features[idx],ch2.scores_[idx])
	    #raw_input()

 
def makePredictions(model=None,Xtest=None,idx=None,filename='submission.csv'):
    # Create your first submission file
    if model is not None: 
	preds = model.predict(Xtest)
    else:
	preds = Xtest

    if idx is None:
	Xtest = pd.read_csv('data/test.csv')
	idx = Xtest.id.values.astype(int)
    
    submission = pd.DataFrame({"id": idx, "prediction": preds})
    submission.to_csv(filename, index=False)


if __name__=="__main__":
    """   
    MAIN PART
    """ 
    #TODO new features from text
    #TODO e.g. length of query, product_title, introduce categories based on titles, use stemmer
    #TODO RandomTreesEmbedding¶
    #https://www.kaggle.com/wliang88/crowdflower-search-relevance/extra-engineered-features-w-svm
    #introduce penalty in classification loss func due to class distance ...? jaccard distance
    #http://stackoverflow.com/questions/17388213/python-string-similarity-with-probability
    #http://avrilomics.blogspot.de/2014/01/calculating-similarity-measure-for-two.html
    #use sample weights...remove high variance queries
    #use t-sne instead of SVD
    #from nltk.corpus import wordnet as wn
    #compute maximum cosine similarity
    #use vocabulary from description to transform query...
    #cv: https://www.kaggle.com/c/crowdflower-search-relevance/forums/t/14350/cross-validation-and-leaderboard-score
    #separate SVD for both cases...
    #onehotencoding
    #number of words found in title
    #do they have the same ordeR?
    #either unsupervised manipulation to whole dataset to avoid overfitting or incldue manipulation into cv
    #https://www.kaggle.com/c/crowdflower-search-relevance/forums/t/14502/word2vec-doc2vec
    
    # Set a seed for consistant results
    t0 = time()
    
    print "numpy:",np.__version__
    print "pandas:",pd.__version__
    print "scipy:",sp.__version__
    
    seed = 123
    nsamples=-1#'shuffle'
    preprocess=False
    doSeparateTFID=['product_title','query']#['query','product_title','product_description']#
    concat=False#query+title
    concatAll=False#query+title+desription
    doTFID=False

    doSVD=False
    doSVDseparate=200
    
    standardize=True
    useAll=False#supervised vs. unsupervised
    computeSim=False
    
    doKmeans=False
    
    garbage=["<.*?>", "http", "www","img","border","style","px","margin","left", "right","font","solid","This translation tool is for your convenience only.*?Note: The accuracy and accessibility of the resulting translation is not guaranteed"]
    garbage2=['http','www','img','border','0','1','2','3','4','5','6','7','8','9','a','the']
    stop_words = text.ENGLISH_STOP_WORDS.union(garbage).union(garbage2)
    
    Xtrain, ytrain, Xtest,idx  = prepareDataset(seed=seed,nsamples=nsamples,preprocess=preprocess,concat=concat,doTFID=doTFID,doSeparateTFID=doSeparateTFID,stop_words=stop_words,computeSim=computeSim,doSVD=doSVD,doSVDseparate=doSVDseparate,standardize=standardize,useAll=useAll,concatAll=concatAll,doKmeans=doKmeans)
     
    #model = Pipeline([('reducer', TruncatedSVD(n_components=400)), ('scaler', StandardScaler()), ('model', model)])
    #model = Pipeline([('scaler', StandardScaler()), ('model', SVC(C=32,gamma=0.001) )])
    #model = SVC(C=32,gamma=0.001)
    #model = Pipeline([('scaler', StandardScaler()), ('model', LinearSVC(C=0.01))])
    #model = Pipeline([('reducer', MiniBatchKMeans(n_clusters=400,batch_size=400,n_init=3)), ('scaler', StandardScaler()), ('SVC', model)])
    #model = Pipeline([('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3)),('classification', LinearSVC())])
    #model =  RandomForestClassifier(n_estimators=500,max_depth=None,min_samples_leaf=1,n_jobs=1,criterion='gini', max_features='auto')
    #model = OneVsRestClassifier(model,n_jobs=8)
    #model = OneVsOneClassifier(model,n_jobs=8)
    #model = SGDClassifier(alpha=.001, n_iter=200,penalty='l2',shuffle=True,loss='log')
    #model = SGDClassifier(alpha=0.0005, n_iter=50,shuffle=True,loss='log',penalty='l2',n_jobs=4)#opt  
    #model = SGDClassifier(alpha=0.0001, n_iter=50,shuffle=True,loss='log',penalty='l2',n_jobs=4)#opt simple processing
    #model = SGDClassifier(alpha=0.00014, n_iter=50,shuffle=True,loss='log',penalty='elasticnet',l1_ratio=0.99)
    #model = LogisticRegression(penalty='l2', tol=0.0001, C=1.0)#opt
    #model = Pipeline([('filter', SelectPercentile(f_classif, percentile=50)), ('model', LogisticRegression(penalty='l2', tol=0.0001, C=1.0))])
    #model = Pipeline([('filter', SelectPercentile(chi2, percentile=70)), ('model', LogisticRegression(penalty='l2', tol=0.0001, C=1.0))])
    #model = Pipeline([('filter', SelectPercentile(f_classif, percentile=15)), ('model', KNeighborsClassifier(n_neighbors=150))])
    #model = Pipeline([('filter', SelectPercentile(chi2, percentile=20)), ('model', MultinomialNB(alpha=0.1))])
    #model = LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1.0, class_weight=None)#opt kaggle params
    #model = LogisticRegressionMod(penalty='l2', dual=False, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1.0, class_weight=None)#opt kaggle params
    #model = RidgeClassifier(tol=1e-2, solver="lsqr",alpha=0.1)
    model = KNeighborsClassifier(n_neighbors=5)
    #model = XgboostClassifier(n_estimators=200,learning_rate=0.3,max_depth=4,subsample=.5,n_jobs=1,objective='multi:softmax',eval_metric='mlogloss',booster='gbtree',silent=1)
    #model = MultinomialNB(alpha=0.001)

    
    # Kappa Scorer 
    scoring_func = make_scorer(quadratic_weighted_kappa, greater_is_better = True)
    
    print model
    
    
    #cv=StratifiedShuffleSplit(ytrain,8,test_size=0.2)
    cv=StratifiedShuffleSplit(ytrain,16,test_size=0.3)
    #cv =StratifiedKFold(ytrain,10,shuffle=True)
    #parameters = {'reducer__n_components': [200,300,400,500,600]}
    #parameters = {'reducer__n_clusters': [1000,1200,1500]}
    #parameters = {'SVC__C': [2,8,16,32],'SVC__gamma':[0.001,0.003,0.0008]}
    #parameters = {'C': [0.1,0.01,0.025,0.05]}
    #parameters = {'alpha': [0.005,0.001,0.0025],'n_iter':[200],'penalty':['l2'],'loss':['log','perceptron']}
    #parameters = {'n_estimators':[200],'max_depth':[4,6],'learning_rate':[0.3,0.4,0.5],'subsample':[0.5]}
    #parameters = {'alpha':[0.001]}
    #model = makeGridSearch(model,Xtrain,ytrain,n_jobs=8,refit=True,cv=cv,scoring=scoring_func,parameters=parameters,random_iter=None)
    model = buildModel(model,Xtrain,ytrain,cv=cv,scoring=scoring_func,n_jobs=8,trainFull=True,verbose=True)
    #model = buildClassificationModel(model,Xtrain.values,ytrain,class_names=['1','2','3','4'],trainFull=False,cv=cv)
    makePredictions(model,Xtest,idx=idx,filename='submissions/sub09062015e.csv')
    print("Model building done in %fs" % (time() - t0))