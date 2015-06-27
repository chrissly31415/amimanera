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
      self.wnl = stemmer
      self.stop_words = stop_words
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
    
    Xtrain["product_description"] = Xtrain["product_description"].fillna("no_description")
    Xtest["product_description"] = Xtest["product_description"].fillna("no_description")
    
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


def prepareDataset(seed=123,nsamples=-1,preprocess=False,vectorizer=None,doSVD=None,standardize=False,doBenchMark=False,computeFeatures=None,computeGenSim=None,addNoiseColumns=None,doSeparateTFID=None,doSVDseparate=False,computeSim=False,stop_words=text.ENGLISH_STOP_WORDS,doTFID=False,concat=False,useAll=False,concatAll=False,doKmeans=False):
    np.random.seed(seed)
    
    Xtest,Xtrain,ytrain,idx = loadData(nsamples=nsamples)
    Xall = pd.concat([Xtest, Xtrain])
    print "Original shape:",Xall.shape

    if preprocess:
	pass
    
    if doBenchMark:
	Xall = useBenchmarkMethod_mod(Xall)
    
    Xfeat = None
    if computeFeatures is not None and computeFeatures is not False:
        Xfeat = additionalFeatures(Xall,verbose=False)
        print Xfeat.describe()
    
    Xgensim = None
    if computeGenSim is not None:
	Xgensim = genSimFeatures(Xall,verbose=False)
	print Xgensim.describe()

    Xsim = None
    if computeSim is not None:
	Xsim = computeSimilarityFeatures(Xall)
	print Xsim.describe()
    
    
    if doSeparateTFID is False or doSeparateTFID is not None:
	analyze=False
	print "Vectorize columns separately...",

	tokenizer = MyTokenizer(stemmer=PorterStemmer())
	
	#vectorizer = HashingVectorizer(stop_words=stop_words,ngram_range=(1,2),analyzer="char", non_negative=True, norm='l2', n_features=2**18)
	#vectorizer = CountVectorizer(min_df=3,  max_features=None, lowercase=True,analyzer="word",ngram_range=(1,2),stop_words=stop_words,strip_accents='unicode')
	#vectorizer = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode', analyzer='word',ngram_range=(1, 2), use_idf=True,smooth_idf=True,sublinear_tf=True,stop_words = stop_words,token_pattern=r'\w{1,}',norm='l2')
	if vectorizer is None:
	  print "Using default vectorizer..."
	  vectorizer = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode', analyzer='word',ngram_range=(1, 5), use_idf=True,smooth_idf=True,sublinear_tf=True,stop_words = stop_words,token_pattern=r'\w{1,}',norm='l2')
	else:
	  print vectorizer
	Xtrain = Xall[len(Xtest.index):]
	Xtest_t = Xall[:len(Xtest.index)]
	
	for i,col in enumerate(doSeparateTFID):
	    print "Vectorizing: ",col
	    
	    print "Is null:",Xtrain[col].isnull().sum()
	    print Xtrain[col].describe()
	    
	    if i>0:
		vectorizer.fit(Xall[col])#should reduce overfitting?
		Xs_all_new = vectorizer.transform(Xall[col])		
		
		if doSVDseparate:
		    reducer=TruncatedSVD(n_components=doSVDseparate, algorithm='randomized', n_iter=5, tol=0.0)
		    #Xs_all_new=sparse.vstack((Xs_test_new,Xs_train_new))
		    Xs_all_new=reducer.fit_transform(Xs_all_new)
		    Xs_all = pd.DataFrame(np.hstack((Xs_all,Xs_all_new)))
		    print "Shape Xs_all after SVD:",Xs_all.shape
		
	    else:
		#we fit first column on all data!!!
		vectorizer.fit(Xall[col])#should reduce overfitting?
		Xs_all = vectorizer.transform(Xall[col])
		print "Shape Xs_all after vectorization:",Xs_all.shape
		
		if doSVDseparate:
		    reducer=TruncatedSVD(n_components=doSVDseparate, algorithm='randomized', n_iter=5, tol=0.0)
		    #Xs_all=sparse.vstack((Xs_test,Xs_train))
		    Xs_all=reducer.fit_transform(Xs_all)
		    print "Shape Xs_all after SVD:",Xs_all.shape
	
	    if analyze:
		analyzer(vectorizer,Xs_train, ytrain)
	
	
	if doSVDseparate:
	    #Xall = pd.concat([Xs_test, Xs_train])
	    Xall = Xs_all
	
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
	stop_words = text.ENGLISH_STOP_WORDS
	
	if vectorizer is None: 
	  print "Default vectorizer:"
	  vectorizer = TfidfVectorizer(min_df=3,  max_features=None, 
		strip_accents='unicode', analyzer='word',
		ngram_range=(1, 5), use_idf=True,smooth_idf=True,sublinear_tf=True,
		stop_words = stop_words,token_pattern=r'\w{1,}',norm='l2',tokenizer=None)
	
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
	#reducer = RandomizedPCA(n_components=doSVD, whiten=True)
	
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
    
    """
    if computeSim:
	Xsim = computeSimilarityFeatures(vectorizer=None,nsamples=nsamples,stop_words=stop_words)
      
	if not isinstance(Xall,pd.DataFrame):
	    print "X is not a DataFrame, converting from,",type(Xall)
	    Xall = pd.DataFrame(Xall.todense())
	
	Xall = pd.concat([Xall,Xsim], axis=1)
	print Xall.describe()
	print Xall.columns[-5:]
    """

    if addNoiseColumns is not None:
	print "Adding %d random noise columns"%(addNoiseColumns)
	Xrnd = pd.DataFrame(np.random.randn(Xall.shape[0],addNoiseColumns))
	#print "Xrnd:",Xrnd.shape
	#print Xrnd
	for col in Xrnd.columns:
	    Xrnd=Xrnd.rename(columns = {col:'rnd'+str(col+1)})
	
	Xall = pd.concat([Xall, Xrnd],axis=1)

    if Xsim is not None:
	  if isinstance(computeSim,str) and 'only' in computeSim:
	      Xall = Xsim
	  else:
	      Xall = pd.concat([Xall,Xsim], axis=1)

    if Xfeat is not None:
	  if isinstance(computeFeatures,str) and 'only' in computeFeatures:
	      Xall = Xfeat
	  else:
	      Xall = pd.concat([Xall,Xfeat], axis=1)
	
    if standardize:
	if not isinstance(Xall,pd.DataFrame):
	    print "X is not a DataFrame, converting from,",type(Xall)
	    Xall = pd.DataFrame(Xall.todense())
	Xall = scaleData(lXs=Xall,lXs_test=None)
    
    Xtrain = Xall[len(Xtest.index):]
    Xtest = Xall[:len(Xtest.index)]
    
    if not isinstance(Xtrain,list):
      print "#Xtrain:",Xtrain.shape
      print "#Xtest:",Xtest.shape
    
    #print type(ytrain)
    print "#ytrain:",ytrain.shape
    
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
    #onehotencoding with countvectorizers
    #number of words found in title
    #do they have the same ordeR?
    #either unsupervised manipulation to whole dataset to avoid overfitting or incldue manipulation into cv
    #https://www.kaggle.com/c/crowdflower-search-relevance/forums/t/14502/word2vec-doc2vec
    #use similarity total count like in: https://www.kaggle.com/c/crowdflower-search-relevance/forums/t/14159/beating-the-benchmark-yet-again?page=2
    #use NN: https://www.kaggle.com/c/crowdflower-search-relevance/forums/t/14159/beating-the-benchmark-yet-again?page=3
    #cross-validation: https://www.kaggle.com/c/crowdflower-search-relevance/forums/t/14350/cross-validation-and-leaderboard-score
    #for ensemble do like 3fold 5 repeats-> average predictions from each model trained
    #write wrapper for special regression that is binned into 4 categories...
    #use bagging classifier
    #sample weights
    #char ngrams
    #clean data remove special characters ( - + ) "
    #https://www.kaggle.com/triskelion/crowdflower-search-relevance/normalized-kaggle-distance
    #LINEARsvm
    
    # Set a seed for consistant results
    t0 = time()
    
    print "numpy:",np.__version__
    print "pandas:",pd.__version__
    print "scipy:",sp.__version__
    
    seed = 123
    nsamples=-1#'shuffle'
    preprocess=False
    doBenchMark=False
    computeFeatures=True
    computeGenSim=None
    computeSim=True
    addNoiseColumns=None
    doSeparateTFID=['product_title','query']#['query','product_title','product_description']#
    doSVDseparate=200
    doTFID=False
    doSVD=False
   
    standardize=True
    useAll=False#supervised vs. unsupervised
    concat=False#query+title
    concatAll=False#query+title+desription    
    doKmeans=False
    
    garbage=["<.*?>", "http", "www","img","border","style","px","margin","left", "right","font","solid","This translation tool is for your convenience only.*?Note: The accuracy and accessibility of the resulting translation is not guaranteed"]
    garbage2=['http','www','img','border','0','1','2','3','4','5','6','7','8','9','a','the']
    stop_words = text.ENGLISH_STOP_WORDS.union(garbage).union(garbage2)
    
    Xtrain, ytrain, Xtest,idx  = prepareDataset(seed=seed,nsamples=nsamples,preprocess=preprocess,doBenchMark=doBenchMark,computeFeatures=computeFeatures,computeGenSim=computeGenSim,addNoiseColumns=addNoiseColumns,concat=concat,doTFID=doTFID,doSeparateTFID=doSeparateTFID,stop_words=stop_words,computeSim=computeSim,doSVD=doSVD,doSVDseparate=doSVDseparate,standardize=standardize,useAll=useAll,concatAll=concatAll,doKmeans=doKmeans)
    Xtrain.to_csv('./data/Xtrain.csv',index=False)
    pd.DataFrame(ytrain,columns=['class']).to_csv('./data/ytrain.csv',index=False)
    
    #model = Pipeline([('v',TfidfVectorizer(min_df=5, max_df=500, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2), use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words = 'english')), ('svd', TruncatedSVD(n_components=200, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)), ('scl', StandardScaler(copy=True, with_mean=True, with_std=True)), ('svm', SVC(C=10.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None))])
    
    #model = Pipeline([('reducer', TruncatedSVD(n_components=400)), ('scaler', StandardScaler()), ('model', model)])
    #model = Pipeline([('scaler', StandardScaler()), ('model', SVC(C=10,gamma='auto') )])
    #model = SVC(C=10,gamma='auto')
    #model = Pipeline([('scaler', StandardScaler()), ('model', LinearSVC(C=0.01))])
    #model = Pipeline([('reducer', MiniBatchKMeans(n_clusters=400,batch_size=400,n_init=3)), ('scaler', StandardScaler()), ('SVC', model)])
    #model = Pipeline([('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3)),('classification', LinearSVC())])
    
    
    #model =  RandomForestClassifier(n_estimators=500,max_depth=None,min_samples_leaf=1,n_jobs=1,criterion='gini', max_features='auto')
    #model.fit(Xtrain,ytrain)
    #rfFeatureImportance(model,Xtrain,Xtest,1)
    
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
    #model = KNeighborsClassifier(n_neighbors=5)
    model = XgboostClassifier(n_estimators=200,learning_rate=0.1,max_depth=4,subsample=.5,n_jobs=1,objective='multi:softmax',eval_metric='mlogloss',booster='gbtree',silent=1)
    #model = MultinomialNB(alpha=0.001)
  
    # Kappa Scorer 
    scoring_func = make_scorer(quadratic_weighted_kappa, greater_is_better = True)
    
    print model
    
    
    #cv=StratifiedShuffleSplit(ytrain,8,test_size=0.2)
    cv=StratifiedShuffleSplit(ytrain,16,test_size=0.3)
    #cv =StratifiedKFold(ytrain,10,shuffle=True)
    #parameters = {'reducer__n_components': [200,300,400,500,600]}
    #parameters = {'reducer__n_clusters': [1000,1200,1500]}
    #parameters = {'SVC__C': [8,16,32],'SVC__gamma':[0.001,0.003,0.0008]}
    #parameters = {'C': [8,16,32],'gamma':[0.001,0.003,0.0008,'auto']}
    #parameters = {'C': [0.1,0.01,0.025,0.05]}
    #parameters = {'alpha': [0.005,0.001,0.0025],'n_iter':[200],'penalty':['l2'],'loss':['log','perceptron']}
    #parameters = {'n_estimators':[200],'max_depth':[4],'learning_rate':[0.1,0.08],'subsample':[0.5,1.0]}
    #parameters = {'n_estimators':[50,100,250,500],'max_features':[10,20,30,40,'auto'],'criterion':['gini','entropy'],'min_samples_leaf':[1,2,3]}
    #parameters = {'alpha':[0.001]}
    #model = makeGridSearch(model,Xtrain,ytrain,n_jobs=2,refit=True,cv=cv,scoring=scoring_func,parameters=parameters,random_iter=-1)
    model = buildModel(model,Xtrain,ytrain,cv=cv,scoring=scoring_func,n_jobs=2,trainFull=False,verbose=True)
    #model = buildClassificationModel(model,Xtrain,ytrain,class_names=['1','2','3','4'],trainFull=False,cv=cv)
    #makePredictions(model,Xtest,idx=idx,filename='submissions/sub25062015a.csv')
    print("Model building done in %fs" % (time() - t0))