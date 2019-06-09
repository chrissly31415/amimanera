#!/usr/bin/python
# coding: utf-8

from qsprLib import *
from keras_tools import *

import os,re

#sys.path.append('/home/loschen/calc/smuRF/python_wrapper')
#import smurf as sf

sys.path.append('/home/loschen/programs/xgboost/wrapper')
import xgboost as xgb

sys.path.append('/home/loschen/calc/amimanera/utils')
from nlp_features import *
from interact_analysis import *

import matplotlib.pyplot as plt


def prepareDataset(quickload=False, seed=123, nsamples=-1, holdout=False, keepFeatures=None, dropFeatures = None,resampletrain=None,analyzeUniques=None,notebook_feats=None, notebook_ext=None,dummy_encoding=None,labelEncode=None, oneHotenc=None,removeRare_freq=None, logtransform = None, stemmData=False,  createCommonWords=False, useAttributes= False, doTFIDF= None, computeSim = None, n_components = 20, cleanData = False, merge_product_infos = None,computeAddFeates=False, computeAddFeates_new = False, word2vecFeates= False, word2vecFeates_new = False, spellchecker = False, removeCorr=False, query_correction=False, createVerticalFeatures=False, use_new_spellchecker=False):
    print("Preparing data ...")
    np.random.seed(seed)

    if isinstance(quickload,str):
        store = pd.HDFStore(quickload)
        print(store)
        Xtest = store['Xtest']
        Xtrain = store['Xtrain']
        ytrain = store['ytrain']
        Xval = store['Xval']
        yval = store['yval']
        test_id = store['test_id']

        return Xtest, Xtrain, ytrain.values, test_id, None, Xval, yval.values

    os.remove('./data/store.h5')
    store = pd.HDFStore('./data/store.h5')
    #print store

    Xtrain = pd.read_csv('./data/train.csv')
    #Xtrain = pd.read_csv('./data/train.csv')
    Xtest = pd.read_csv('./data/test.csv').iloc[:1000]

    print(Xtrain.describe(include='all'))
    print("Xtrain.shape:",Xtrain.shape)

    print("Xtest.shape:",Xtest.shape)

    print("Xtrain - ISNULL:",Xtrain.isnull().any(axis=0))
    print("Xtest - ISNULL:",Xtest.isnull().any(axis=0))

    if nsamples != -1:
        if isinstance(nsamples, str) and 'shuffle' in nsamples:
            print("Shuffle train data...")
            rows = np.random.choice(len(Xtrain.index), size=len(Xtrain.index), replace=False)
        else:
            rows = np.random.choice(len(Xtrain.index), size=nsamples, replace=False)

        print("unique rows: %6.2f" % (float(np.unique(rows).shape[0]) / float(rows.shape[0])))
        Xtrain = Xtrain.iloc[rows, :]

    #Xtrain.groupby([Xtrain.Date.dt.year,Xtrain.Date.dt.month])['Sales'].mean().plot(kind="bar")
    #Xtest.groupby([Xtest.Date.dt.year,Xtest.Date.dt.month])['Store'].mean().plot(kind="bar")

    #rearrange
    ytrain = Xtrain['is_duplicate'].values
    train_id = Xtrain['id']

    test_id = Xtest['test_id']
    store['test_id'] = test_id
    Xtest.drop(['test_id'],axis=1,inplace=True)

    #check
    print(Xtrain.shape)
    print(Xtest.shape)

    if analyzeUniques is not None:
        print("Analyzing search terms question1:")
        compareList(Xtrain,Xtest, 'question1',verbose=True)

        print("Analyzing search terms question2:")
        compareList( Xtrain, Xtest,'question2', verbose=True)

        qids = pd.Series(Xtrain['qid1'].tolist() + Xtest['qid2'].tolist())
        print(('Total number of questions in the training data: {}'.format(len(
            np.unique(qids)))))
        print(('Number of questions that appear multiple times: {}'.format(np.sum(qids.value_counts() > 1))))

    Xtrain.drop(['is_duplicate', 'id', 'qid1', 'qid2'], axis=1, inplace=True)
    Xall = pd.concat([Xtest, Xtrain], ignore_index=True)

    if spellchecker:
        #https://www.kaggle.com/steubk/home-depot-product-search-relevance/fixing-typos/discussion
        print("Google Spellchecker for search term...")
        Xall['question1'] = Xall['question1'].map(lambda x:spell_checking(x))

    if cleanData is not None:
        Xall.fillna("NA", inplace=True)


    print("NaN in Xtrain:", pd.isnull(Xall).any(1).nonzero()[0])
    if  stemmData is not None:
       pass

    if createVerticalFeatures:
        pass

    if word2vecFeates:
        pass

    if computeAddFeates:
        pass

    if notebook_feats:

        print("Creating notebook features:")
        Xall['word_match'] = Xall.apply(word_match_share, axis=1, raw=True)
        Xall['word_match'].fillna(Xall['word_match'].mean(),inplace=True)
        Xall['tfidf_word_match'] = get_tfidf_share(Xall)
        Xall['tfidf_word_match'].fillna(Xall['tfidf_word_match'].mean(), inplace=True)
        print("NA,after:", Xall.isnull().values.sum())
        print(Xall.describe())

    if notebook_ext:
        print("Creating ext. notebook features:")
        train_qs = pd.Series(Xall['question1'].tolist() + Xall['question2'].tolist()).astype(str)
        Xall['marks'] = np.mean(train_qs.apply(lambda x: '?' in x))
        Xall['math'] = np.mean(train_qs.apply(lambda x: '[math]' in x))
        Xall['fullstop'] = np.mean(train_qs.apply(lambda x: '.' in x))
        Xall['capital_first'] = np.mean(train_qs.apply(lambda x: x[0].isupper()))
        Xall['capitals'] = np.mean(train_qs.apply(lambda x: max([y.isupper() for y in x])))
        Xall['numbers'] = np.mean(train_qs.apply(lambda x: max([y.isdigit() for y in x])))


    Xsim = None
    if computeSim is not None:
        reducer = None
        if 'reducer' in computeSim:
            reducer = computeSim['reducer']
        Xsim = computeSimilarityFeatures(Xall, columns=computeSim['columns'], verbose=False, useOnlyTrain=True,
                                         stop_words=stop_words, doSVD=computeSim['doSVD'],
                                         vectorizer=computeSim['vectorizer'], reducer=reducer)
        print(Xsim.describe())
        Xall = pd.concat([Xall, Xsim], axis=1)

    if doTFIDF is not None:
        print("Doing TFIDF:", doTFIDF)
        vectorizer = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word',
                                     ngram_range=(1, 2), use_idf=True, smooth_idf=True, sublinear_tf=True,
                                     stop_words=stop_words, token_pattern=r'\w{1,}', norm='l2')
        if isinstance(doTFIDF, dict):
            print(list(doTFIDF.keys()))
            vectorizer = doTFIDF['vectorizer']
            doTFIDF = doTFIDF['columns']
        for i, (n_comp, col) in enumerate(zip(n_components, doTFIDF)):
            print(Xall.shape)
            print("Xall - ISNULL:", Xall.isnull().any(axis=0))
            print("Vectorizing: " + col + " n_components:" + str(n_comp))
            vectorizer.fit(Xall[col])
            Xs_all_new = vectorizer.transform(Xall[col])
            reducer = TruncatedSVD(n_components=n_comp, algorithm='randomized', n_iter=5, tol=0.0)
            Xs_all_new = reducer.fit_transform(Xs_all_new)
            print("Variance explained:", np.sum(reducer.explained_variance_ratio_))
            Xs_all_new = pd.DataFrame(Xs_all_new)
            Xs_all_new.columns = [col + "_svd_" + str(x) for x in Xs_all_new.columns]
            # store['Xs_all_new'] = Xs_all_new

            Xall = pd.concat([Xall, Xs_all_new], axis=1)



    if createCommonWords:
        """
        See https://www.kaggle.com/wenxuanchen/home-depot-product-search-relevance/sklearn-random-forest/code
        also: https://www.kaggle.com/the1owl/home-depot-product-search-relevance/rf-mean-squared-error/code
        """
        pass


    if dummy_encoding is not None:
        print("Dummy encoding,skip label encoding")
        Xall = pd.get_dummies(Xall,columns=dummy_encoding)

    if labelEncode is not None:
        print("Label encode")
        for col in labelEncode:
            lbl = preprocessing.LabelEncoder()
            Xall[col] = lbl.fit_transform(Xall[col].values)
            vals = Xall[col].unique()
            print("Col: %s Vals %r:"%(col,vals))
            #print "Orig:",list(lbl.inverse_transform(Xall[col].unique()))

    if removeRare_freq is not None:
        print("Remove rare features based on frequency...")
        for col in oneHotenc:
            ser = Xall[col]
            counts = list(ser.value_counts().keys())
            idx = ser.value_counts() > removeRare_freq
            threshold = idx.astype(int).sum()
            print("%s has %d different values before, min freq: %d - threshold %d" % (
                col, len(counts), removeRare_freq, threshold))
            if len(counts) > threshold:
                ser[~ser.isin(counts[:threshold])] = 9999
            if len(counts) <= 1:
                print(("Dropping Column %s with %d values" % (col, len(counts))))
                Xall = Xall.drop(col, axis=1)
            else:
                Xall[col] = ser.astype('category')
            print(ser.value_counts())
            counts = list(ser.value_counts().keys())
            print("%s has %d different values after" % (col, len(counts)))


    if oneHotenc is not None:
        print("1-0 Encoding categoricals...", oneHotenc)
        for col in oneHotenc:
            #print "Unique values for col:", col, " -", np.unique(Xall[col].values)
            encoder = OneHotEncoder()
            X_onehot = pd.DataFrame(encoder.fit_transform(Xall[[col]].values).todense())
            X_onehot.columns = [col + "_hc_" + str(column) for column in X_onehot.columns]
            print("One-hot-encoding of %r...new shape: %r" % (col, X_onehot.shape))
            Xall.drop([col], axis=1, inplace=True)
            Xall = pd.concat([Xall, X_onehot], axis=1)
            print("One-hot-encoding final shape:", Xall.shape)
            # raw_input()

    if logtransform is not None:
        print("log Transform")
        for col in logtransform:
            if col in Xall.columns:
                if Xall[col].min()>-0.99:
                    Xall[col] = Xall[col].map(np.log1p+1.0)

    if keepFeatures is not None:
        dropcols = [col for col in Xall.columns if col not in keepFeatures]
        for col in dropcols:
            if col in Xall.columns:
                print("Dropping: ", col)
                Xall.drop([col], axis=1, inplace=True)
        Xall.sort(columns=keepFeatures,inplace=True)

    if dropFeatures is not None:
        for col in dropFeatures:
            if col in Xall.columns:
                print("Dropping: ", col)
                Xall.drop([col], axis=1, inplace=True)

    if removeCorr:
        Xall = removeCorrelations(Xall, threshhold=0.99)

    #Xall = Xall.astype(np.float32)
    print("Columns used",list(Xall.columns))

    #split data
    Xtrain = Xall[len(Xtest.index):]
    Xtest = Xall[:len(Xtest.index)]
    Xval = None
    yval = pd.Series()

    if resampletrain:
        Xtrain_p = Xtrain[ytrain==1]
        Xtrain_n = Xtrain[ytrain==0]
        print("old ratio:", (len(Xtrain_p) / float(len(Xtrain_p) + len(Xtrain_n))))
        print(Xtrain_p.shape)
        print(Xtrain_n.shape)
        # Now we oversample the negative class
        # There is likely a much more elegant way to do this...
        p = 0.165
        scale = ((len(Xtrain_p) / float(len(Xtrain_p) + len(Xtrain_n))) / p) - 1.0
        print(scale)
        while scale > 1:
            Xtrain_n = pd.concat([Xtrain_n, Xtrain_n])
            scale -= 1
        Xtrain_n = pd.concat([Xtrain_n, Xtrain_n[:int(scale * len(Xtrain_n))]])
        print("new ratio:",(len(Xtrain_p) / float(len(Xtrain_p) + len(Xtrain_n))))

        Xtrain = pd.concat([Xtrain_p, Xtrain_n])
        print(Xtrain.shape)
        ytrain = np.asarray((np.zeros(len(Xtrain_p)) + 1).tolist() + np.zeros(len(Xtrain_n)).tolist())

    if isinstance(holdout,str) or holdout:
        print("Split holdout...")
        if isinstance(holdout,str):
            hl = pd.read_csv(holdout)['id']
            hmask = np.in1d(train_id.values, hl.values)
            tmask = np.logical_not(hmask)

            Xval = Xtrain[hmask]
            yval = ytrain[hmask]

            ytrain = ytrain[tmask]
            Xtrain = Xtrain[tmask]

        else:
            Xtrain, Xval, ytrain, yval = train_test_split(Xtrain, ytrain, test_size=0.3, stratify=ytrain,random_state=666)

        print(Xtrain.head(10))
        print(Xval.head(10))
        print("Shape Xtrain:",Xtrain.shape)
        print("Shape Xval  :",Xval.shape)
        #print ytrain
        #print ytrain.shape
        #raw_input()


    df_list = [Xtest,Xtrain,ytrain,Xval,yval,test_id]
    name_list = ['Xtest','Xtrain','ytrain','Xval','yval','test_id']
    name_list = []
    for label,ldf in zip(name_list,df_list):
        if ldf is not None:
            try:
                print("Store:",label)
                ldf = ldf.reindex(copy = False)
                store.put(label, ldf, format='table', data_columns=True)
            except:
                print("Could not save to pytables:")
                print(ldf.apply(lambda x: pd.lib.infer_dtype(x.values)))

    store.close()

    return Xtest, Xtrain, ytrain, test_id, None, Xval, yval


def mergeWithXval(Xtrain,Xval,ytrain,yval):
    Xtrain =  pd.concat([Xtrain, Xval], ignore_index=True)
    ytrain = np.hstack((ytrain,yval))
    return Xtrain,ytrain


def makePredictions(model=None,Xtest=None,idx=None,filename='submission.csv'):
    print("Saving submission: ", filename)
    if model is not None:
        preds = model.predict_proba(Xtest).flatten()
        print(preds.shape)
    else:
        preds = Xtest
    if idx is None:
        idx = np.arange(Xtest.shape[0])+1

    result = pd.DataFrame({"id": idx, 'relevance': preds})

    result['relevance'] = result['relevance'].clip(1.0,3.0)
    result.to_csv(filename, index=False)


if __name__ == "__main__":
    """
    MAIN PART
    """

    """
    TODO:
    question features: e.g. https://www.kaggle.com/anokas/quora-question-pairs/data-analysis-xgboost-starter-0-35460-lb/notebook
    word share features
    """

    pd.options.display.mpl_style = 'default'
    plt.interactive(False)

    t0 = time()

    print("numpy:", np.__version__)
    print("pandas:", pd.__version__)
    print("scipy:", sp.__version__)

    all_feats = None

    pmt = dict()
    pmt['quickload'] = False #'./data/store.h5'
    pmt['seed'] = 42
    pmt['nsamples'] = -1
    pmt['holdout'] = True
    pmt['resampletrain'] = True
    pmt['dropFeatures'] = ['question1','question2']
    pmt['keepFeatures'] = None
    pmt['analyzeUniques'] = None
    pmt['notebook_feats'] = True
    pmt['notebook_ext'] = True
    pmt['dummy_encoding'] = None
    pmt['labelEncode'] = None
    pmt['removeRare_freq'] = None#
    pmt['oneHotenc'] = None
    pmt['logtransform'] = None
    pmt['cleanData'] = True
    pmt['stemmData'] =  None
    pmt['createCommonWords'] = None
    pmt['useAttributes'] = None
    pmt['doTFIDF'] = None#['search_term','product_title']
    pmt['n_components'] = [50,50]
    pmt['computeSim'] = {'columns': ['question1','question2'],'doSVD': 100, 'vectorizer': TfidfVectorizer(min_df=10,  max_features=None, strip_accents='unicode', analyzer='word',ngram_range=(1, 2), use_idf=True,smooth_idf=True,sublinear_tf=True,stop_words = stop_words,token_pattern=r'\w{1,}')}
    #pmt['computeSim'] = {'columns': ['question1', 'question2'], 'doSVD': 50,'vectorizer': HashingVectorizer(stop_words='english',ngram_range=(1,2),analyzer="word", non_negative=True, norm='l2', n_features=2**10)}
    #pmt['computeSim'] = None
    pmt['computeAddFeates'] = None
    pmt['computeAddFeates_new'] = None
    pmt['word2vecFeates'] = None
    pmt['word2vecFeates_new'] = None
    pmt['removeCorr'] = None
    pmt['spellchecker'] = None
    pmt['query_correction'] = None
    pmt['createVerticalFeatures'] = None
    

    Xtest, Xtrain, ytrain, idx, sample_weight, Xval, yval = prepareDataset(**pmt)
    print(list(Xtrain.columns))
    print(Xtrain.describe())
    #interact_analysis(Xtrain)
    #model = RandomForestClassifier(n_estimators=250,max_depth=None,min_samples_leaf=5,n_jobs=2, max_features=Xtrain.shape[1]/3,oob_score=False)
    model = XgboostClassifier(n_estimators=400,learning_rate=0.02,max_depth=4,n_jobs=2,objective='binary:logistic',eval_metric='logloss',silent=1,eval_size=0.0)
    #model = BaggingClassifier(RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0), n_estimators=45, max_samples=0.1, random_state=25)
    #model = ExtraTreesClassifier(n_estimators=250,max_depth=None,min_samples_leaf=1,n_jobs=2, max_features=3*Xtrain.shape[1]/4)
    #model = KNeighbors(n_neighbors=20)
    #model = LogisticRegression()
    #model = DummyClassifier(constant=0.376)
    #model  = KerasNN(dims=Xtrain.shape[1],nb_classes=1,nb_epoch=20,learning_rate=0.0001,validation_split=0.2,batch_size=512,verbose=1,activation='sigmoid', layers=[256,256], dropout=[0.1,0.1],loss='mse') # best

    #model  = KerasNN(dims=Xtrain.shape[1],nb_classes=1,nb_epoch=20,learning_rate=0.0001,validation_split=0.0,batch_size=512,verbose=1,activation='sigmoid', layers=[256,256], dropout=[0.1,0.0],loss='mse')
    #model = Pipeline([('scaler', StandardScaler()), ('nn',model)])
    #model = BaggingRegressor(model,n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=False) #RMSE0.461
    #model = Pipeline([('scaler', RobustScaler()), ('nn',model)])

    #model = KNeighborsRegressor(n_neighbors=10)

    #cv = LabelKFold(Xtrain['product_uid'], n_folds=8)
    # cv = KFold(X.shape[0], n_folds=folds,shuffle=True)
    #cv = StratifiedShuffleSplit(ytrain,2)
    #cv_labels = pd.Series.from_csv('./data/labels_for_cv.csv')
    #cv = LabelKFold(cv_labels, n_folds=8)
    #cv = LabelShuffleSplit(Xtrain['search_term'],n_iter=8, test_size= 0.2 )
    cv = KFold(Xtrain.shape[0],8,shuffle=True)
    #cv = KLabelFolds(Xtrain['search_term'], n_folds=8, repeats=1)

    #print df.printSummary()
    parameters = {'n_estimators':[400,800],'max_depth':[4,8],'learning_rate':[0.02,0.005],'subsample':[0.75],'colsample_bytree':[0.75]}
    #parameters = {'n_estimators':[500],'min_samples_leaf':[5,8,10],'max_features':[100]}
    #parameters = {'nn__nb_epoch':[20,40],'nn__learning_rate':[0.01], 'nn__dropout':[[0.1]*2,[0.05]*2],'nn__layers':[[256]*2,[512]*2],'nn__batch_size':[64]}
    #parameters={'n_neighbors':[20,25,30]}
    model = makeGridSearch(model, Xtrain, ytrain, n_jobs=4, refit=True, cv=cv, scoring='neg_log_loss',parameters=parameters, random_iter=-1)

    #Xtrain, ytrain = mergeWithXval(Xtrain,Xval,ytrain,yval)
    print(Xtrain.shape)
    print(model)
    buildModel(model,Xtrain,ytrain,cv=cv, scoring='neg_log_loss', n_jobs=2,trainFull=False,verbose=True)
    #analyzeLearningCurve(model, Xtrain, ytrain, cv=cv, score_func='roc_auc')
    #buildXvalModel(model,Xtrain,ytrain,sample_weight=None,class_names=None,refit=False,cv=cv)

    print("Evaluation data set...")
    model.fit(Xtrain,ytrain)
    yval_pred = model.predict_proba(Xval)[:,1]

    print(" Eval-score: {:5.4f}".format(log_loss(yval, yval_pred, eps=1e-15)))

    #print "Training the final model (incl. Xval.)"
    #Xtrain, ytrain = mergeWithXval(Xtrain,Xval,ytrain,yval)
    #model.fit(Xtrain,ytrain)

    #makePredictions(model,Xtest,idx=idx, filename='./submissions/subtest.csv')

    plt.show()
    print(("Model building done in %fs" % (time() - t0)))
