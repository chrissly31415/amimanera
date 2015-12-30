#!/usr/bin/python
# coding: utf-8

from qsprLib import *
from lasagne_tools import *
from keras_tools import *

sys.path.append('/home/loschen/calc/smuRF/python_wrapper')
import smurf as sf

sys.path.append('/home/loschen/programs/xgboost/wrapper')
import xgboost as xgb

from interact_analysis import *

import matplotlib.pyplot as plt
import math
import os
pd.options.display.mpl_style = 'default'


def prepareDataset(quickload=False, seed=123, nsamples=-1, holdout=False, keepFeatures=None, dropFeatures = None,dummy_encoding=None,labelEncode=None, oneHotenc=None,removeRare_freq=None, createVerticalFeatures=False, logtransform = None, exludePatients=True, useActivity=True):
    np.random.seed(seed)

    large = pd.HDFStore('./data/store_large.h5')
    store = pd.HDFStore('./data/store.h5')
    print large
    print store

    if quickload:
        Xtest = store['Xtest']
        Xtrain = store['Xtrain']
        ytrain = store['ytrain']
        Xval = store['Xval']
        yval = store['yval']
        test_id = store['test_id']

        return Xtest, Xtrain, ytrain, test_id, None, Xval, yval


    Xtrain = pd.read_csv('./data/patients_train.csv')
    print Xtrain.info()
    Xtrain.drop(['patient_gender'],axis=1,inplace=True)


    print Xtrain.describe(include='all')
    print "Xtrain.shape:",Xtrain.shape

    Xtest = pd.read_csv('./data/patients_test.csv')
    Xtest.drop(['patient_gender'],axis=1,inplace=True)
    print "Xtest.shape:",Xtest.shape

    #exclude
    if exludePatients:
        print "Excluding patients:"
        do_not_use_train = pd.read_csv('./data/train_patients_to_exclude.csv',header=None,index_col=False,squeeze=True)
        Xtrain = Xtrain.loc[~(Xtrain.patient_id.isin(list(do_not_use_train.values))),:]
        do_not_use_test = pd.read_csv('./data/test_patients_to_exclude.csv',header=None,index_col=False,squeeze=True)
        Xtest = Xtest.loc[~(Xtest.patient_id.isin(do_not_use_test.values)),:]
        print "Xtrain.shape:",Xtrain.shape
        print "Xtest.shape:",Xtest.shape




    print "Xtrain - ISNULL:",Xtrain.isnull().any(axis=0)
    print "Xtest - ISNULL:",Xtest.isnull().any(axis=0)


    if nsamples != -1:
        if isinstance(nsamples, str) and 'shuffle' in nsamples:
            print "Shuffle train data..."
            rows = np.random.choice(len(Xtrain.index), size=len(Xtrain.index), replace=False)
        else:
            rows = np.random.choice(len(Xtrain.index), size=nsamples, replace=False)

        print "unique rows: %6.2f" % (float(np.unique(rows).shape[0]) / float(rows.shape[0]))
        Xtrain = Xtrain.iloc[rows, :]
    print Xtrain.shape

    if createVerticalFeatures:
        print "Creating Sales per Store features..."
        pass


    #Xtrain.groupby([Xtrain.Date.dt.year,Xtrain.Date.dt.month])['Sales'].mean().plot(kind="bar")
    #Xtest.groupby([Xtest.Date.dt.year,Xtest.Date.dt.month])['Store'].mean().plot(kind="bar")

    #rearrange
    ytrain = Xtrain['is_screener']
    Xtrain.drop(['is_screener'],axis=1,inplace=True)
    test_id = Xtest['patient_id']

    Xall = pd.concat([Xtest, Xtrain], ignore_index=True)

    if useActivity:
        print "Reading patient activity..."
        df_count = large['df_count']
        df_count1 = df_count.loc[df_count.activity_type==0,:]
        Xall = pd.merge(Xall, df_count1[['patient_id','activity_year']], on='patient_id', how='left')
        df_count2 = df_count.loc[df_count.activity_type==1,:]
        Xall = pd.merge(Xall, df_count2[['patient_id','activity_year']], on='patient_id', how='left')
        Xall.fillna(0, inplace=True)
        print Xall.head()
        print Xall.info()

        """
        (262158389, 4)
            patient_id activity_type  activity_year  activity_month
        0   103121024             A           2008               5
        1   209481527             R           2009              11
        2   209482911             R           2013               2
        3   209484601             A           2012               2
        4   209485106             A           2012               3
        """

    Xall.drop(['patient_id'],axis=1,inplace=True)

    if dummy_encoding is not None:
        print "Dummy encoding,skip label encoding"
        Xall = pd.get_dummies(Xall,columns=dummy_encoding)

    if labelEncode is not None:
        print "Label encode"
        for col in labelEncode:
            lbl = preprocessing.LabelEncoder()
            Xall[col] = lbl.fit_transform(Xall[col].values)
            vals = Xall[col].unique()
            print "Col: %s Vals %r:"%(col,vals)
            print "Orig:",list(lbl.inverse_transform(Xall[col].unique()))

    if removeRare_freq is not None:
        print "Remove rare features based on frequency..."
        for col in oneHotenc:
            ser = Xall[col]
            counts = ser.value_counts().keys()
            idx = ser.value_counts() > removeRare_freq
            threshold = idx.astype(int).sum()
            print "%s has %d different values before, min freq: %d - threshold %d" % (
                col, len(counts), removeRare_freq, threshold)
            if len(counts) > threshold:
                ser[~ser.isin(counts[:threshold])] = 9999
            if len(counts) <= 1:
                print("Dropping Column %s with %d values" % (col, len(counts)))
                Xall = Xall.drop(col, axis=1)
            else:
                Xall[col] = ser.astype('category')
            print ser.value_counts()
            counts = ser.value_counts().keys()
            print "%s has %d different values after" % (col, len(counts))


    if oneHotenc is not None:
        print "1-0 Encoding categoricals...", oneHotenc
        for col in oneHotenc:
            #print "Unique values for col:", col, " -", np.unique(Xall[col].values)
            encoder = OneHotEncoder()
            X_onehot = pd.DataFrame(encoder.fit_transform(Xall[[col]].values).todense())
            X_onehot.columns = [col + "_" + str(column) for column in X_onehot.columns]
            print "One-hot-encoding of %r...new shape: %r" % (col, X_onehot.shape)
            Xall.drop([col], axis=1, inplace=True)
            Xall = pd.concat([Xall, X_onehot], axis=1)
            print "One-hot-encoding final shape:", Xall.shape
            # raw_input()

    if logtransform is not None:
        print "log Transform"
        for col in logtransform:
            if col in Xall.columns: Xall[col] = Xall[col].map(np.log1p)
            # print Xall[col].describe(include=all)


    if keepFeatures is not None:
        dropcols = [col for col in Xall.columns if col not in keepFeatures]
        for col in dropcols:
            if col in Xall.columns:
                print "Dropping: ", col
                Xall.drop([col], axis=1, inplace=True)

    if dropFeatures is not None:
        for col in dropFeatures:
            if col in Xall.columns:
                print "Dropping: ", col
                Xall.drop([col], axis=1, inplace=True)

    #Xall = Xall.astype(np.float32)
    print "Columns used",list(Xall.columns)


    #split data
    Xtrain = Xall[len(Xtest.index):]
    Xtest = Xall[:len(Xtest.index)]
    Xval = None
    yval = None

    if holdout:
        print "Split holdout..."
        print Xtrain.shape
        #print Xval.shape
        print ytrain.shape
        Xtrain, Xval, ytrain, yval = train_test_split(Xtrain, ytrain, test_size=0.25, random_state=42)

        print "Shape Xtrain:",Xtrain.shape
        print "Shape Xval  :",Xval.shape


    print "Training data:",Xtrain.info()

    store['Xtest'] = Xtest
    store['Xtrain'] = Xtrain
    store['ytrain'] = ytrain
    store['Xval'] = Xval
    store['yval'] = yval
    store['test_id'] = test_id
    print store
    store.close()



    return Xtest, Xtrain, ytrain.values, test_id, None, Xval, yval.values


def mergeWithXval(Xtrain,Xval,ytrain,yval):
    Xtrain =  pd.concat([Xtrain, Xval], ignore_index=True)
    ytrain = np.hstack((ytrain,yval))
    return Xtrain,ytrain


def makePredictions(model=None,Xtest=None,idx=None,filename='submission.csv',scale=0.985):
    print "Saving submission: ", filename
    if model is not None:
        log_preds = model.predict(Xtest)
    else:
        log_preds = Xtest
    if idx is None:
        idx = np.arange(Xtest.shape[0])+1

    if scale>-1:
        print "Scaling predictions: %4.2f"%(scale)
        preds = np.expm1(log_preds)*scale
    else:
        preds = np.expm1(log_preds)

    result = pd.DataFrame({"Id": idx, 'Sales': preds})
    result.to_csv(filename, index=False)


if __name__ == "__main__":
    """
    MAIN PART
    """
    plt.interactive(False)
    #TODO

    t0 = time()

    print "numpy:", np.__version__
    print "pandas:", pd.__version__
    print "scipy:", sp.__version__

    quickload = False
    seed = 51176
    nsamples = -1#'shuffle'
    holdout = True
    dropFeatures = None
    keepFeatures = None
    dummy_encoding = None#['patient_age_group','patient_state','ethinicity','household_income','education_level']
    labelEncode = ['patient_age_group','patient_state','ethinicity','household_income','education_level']
    oneHotenc = ['patient_state','ethinicity']#None#['patient_age_group','patient_state','ethinicity','household_income','education_level']
    removeRare_freq = None
    createVerticalFeatures = True
    logtransform = None
    useActivity = True

    Xtest, Xtrain, ytrain, idx,  sample_weight, Xval, yval = prepareDataset(quickload=quickload,seed=seed, nsamples=nsamples, holdout=holdout,keepFeatures = keepFeatures, dropFeatures= dropFeatures, dummy_encoding=dummy_encoding, labelEncode=labelEncode, oneHotenc= oneHotenc, removeRare_freq=removeRare_freq,  logtransform=logtransform, useActivity= useActivity)

    #interact_analysis(Xtrain)
    #model = sf.RandomForest(n_estimators=120,mtry=5,node_size=5,max_depth=6,n_jobs=2,verbose_level=0)
    #model = Pipeline([('scaler', StandardScaler()), ('model',ross1)])
    #model = RandomForestClassifier(n_estimators=100,max_depth=None,min_samples_leaf=5,n_jobs=2, max_features=Xtrain.shape[1]/3,oob_score=False)
    #model = XgboostClassifier(n_estimators=200,learning_rate=0.4,max_depth=6, NA=0,subsample=.9,colsample_bytree=0.7,min_child_weight=5,n_jobs=1,objective='binary:logistic',eval_metric='logloss',booster='gbtree',silent=1,eval_size=0.0)
    #model = xgb.XGBClassifier(n_estimators=200,learning_rate=0.4)
    #model = ExtraTreesClassifier(n_estimators=250,max_depth=None,min_samples_leaf=1,n_jobs=2, max_features=3*Xtrain.shape[1]/3)
    #model = KNeighborsClassifier(n_neighbors=20)
    #model = LogisticRegression()

    model = KerasNN(dims=Xtrain.shape[1],nb_classes=1,nb_epoch=3,learning_rate=0.02,validation_split=0.0,batch_size=64,verbose=1,loss='mse')
    model = Pipeline([('scaler', StandardScaler()), ('nn',model)])

    cv = StratifiedKFold(ytrain,2,shuffle=True)
    # cv =
    # cv = KFold(X.shape[0], n_folds=folds,shuffle=True)
    #cv = StratifiedShuffleSplit(ytrain,2)
    #scoring_func = roc_auc_score
    #print df.printSummary()
    #parameters = {'n_estimators':[300],'max_depth':[10],'learning_rate':[0.05,0.1,0.2,0.01],'subsample':[0.5],'colsample_bytree':[0.5],'min_child_weight':[1]}
    #parameters = {'nn__nb_epoch':[10,20],'nn__learning_rate':[0.2 ]}
    #parameter={}
    #model = makeGridSearch(model, Xtrain, ytrain, n_jobs=4, refit=True, cv=cv, scoring='roc_auc',parameters=parameters, random_iter=-1)

    #Xtrain, ytrain = mergeWithXval(Xtrain,Xval,ytrain,yval)
    print Xtrain.shape
    #model = buildModel(model,Xtrain,ytrain,cv=cv, scoring='roc_auc', n_jobs=4,trainFull=True,verbose=True)
    #analyzeLearningCurve(model, Xtrain, ytrain, cv=cv, score_func='roc_auc')
    model = buildXvalModel(model,Xtrain,ytrain,sample_weight=None,class_names=None,refit=True,cv=cv)

    model.fit(Xtrain,ytrain)
    yval_pred = model.predict_proba(Xval)[:,1]
    print yval_pred
    print "Eval-score: %5.3f"%(roc_auc_score(yval,yval_pred))

    #print "Training the final model (incl. Xval.)"
    #Xtrain, ytrain = mergeWithXval(Xtrain,Xval,ytrain,yval)
    #model.fit(Xtrain,ytrain)

    #makePredictions(model,Xtest,idx=idx, filename='./submissions/sub30112015.csv')

    plt.show()
    print("Model building done in %fs" % (time() - t0))
