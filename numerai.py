#!/usr/bin/python
# coding: utf-8
from keras.utils.np_utils import probas_to_classes

from qsprLib import *

from interact_analysis import *
from keras_tools import *

import matplotlib.pyplot as plt
import math
import os
pd.options.display.mpl_style = 'default'


from subprocess import call
from tsne import bh_sne


sys.path.append('/home/loschen/calc/smuRF/python_wrapper')
import smurf as sf

def prepareDataset(quickload=False,data_id = 0,store_data = True, append_old=None, seed=123, nsamples=-1, holdout=False, keepFeatures=None, dropFeatures = None,dummy_encoding=None,labelEncode=None, oneHotenc=None,removeRare_freq=None, createVerticalFeatures=None, logtransform = None, polynomialFeatures=None,poly3rdOrder=None, makeDiff=None, makeBins=None, makeTSNE=None,find_clusters=None,removeCor=None,adversarial=None,useDBSCAN=None,dimReduce=None,renameFeatures=None):
    np.random.seed(seed)

    store = pd.HDFStore('./data_numerai/store.h5')

    if not quickload:

        Xtrain = pd.read_csv('/home/loschen/Desktop/datamining-kaggle/numerai/data/numerai_datasets_'+str(data_id)+'/numerai_training_data.csv')
        Xtest =  pd.read_csv('/home/loschen/Desktop/datamining-kaggle/numerai/data/numerai_datasets_'+str(data_id)+'/numerai_tournament_data.csv')



        print Xtrain.info()
        print "Xtrain.shape:",Xtrain.shape

        test_id = Xtest['t_id']
        Xtest.drop(['t_id'],axis=1,inplace=True)
        #Xtest['validation']=-1
        print "Xtest.shape:",Xtest.shape

        if nsamples != -1:
            if isinstance(nsamples, str) and 'shuffle' in nsamples:
                print "Shuffle train data..."
                rows = np.random.choice(len(Xtrain.index), size=len(Xtrain.index), replace=False)
            else:
                rows = np.random.choice(len(Xtrain.index), size=nsamples, replace=False)

            print "unique rows: %6.2f" % (float(np.unique(rows).shape[0]) / float(rows.shape[0]))
            Xtrain = Xtrain.iloc[rows, :]


        ytrain = Xtrain['target']
        Xtrain.drop(['target'],axis=1,inplace=True)






    else:
        print "Loading previous dataset..."
        Xtrain = store['Xtrain']
        ytrain= store['ytrain']
        Xtest = store['Xtest']
        test_id = store['test_id']
        return Xtest, Xtrain, ytrain.values, test_id, None, None, None

    #print "Xtrain - ISNULL:",Xtrain.isnull().any(axis=0)
    #print "Xtest - ISNULL:",Xtest.isnull().any(axis=0)


    Xall = pd.concat([Xtest, Xtrain], ignore_index=True)

    if renameFeatures is not None:
        print "Renaming features!"
        corr_train = pd.DataFrame(Xall).corr()
        sns.set(context="paper", font="monospace")
        m = sns.clustermap(corr_train)
        new_cols = m.data2d.columns
        Xall = Xall[new_cols]
        print "New feature order:",Xall.columns
        print "Renaming..."
        Xall.columns = ['feature'+str(x+1) for x in xrange(Xall.shape[1])]

    if append_old is not None:
            for dset in append_old:
                print "Adding old data:"+str(dset)
                #Xtrain2 = pd.read_csv('/home/loschen/Desktop/datamining-kaggle/numerai/data/'+dset+'/numerai_training_data.csv')
                _, Xtrain_old, ytrain_old, _,  _ , _ , _ = prepareDataset(data_id=dset,renameFeatures=True,adversarial=None,store_data=False)

                print type(ytrain)
                print type(ytrain_old)

                Xtrain = Xall[len(Xtest.index):]
                Xtest = Xall[:len(Xtest.index)]

                Xtrain = pd.concat([Xtrain, Xtrain_old], ignore_index=True)
                ytrain = pd.concat([ytrain,pd.Series(ytrain_old)],ignore_index=True)
                #ytrain =np.concatenate((ytrain.values,ytrain_old))
                print Xtrain.shape
                print ytrain.shape

                Xall = pd.concat([Xtest, Xtrain], ignore_index=True)



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

    if useDBSCAN is not None:
        print "DBSCan..."
        db = DBSCAN(eps=0.7, min_samples=5)
        db.fit(Xall.values)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        noise_labels = labels == -1
        print "n noise:",noise_labels.sum()
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        print "n clusters:",n_clusters_
        print noise_labels
        print noise_labels.shape
        #Xall['noise'] = noise_labels
        Xall['clusters'] = labels
        grouped = Xall.groupby('clusters')


    if find_clusters is not None:
        Xf1 = Xall.values.T
        est1 = KMeans(n_clusters=7, n_jobs=4)
        est1.fit(Xf1)
        labels1 = est1.labels_
        clustered1 = []
        for i in xrange(7):
            clustered1.append(list(Xall.columns[labels1==i].values))
        print clustered1
        createVerticalFeatures = clustered1


    if makeBins is not None:
        Xall = data_binning(Xall, makeBins)
        print Xall.head(10)

    if createVerticalFeatures is not None:
        print "Creating vert features..."
        for i,fgroup in enumerate(createVerticalFeatures):
            print fgroup
            m = Xall[fgroup].mean(axis=1)
            Xall['fgroup_'+str(i)+'_mean'] = m
            sd = Xall[fgroup].std(axis=1)
            Xall['fgroup_'+str(i)+'_std'] = sd
            #md = Xall[fgroup].median(axis=1)
            #Xall['fgroup_'+str(i)+'_median'] = md

            #k = Xall[fgroup].kurtosis(axis=1)
            #Xall['fgroup_'+str(i)+'_kurt'] = k

            #Xall.drop(fgroup,axis=1,inplace=True)


    if adversarial is not None:
        print "Selecting training instances..."
        #train for train / test similarity

        Xtrain = Xall[len(Xtest.index):]
        Xtest = Xall[:len(Xtest.index)]

        Xtest['test'] = 1
        Xtrain['test'] = 0
        Xall = pd.concat([Xtest, Xtrain], ignore_index=True)
        ytemp = Xall['test']
        Xall.drop(['test'],axis=1,inplace=True)

        #model = LogisticRegression(C=1.0,penalty='l2')
        model = RandomForestClassifier(n_estimators=100)
        #model = XgboostClassifier(n_estimators=100,learning_rate=0.01,max_depth=2, NA=0,subsample=.5,colsample_bytree=1.0,min_child_weight=5,n_jobs=4,objective='binary:logistic',eval_metric='logloss',booster='gbtree',silent=1,eval_size=0.0)
        #Xall_train = Xall.iloc[ np.random.permutation(len( Xall )) ]
        #buildModel(model,Xall,ytemp,cv=StratifiedShuffleSplit(ytemp,n_iter=5,test_size=0.2), scoring='accuracy', n_jobs=1,trainFull=False,verbose=True)

        model.fit(Xall,ytemp)
        Xall['sim'] = model.predict_proba(Xall)[:,1]

        Xtrain = Xall[len(Xtest.index):]
        Xtest = Xall[:len(Xtest.index)]

        Xtrain['sim'].hist(bins=30)
        Xtest['sim'].hist(bins=30)
        plt.draw()
        train_mask = (Xtrain['sim']>0.1).values # PBLL = 0.69579  # 26000 samples
        #train_mask = (Xtrain['sim']<0.5).values # PBLL = 0.69175 # 70136 samples

        Xtrain = Xtrain.loc[train_mask,:]
        ytrain = ytrain[train_mask]

        print "New shape:",Xtrain.shape

        Xall = pd.concat([Xtest, Xtrain], ignore_index=True)
        Xall.drop(['sim'],axis=1,inplace=True)

    if polynomialFeatures is not None:
        quadratic = True
        if isinstance(polynomialFeatures, str) and 'all' in polynomialFeatures:
            polynomialFeatures = Xall.columns
        elif isinstance(polynomialFeatures, str) and 'fgroup' in polynomialFeatures:
            polynomialFeatures = [ x for x in Xall.columns if x.startswith('fgroup')]

        print "Polynomial feature of degree:", polynomialFeatures
        if isinstance(polynomialFeatures, str) and 'load' in polynomialFeatures:
            print "Loading polynomials..."
            X_poly = pd.read_csv('poly.csv').reset_index(drop=True)
            print X_poly.describe()

        #grouped
        else:
            if isinstance(polynomialFeatures[0], list):
                for el in polynomialFeatures:
                    X_poly = make_polynomials(Xall[el],degree=2,cutoff=100,quadratic=quadratic)
                    print X_poly.head()
                    Xall = pd.concat([Xall, X_poly], axis=1)

            else:
                X_poly = make_polynomials(Xall[polynomialFeatures],degree=2,cutoff=100,quadratic=quadratic)
                X_poly.to_csv('poly.csv')

                Xall = pd.concat([Xall, X_poly], axis=1)


    if oneHotenc is not None:
        print "1-0 Encoding categoricals...", oneHotenc
        if oneHotenc: oneHotenc = Xall.columns
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

    if makeDiff is not None:
        X_diff = differentiateFeatures(Xall.iloc[:,:])
        if '2nd' in makeDiff:
            X_diff = differentiateFeatures(X_diff)
        Xall = pd.concat([Xall, X_diff],axis=1)
        Xall = Xall.ix[:,(Xall != 0).any(axis=0)]

    if logtransform is not None:
        print "log Transform"
        for col in logtransform:
            if col in Xall.columns: Xall[col] = Xall[col].map(np.log1p)
            # print Xall[col].describe(include=all)



    if keepFeatures is not None:
        dropcols = [col for col in Xall.columns if col not in keepFeatures]
        for col in dropcols:
            #if col in Xall.columns and not 'validation' in col:
                print "Dropping: ", col
                Xall.drop([col], axis=1, inplace=True)

    if dropFeatures is not None:
        for col in dropFeatures:
            if col in Xall.columns:
                print "Dropping: ", col
                Xall.drop([col], axis=1, inplace=True)

    if makeTSNE is not None:
        print "Making tsne ..."
        if makeTSNE.has_key('uselib') and 'bh_sne' in makeTSNE['uselib']:
            print "Using bh_sne..."
            Xall = bh_sne(Xall)
            plt.scatter(Xall[:, 0], Xall[:, 1])
            Xall = pd.DataFrame(Xall)

        else:
            Xall.to_csv('pure.csv',header=False,index=False,sep='\t')
            numDims = makeTSNE['numDims']
            pcaDims = makeTSNE['pcaDims']
            perplexity = makeTSNE['perplexity']
            theta = makeTSNE['theta']
            #alg = 'svd'
            if numDims<4:
                call(("/home/loschen/programs/bhtsne/bhtsne.py -r 42 -v -d "+str(numDims)+" -p "+str(perplexity)+" -t "+str(theta)+" -n "+str(pcaDims)+" -i pure.csv -o tsne.csv").split())
            else:
                call(("/home/loschen/programs/bhtsne_serial/bhtsne.py -r 42 -v -d "+str(numDims)+" -p "+str(perplexity)+" -t "+str(theta)+" -n "+str(pcaDims)+" -i pure.csv -o tsne.csv").split())

            Xall = pd.read_csv('tsne.csv',sep='\t',header=None)

        Xall.columns = ['d'+str(i) for i in xrange(Xall.shape[1])]

        #print Xall.info()
        #print Xall.head()


    if poly3rdOrder is not None:
        print "Again: Polynomial feature of degree:", poly3rdOrder
        if isinstance(poly3rdOrder, str) and 'load' in poly3rdOrder:
            print "Loading polynomials..."
            X_poly = pd.read_csv('poly2.csv').reset_index(drop=True)
            print X_poly.describe()
        else:
            X_poly = make_polynomials(Xall[poly3rdOrder],degree=2,cutoff=100,quadratic=False)
            X_poly.to_csv('poly2.csv')

        print X_poly.head()
        #drop duplicates
        #Xall.drop(['feature16xfeature16'],inplace=True)
        duplicates = list(set(Xall.columns) & set(X_poly.columns))
        print "Duplicates:",duplicates
        X_poly.drop(duplicates,axis=1,inplace=True)
        Xall = pd.concat([Xall, X_poly], axis=1)
        print Xall.shape

        #Xall = Xall.T.drop_duplicates().T

    if removeCor is not None:
        print "Removing correlations..."
        Xall = removeCorrelations(Xall, threshhold=0.95)


    if dimReduce is not None:
        print "Reducing dimensions!"
        if isinstance(dimReduce,int):
            reducer = TruncatedSVD(n_components=dimReduce)
        else:
            reducer = dimReduce
        print reducer
        Xall = pd.DataFrame(reducer.fit_transform(Xall))
        Xall.columns = ["d_" + str(column) for column in xrange(Xall.shape[1])]


    #split data
    Xall = Xall.astype(np.float32)
    Xtrain = Xall[len(Xtest.index):]
    Xtest = Xall[:len(Xtest.index)]
    Xval = None
    yval = None

    if holdout:
        print "Split holdout..."
        print Xtrain.shape
        #print Xval.shape
        print ytrain.shape
        #Xtrain, Xval, ytrain, yval = train_test_split(Xtrain, ytrain, test_size=0.25, random_state=42)
        train_mask = (Xtrain.validation == 0).values
        val_mask = (Xtrain.validation == 1).values


        Xval = Xtrain.loc[val_mask,:]
        yval = ytrain[val_mask].values

        Xtrain = Xtrain.loc[train_mask,:]
        ytrain = ytrain[train_mask]

        Xval.drop(['validation'],axis=1,inplace=True)
        store['Xval'] = Xval
        store['yval'] = yval
        print "Shape Xtrain:",Xtrain.shape
        print "Shape Xval  :",Xval.shape

    #Xtrain.drop(['validation'],axis=1,inplace=True)
    #Xtest.drop(['validation'],axis=1,inplace=True)

    #print "Training data:",Xtrain.info()
    #print "Test data:",Xtest.info()

    if store_data:
        store['Xtest'] = Xtest
        store['Xtrain'] = Xtrain
        store['ytrain'] = ytrain

        store['test_id'] = test_id
        print store

        store.close()



    return Xtest, Xtrain, ytrain.values, test_id, None, Xval, yval


def mergeWithXval(Xtrain,Xval,ytrain,yval):
    Xtrain =  pd.concat([Xtrain, Xval], ignore_index=True)
    ytrain = np.hstack((ytrain,yval))
    return Xtrain,ytrain


def makePredictions(model=None,Xtest=None,idx=None,filename='submission.csv'):
    print "Saving submission: ", filename
    if model is not None:
        if not hasattr(model,'predict_proba'):
            preds = model.predict(Xtest)
        else:
            preds = model.predict_proba(Xtest)[:,1]
    else:
        preds = Xtest
    if idx is None:
        idx = np.arange(Xtest.shape[0])+1

    result = pd.DataFrame({"t_id": idx, 'probability': preds})
    result.to_csv(filename, index=False)


def train_alldata():
    # http://tiny.cc/numerai_datasets
    datasets = [16]
    model = KerasNN(dims=21,nb_classes=2,nb_epoch=60,learning_rate=0.1,validation_split=0.0,batch_size=1024,verbose=1,activation='relu', layers=[20,20], dropout=[0.2],loss='categorical_crossentropy')
    for ds in datasets:
        Xtest, Xtrain, ytrain, idx,  sample_weight, Xval, yval = prepareDataset(data_id=ds)
        model.fit(Xtrain,ytrain)
        model.save_model('nn_model_ds'+str(ds)+'.h5')
        #Xtest, Xtrain, ytrain, idx,  sample_weight, Xval, yval = prepareDataset(data_id=ds)
        #model.fit(Xtrain,ytrain)
        model.load_model('nn_model_ds'+str(ds)+'.h5')

    Xtest, Xtrain, ytrain, idx,  sample_weight, Xval, yval = prepareDataset(data_id=17)
    #model = KerasNN(dims=Xtrain.shape[1],nb_classes=2,nb_epoch=40,learning_rate=0.1,validation_split=0.0,batch_size=1024,verbose=1,activation='relu', layers=[20,20], dropout=[0.2],loss='categorical_crossentropy')
    model.fit(Xtrain,ytrain)
    makePredictions(model,Xtest,idx=idx, filename='./submissions/numerai_01sept_cascacde.csv')
    sys.exit(0)


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

    numericals = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8', 'feature9', 'feature10', 'feature11', 'feature12', 'feature13', 'feature14', 'feature15','feature16','feature17', 'feature18','feature19','feature20','feature21']
    #rfecv_fetures =

    #rfecv_fetures2 =

    #rfecv_vertical =


    #LR C=100 penalty='l1' same as l2 LL=0.69168
    #greedy_fw18 = ['fgroup_3_medianxfgroup_6_median', 'fgroup_1_meanxfgroup_2_median', 'fgroup_0_meanxfgroup_1_std', 'fgroup_0_stdxfgroup_2_std', 'fgroup_2_meanxfgroup_6_std', 'fgroup_0_stdxfgroup_2_mean', 'fgroup_4_medianxfgroup_4_median', 'fgroup_2_stdxfgroup_2_std', 'fgroup_0_stdxfgroup_5_mean', 'fgroup_2_stdxfgroup_4_mean', 'fgroup_2_stdxfgroup_6_std', 'fgroup_0_meanxfgroup_6_median', 'fgroup_3_medianxfgroup_4_mean', 'fgroup_2_stdxfgroup_3_std']

    #LR C=1 penalty='l2' LL=0.6914
    #greedy_fw18b = ['feature14xfeature19', 'feature11xfeature20', 'feature1xfeature9', 'feature8xfeature20', 'feature1xfeature12', 'feature18xfeature21', 'feature11xfeature12', 'feature14xfeature17', 'feature2xfeature21', 'feature5xfeature21', 'feature11xfeature14', 'feature7xfeature10', 'feature12xfeature21']

    greedy_fw18 = ['feature2xfeature21', 'feature10xfeature17', 'feature7xfeature20', 'feature14xfeature18', 'feature2xfeature6', 'feature2xfeature5', 'feature6xfeature14', 'feature2xfeature4', 'feature1xfeature10', 'feature1xfeature4', 'feature10xfeature10']
    greedy_fw19 = ['feature10xfeature18', 'feature2xfeature20', 'feature4xfeature5', 'feature10xfeature17', 'feature14xfeature18', 'feature4xfeature10', 'feature18xfeature18', 'feature1xfeature13', 'feature2xfeature7', 'feature1xfeature8', 'feature15xfeature18']
    greedy_fw20 = ['feature18', 'feature3', 'feature21', 'feature10', 'feature7', 'feature2', 'feature5', 'feature14', 'feature1', 'feature17', 'feature11', 'feature4', 'feature12', 'feature6', 'feature16', 'feature8', 'feature15']

    #before: 14 & 15:  19 1 8, 13 15 16, 5,7,20, 6,4,14, 12,7,9, 2,3,18 21,10,11

    #since round 16
    #cluster1 = []
    #cluster2 = []
    #cluster3 = []
    #cluster4 = []
    #cluster5 = []
    #cluster6 = []
    #cluster7 = []

    #clustered = [cluster1,cluster2,cluster3,cluster4,cluster5,cluster6,cluster7]

    #cluster_stat2 =

    #cluster_stat =

    clustered_mean = [u'fgroup_0_mean', u'fgroup_1_mean', u'fgroup_2_mean', u'fgroup_3_mean', u'fgroup_4_mean', u'fgroup_5_mean', u'fgroup_6_mean']

    quickload = False
    data_id = 21
    append_old = [20]#['August3']
    seed = 421
    nsamples = -1
    renameFeatures = True
    holdout = False
    makeDiff = None #'1st'
    makeBins = None
    makeTSNE = None #{'uselib': 'bh_sne'} #None #tsnelib= {'numDims': 2, 'perplexity': 50, 'theta':0.5, 'pcaDims':21}
    dummy_encoding = None# ['c1']
    adversarial = None
    useDBSCAN = None
    dimReduce= None #TruncatedSVD(n_components=30)
    labelEncode = None#['c1']
    oneHotenc = None#['c1']
    removeCor = True
    removeRare_freq = None
    find_clusters = None
    createVerticalFeatures = None #clustered#True
    logtransform = None#numericals
    polynomialFeatures = None #'all' #'fgroup' # [,cluster_stat]#None#clustered_mean#clustered_mean#None #numericals#numericals
    keepFeatures = None #greedy_fw20 #greedy_fw18b # rfecv_fetures2 + rfecv_vertical #clustered #rfecv_fetures2 # # greedy6
    dropFeatures = None
    poly3rdOrder = None
    evaluate_old = None

    #train_alldata()

    Xtest, Xtrain, ytrain, idx,  sample_weight, Xval, yval = prepareDataset(quickload=quickload,data_id=data_id,append_old = append_old, seed=seed, nsamples=nsamples, holdout=holdout,keepFeatures = keepFeatures, dropFeatures= dropFeatures, dummy_encoding=dummy_encoding, labelEncode=labelEncode, oneHotenc= oneHotenc, removeRare_freq=removeRare_freq,  logtransform=logtransform, createVerticalFeatures=createVerticalFeatures,polynomialFeatures=polynomialFeatures,poly3rdOrder=poly3rdOrder, makeDiff=makeDiff, makeBins=makeBins, makeTSNE=makeTSNE, find_clusters=find_clusters,removeCor=removeCor,adversarial=adversarial,useDBSCAN=useDBSCAN,dimReduce=dimReduce,renameFeatures=renameFeatures)
    print list(Xtrain.columns)
    #Xtrain.iloc[:10,:-1].plot(kind='area',stacked=False)
    #Xtrain.hist(bins=100)
    plt.show()
    #interact_analysis(Xtrain)

    #TODO check different lossfunctions, classweight
    #sklearn.linear_model.HuberRegressor -> SVM -Ensemble
    # Calibration ofprobs!!!
    # make diff with older set
    # automatically cluster dataset
    # make training set more similar to test set!!!
    #http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#example-cluster-plot-dbscan-py
    #https://github.com/danielfrg/tsne

    ###########################################################
    #class prior: [ 0.49482973  0.50517027] log_loss = 0.693
    ###########################################################

    #cv = StratifiedKFold(ytrain,8,shuffle=True)
    cv = StratifiedShuffleSplit(ytrain,n_iter=20,test_size=0.2)

    #model = RandomForestClassifier(n_estimators=100,max_depth=None,min_samples_leaf=5,n_jobs=2, max_features=Xtrain.shape[1]/3,oob_score=False)
    #model = SVC(C=1,kernel='rbf',probability=True) #cv 8fold ~ 80 min on 4 procs!!! 20000 samples ~ 2min. per fold 40000 12 min.
    #model = LogisticRegression(C=1.0,penalty='l2')
    #model = LogisticRegression(C=100,penalty='l1')
    #model = KernelRidge()
    #model = XgboostClassifier(n_estimators=200,learning_rate=0.01,max_depth=2, NA=0,subsample=.5,colsample_bytree=1.0,min_child_weight=5,n_jobs=4,objective='binary:logistic',eval_metric='logloss',booster='gbtree',silent=1,eval_size=0.0)
    model = KerasNN(dims=Xtrain.shape[1],nb_classes=2,nb_epoch=40,learning_rate=0.1,validation_split=0.0,batch_size=1024,verbose=1,activation='relu', layers=[20,20], dropout=[0.2,0.2],loss='categorical_crossentropy')
    #model = VotingClassifier(estimators=[('lr', model1),('xgb', model2) ,('nn', model3)], voting='soft',weights=[1,1,1])
    #model = KerasNN(dims=Xtrain.shape[1],nb_classes=2,nb_epoch=40,learning_rate=0.05,validation_split=0.0,batch_size=64,verbose=1,activation='sigmoid', layers=[20,20], dropout=[0.0,0.1],loss='categorical_crossentropy')
    #model = CalibratedClassifierCV(model,cv=8,method='sigmoid')
    model = Pipeline([('scaler', StandardScaler()), ('m',model)])
    #model = Pipeline([('pca', PCA(n_components=5,whiten=True)),('m', model)])
    #model = Pipeline([('kmeans', MiniBatchKMeans(n_clusters=7)),('m', model)]) #->nothing
    #model = Pipeline([('pca', TruncatedSVD(n_components=5)),('m', model)])

    #model = KNeighborsClassifier(n_neighbors=5) # NO
    #model = BernoulliNB()

    #model = LinearRegression()
    #model = DummyClassifier(strategy='prior')
    #model = SVC(C=1,kernel='linear',probability=True)
    #model = Pipeline([('scaler', StandardScaler()),('filter', GenericUnivariateSelect(f_regression, param=60,mode='percentile')), ('model', model)])
    #model = BaggingRegressor(base_estimator=model,n_estimators=20,n_jobs=1,verbose=0,random_state=None,max_samples=0.9,max_features=0.9,bootstrap=False)
    #model = BaggingClassifier(base_estimator=model,n_estimators=5,n_jobs=4,verbose=0,random_state=None,max_samples=0.5,max_features=1.0,bootstrap=False)

    #model = Pipeline([('scaler', StandardScaler()),('filter', GenericUnivariateSelect(f_regression, param=95,mode='percentile')), ('model', SVC(C=1))])
    #model = Pipeline([('pca', PCA(n_components=10)),('model', Ridge())])

    #model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=99,mode='percentile')),('model', ElasticNet(alpha=.01,l1_ratio=0.001,max_iter=1000)) ])
    #model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=100,mode='percentile')), ('model', PLSRegression(n_components=20))])


    #model = LinearSVC()
    #model = GaussianNB()
    #model = BaggingClassifier(base_estimator=model,n_estimators=10,n_jobs=1,verbose=2,random_state=None,max_samples=0.9,max_features=0.9,bootstrap=True)
    #
    parameters = {}
    #parameters = {'weights':[[2,1]],'voting':['soft'],'xgb__n_estimators':[100,150,200],'lr__C':[10,1.0,0.1]}
    #parameters = {'n_estimators':[500,1000],'max_depth':[1,2],'learning_rate':[0.01],'subsample':[0.75],'colsample_bytree':[0.75],'min_child_weight':[5]}
    #parameters = {'n_estimators':[150],'min_samples_leaf':[5,10,15],'max_features':[100],'criterion':['entropy']}
    #parameters = {'m__nb_epoch':[10,20,30,40],'m__learning_rate':[0.2,0.02]}
    #parameters = {'m__n_neighbors':[5]}
    #parameters = {'filter__param':[99,100],'model__C':[1,1E-3,1E-5,1E-7],'model__penalty':['l2']}
    #parameters = {'m__learning_rate':[0.05,0.01],'m__layers':[[500,500],[100,100]],'m__dropout':[[0.5,0.5]],'m__batch_size':[512,256]}
    #parameters = {'m__C':[100,10,1,0.1],'m__penalty':['l2','l1']}
    #parameters = {'m__C':[0.01,0.001],'m__gamma':['auto']}
    #parameters = {'pca__n_components':[10,15,19,20],'m__C':[1.0,10.0]}
    #parameters = {'kmeans__n_clusters':[4,6,7,10,15,20],'m__C':[1.0,10,0.1]}
    #mask = recursive_featureselection(model,Xtrain,ytrain,cv=cv,step=1,scoring='log_loss')
    #Xtrain = Xtrain.loc[:,mask]
    #Xtest = Xtest.loc[:,mask]
    print Xtrain.columns
    #model = makeGridSearch(model, Xtrain, ytrain, n_jobs=1, refit=True, cv=cv, scoring='log_loss',parameters=parameters, random_iter=-1)
    #greedyFeatureSelection(model, Xtrain, ytrain, itermax=30, itermin=20, pool_features=None, start_features=[],verbose=True, cv=cv, n_jobs=4, scoring_func='log_loss')
    #model.load_model("August3.h5")
    #Xtrain, ytrain = mergeWithXval(Xtrain,Xval,ytrain,yval)
    model = buildModel(model,Xtrain,ytrain,cv=cv, scoring='log_loss', n_jobs=1,trainFull=True,verbose=True)
    #model = buildXvalModel(model,Xtrain,ytrain,sample_weight=None,class_names=None,refit=True,cv=cv)
    #model.save_model("August3.h5")

    #analyzeLearningCurve(model, Xtrain, ytrain, cv=cv, score_func='log_loss',train_sizes=np.linspace(.1, 1.0, 10),ylim=(0.68, 0.7))
    print type(Xtrain)
    print type(ytrain)
    model.fit(Xtrain.values,ytrain)

    if holdout:
        if not hasattr(model,'predict_proba'):
            yval_pred = model.predict(Xval)
        else:
            yval_pred = model.predict_proba(Xval)[:,1]

        print "Eval-score: %6.4f"%(roc_auc_score(yval,yval_pred))
        Xtrain, ytrain = mergeWithXval(Xtrain,Xval,ytrain,yval)

        print "Training the final model (incl. Xval.)"

        #ytrain_pred = model.fit(Xtrain,ytrain)

    #print model.class_prior_

    ytrain_pred = model.predict_proba(Xtrain.values)[:,1]
    print "Training-score: %6.4f"%(log_loss(ytrain,ytrain_pred))


    makePredictions(model,Xtest,idx=idx, filename='./submissions/numerai_14septa.csv')

    plt.show()
    print("Model building done in %fs" % (time() - t0))

    if evaluate_old is not None:
        _, Xval, yval, idx,  sample_weight, _, _ = prepareDataset(quickload=quickload,data_id=evaluate_old,append_old = append_old, seed=seed, nsamples=nsamples, holdout=holdout,keepFeatures = keepFeatures, dropFeatures= dropFeatures, dummy_encoding=dummy_encoding, labelEncode=labelEncode, oneHotenc= oneHotenc, removeRare_freq=removeRare_freq,  logtransform=logtransform, createVerticalFeatures=createVerticalFeatures,polynomialFeatures=polynomialFeatures,poly3rdOrder=poly3rdOrder, makeDiff=makeDiff, makeBins=makeBins, makeTSNE=makeTSNE, find_clusters=find_clusters,removeCor=removeCor,adversarial=adversarial,useDBSCAN=useDBSCAN,dimReduce=dimReduce,renameFeatures=renameFeatures)
        yval_pred = model.predict_proba(Xval)[:,1]
        print "Eval-score: %6.4f"%(log_loss(yval,yval_pred))



    """
     RoundAvg. Private LoglossAvg. Public/Private Difference  Public     LocalCV     Model         Features
     Round 10   0.69183                              0.00054
     Round 11   0.69183                              0.00035
     Round 12   0.69328                              0.00030
     Round 13   0.69230                                        0.69084    0.6918      LogLoss(C=10) greedyfeatureselection,polynomial,1stdiff
     Round 14   0.69182                                        0.69121    0.6915      xgboost / LR with 1st derivative voting classifier with 2:1 weight
     Roubd 15   $0.35                                          0.69131    0.6917      xgboost / LR with 1st derivative voting classifier with 2:1 weight + rfcev2
     Round 16   $0.05                                          0.69050    0.6917      xgboost / LR with 1st derivative voting classifier with 2:1 weight + rfcev2 - retuned
     Round 17   $0.46                                          0.69123    0.6915      Simple NNet - trained with R16 and R17 dataset
     Round 18   $1.31                                          0.69168    0.6911      Ensemble ['nn1', 'lr3', 'lr5']
     Round 19   $0.00                                          0.69177    0.6913      Ensemble ['nn1', 'lr3g']
     Round 20   $0.01                                          0.69274    0.6912      Simple NN with renameFeatures and old datasets 19
     Round 211  $0.00                                          0.69223    0.6912      Ensemble ['nn1', 'lr3', 'lr5','nn3'] with Logistic regression

    """