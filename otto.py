#!/usr/bin/python 
# coding: utf-8

"""
Otto product classification 
"""


from qsprLib import *
import pandas as pd
from sklearn import preprocessing
from sklearn.lda import LDA
from sklearn.qda import QDA
from pandas.tools.plotting import scatter_matrix

from xgboost_sklearn import *
from OneHotEncoder import *

import theano
from lasagne import layers
from lasagne import nonlinearities
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator


#from nolearn.lasagne import l2


def analyzeDataset(Xtrain,Xtest,ytrain):
    plt.hist(ytrain,bins=9)
    plt.show()
    #Xtrain.iloc[:5].hist(color='b', alpha=0.5, bins=50)
    Xtrain.iloc[:,:30].hist(color='b', alpha=1.0, bins=20)
    #scatter_matrix(Xtrain.iloc[:5], alpha=0.2, figsize=(6, 6), diagonal='hist')
    plt.show()
    #pcAnalysis(Xtrain,Xtest,None,None,ncomp=2,transform=False,classification=False)
    #for col in Xtrain.columns:
#	print "Column:",col
#	print Xtrain[col].describe()
#	raw_input()


def prepareDataset(nsamples=-1,standardize=False,featureHashing=False,polynomialFeatures=None,OneHotEncoding=False,featureFilter=None,call_group_data=False,addNoiseColumns=None,log_transform=None,addFeatures=False,doSVD=False,analyzeIt=False):
  # import data
  Xtrain = pd.read_csv('/home/loschen/Desktop/datamining-kaggle/otto/train.csv')
  Xtest = pd.read_csv('/home/loschen/Desktop/datamining-kaggle/otto/test.csv')
  #sample = pd.read_csv('/home/loschen/Desktop/datamining-kaggle/otto/sampleSubmission.csv')

  

  # drop ids and get labels
  labels = Xtrain.target.values
  Xtrain = Xtrain.drop('id', axis=1)
  Xtrain = Xtrain.drop('target', axis=1).astype(np.int32)
  Xtest = Xtest.drop('id', axis=1).astype(np.int32)
  
  ytrain = preprocessing.LabelEncoder().fit_transform(labels)
  
  if nsamples != -1: 
      rows = np.random.randint(0,len(Xtrain.index), nsamples)
      Xtrain = Xtrain.iloc[rows,:]
      ytrain = ytrain[rows]
  
  #Xtrain = Xall[len(Xtest.index):]
  #Xtest = Xall[:len(Xtest.index)]
  
  # encode labels 
  
  if analyzeIt:
    analyzeDataset(Xtrain,Xtest,ytrain)
    sys.exit(1)
   
   
  Xall = pd.concat([Xtest, Xtrain],ignore_index=True)
  
  if featureFilter is not None:
	print "Using featurefilter..."
	Xall=Xall[featureFilter]
  
  if call_group_data:
	print "Group data...",
	print density(Xall)	
	#Xall=pd.DataFrame(group_data2(Xall.values))
	Xall=pd.DataFrame(group_data(Xall))
	#Xall=sparse.csc_matrix(Xall.values)
	print "...new shape:",Xall.shape
	print Xall.describe()
	print density(Xall)

  if polynomialFeatures is not None and polynomialFeatures is not False:
      print "Polynomial feature of degree:",polynomialFeatures,
      X_poly = make_polynomials2(Xall)
      Xall = pd.concat([Xall, X_poly],axis=1)
      #print Xall.describe()
      print "...",Xall.shape
  
  if doSVD: 
      print "SVD..."
      tsvd=TruncatedSVD(n_components=25, algorithm='randomized', n_iter=5, tol=0.0)
      #tsvd = RandomizedPCA(n_components=25, whiten=False)
      Xall=tsvd.fit_transform(Xall)
      Xall = pd.DataFrame(Xall)
            
      #df_info(Xall)
  
  if OneHotEncoding:
      #Xtrain = Xall[len(Xtest.index):].values
      #Xtest = Xall[:len(Xtest.index)].values
      encoder = OneHotEncoder()    
      Xall_sparse = encoder.fit_transform(Xall)
      Xall = Xall_sparse

      print "One-hot-encoding...new shape:",Xall.shape
      print type(Xall)
      Xall = Xall.tocsr()
      print density(Xall)
  
  if featureHashing:
      #Xtrain = Xall[len(Xtest.index):]
      #Xtest = Xall[:len(Xtest.index)]
      print "Feature hashing...",#Feature hashing not necessary
      encoder = FeatureHasher(n_features=2**10,dtype=np.int32)
      print encoder
      #encoder = DictVectorizer()#basically one-hot-encoding
      #encoder = OneHotEncoder()
      all_as_dicts = [dict(row.iteritems()) for _, row in Xall.iterrows()]
      #all_as_dicts = [dict(row.iteritems()) for row in Xall.values]
      #print train_as_dicts
      #train_as_dicts = [dict(r.iteritems()) for _, r in Xtrain.iterrows()]  #feature hasher
      Xall_sparse = encoder.fit_transform(all_as_dicts)
      #test_as_dicts = [dict(r.iteritems()) for _, r in Xtest.applymap(str).iterrows()]
      #test_as_dicts = [dict(r.iteritems()) for _, r in Xtest.iterrows()]#feature hasher
      #Xtest_sparse = encoder.transform(test_as_dicts)
      print type(Xall_sparse)
      Xall = Xall_sparse
      #Xtest = Xtest_sparse
      
      #Xall = np.vstack((Xtest,Xtrain))
      print "...new shape:",Xall.shape
      print density(Xall)      
  
  
  if addNoiseColumns is False or addNoiseColumns is not None:
	Xrnd = pd.DataFrame(np.random.randn(Xall.shape[0],addNoiseColumns))
	#print "Xrnd:",Xrnd.shape
	#print Xrnd
	for col in Xrnd.columns:
	    Xrnd=Xrnd.rename(columns = {col:'rnd'+str(col+1)})
	
	Xall = pd.concat([Xall, Xrnd],axis=1)
  
  if addFeatures:
      print "Additional columns"
      #Xall['row_sum'] = Xall.sum(axis=1)
      #Xall['row_median'] = Xall.median(axis=1)
      #Xall['row_max'] = Xall.max(axis=1)
      #Xall['row_min'] = Xall.max(axis=1)
      #Xall['row_mean'] = Xall.mean(axis=1)
      #Xall['row_kurtosis'] = Xall.kurtosis(axis=1)
      Xall_orig = Xall.copy()
      Xall['arg_max'] = pd.DataFrame(Xall_orig.values).idxmax(axis=1)
      #print Xall['arg_max']
      Xall['arg_min'] = pd.DataFrame(Xall_orig.values).idxmin(axis=1)
      #print Xall['arg_min']
      Xall['non_null'] = (Xall_orig != 0).astype(int).sum(axis=1)
      Xall['row_sd'] = Xall_orig.std(axis=1)
      #Xall['row_prod'] = Xall.prod(axis=1)
      #Xall['feat_11+feat_60'] = (Xall['feat_11'] +Xall['feat_60'])/2
      #Xall['feat_11xfeat_60'] = (Xall['feat_11'] *Xall['feat_60'])/2

      #print Xall.loc[:,['non-null']].describe()
      print Xall.iloc[:,-10:].describe()
      #print Xall.loc[1:5,['sum_counts']]
      #raw_input()
	
  
  if log_transform:
	print "log_transform"
	Xall = Xall + 1.0
	Xall=Xall.apply(np.log)
	#Xall=Xall.apply(np.log)
	print Xall.describe()
  
  if standardize:
    Xall = scaleData(lXs=Xall,lXs_test=None)
  

  if isinstance(Xall,pd.DataFrame): Xall = removeLowVariance(Xall,1E-1)
  #Xall = removeCorrelations(Xall,0.99)
  
  
  Xtrain = Xall[len(Xtest.index):].astype(np.float32)
  Xtest = Xall[:len(Xtest.index)].astype(np.float32)
  ytrain = ytrain.astype(np.int32)
  
  #analyzeDataset(Xtrain,Xtest,ytrain)
  
  print "#Xtrain:",Xtrain.shape
    
  
  #if isinstance(Xtest,pd.DataFrame): print Xtest.describe()
  
  if isinstance(Xtrain,pd.DataFrame):
    df_info(Xtrain)
    df_info(ytrain)
    print Xtrain.describe()
    print Xtrain.columns
  
  print "\n#Xtest:",Xtest.shape
  
  return (Xtrain,ytrain,Xtest,labels)


def make_polynomials_old(Xall,polynomialFeatures):
  print "Polynomial Features..."

  if isinstance(polynomialFeatures,str) and 'load' in polynomialFeatures:
    print "load..."
    Xall = pd.read_csv('tmp.csv',index_col=0,dtype=np.int32)
    #Xall.to_csv('tmp.csv')
    #Xall=sparse.csc_matrix(Xall.values)
    print type(Xall)
  else:#mini_batch features
    poly = PolynomialFeatures(polynomialFeatures,interaction_only=True)
    batch_size=5000
    n,m = Xall.shape
    n_features = 1 + m + m*(m-1)/2
    print "n_features",n_features
    X_new = np.zeros((Xall.shape[0],n_features))
    Xall = Xall.values
    for i in xrange(0,n,batch_size):
	il = min(i + batch_size,n)
	print "at idx:",il
	#poly.fit(Xall[i:il,:])
	#X_new[i:il,:] = poly.transform(Xall[i:il,:])
	X_new[i:il,:] = poly.fit_transform(Xall[i:il,:])
    
    X_new = X_new.astype(np.int32)
    Xall = pd.DataFrame(X_new)
    Xall.to_csv('tmp.csv')
  
  print df_info(Xall)
  return Xall

def group_data(data, degree=2):
  """
  Using groupby pandas
  """
  new_data = []
  m,n = data.shape
  for indices in itertools.combinations(range(n), degree):
    tmp=list(data.columns[list(indices)])
    #print tmp
    tmp2 = data.groupby(tmp)
    #print tmp2.describe()
    group_ids = tmp2.grouper.group_info[0]
    #print group_ids
    #raw_input()
    new_data.append(group_ids)
  return np.array(new_data).T


def dict_encode(encoding, value):
    if not value in encoding:
        encoding[value] = {'code': len(encoding)+1, 'count': 0}
    enc = encoding[value]
    enc['count'] += 1
    encoding[value] = enc


def dict_decode(encoding, value, min_occurs):
    enc = encoding[value]
    if enc['count'] < min_occurs:
        return -1
    else:
        return enc['code']


def group_data2(data, degree=2, min_occurs=2):
    """ 
    Group data using min_occurs
    
    Groups all columns of data into all combinations of degree
    """
    m, n = data.shape
    encoding = dict()
    for indexes in itertools.combinations(range(n), degree):
        for v in data[:, indexes]:
            dict_encode(encoding, tuple(v))
    new_data = []
    for indexes in itertools.combinations(range(n), degree):
        new_data.append([dict_decode(encoding, tuple(v), min_occurs) for v in data[:, indexes]])
    return np.array(new_data).T


def group_data3(data, degree=3, cutoff = 1, hash=hash):
    """ 
    Luca Massaron
    numpy.array -> numpy.array
    
    Groups all columns of data into all combinations of triples
    """
    
    new_data = []
    m,n = data.shape
    for indexes in combinations(range(n), degree):
        new_data.append([hash(tuple(v)) for v in data[:,indexes]])
    for z in range(len(new_data)):
        counts = dict()
        useful = dict()
        for item in new_data[z]:
            if item in counts:
                counts[item] += 1
                if counts[item] > cutoff:
                    useful[item] = 1
            else:
                counts[item] = 1
        for j in range(len(new_data[z])):
            if not new_data[z][j] in useful:
                new_data[z][j] = 0
    return np.array(new_data).T


def makePredictions(model=None,Xtest=None,filename='submission.csv'):
    sample = pd.read_csv('/home/loschen/Desktop/datamining-kaggle/otto/sampleSubmission.csv')
    if model is not None: 
	preds = model.predict_proba(Xtest)
    else:
	preds = Xtest
    # create submission file
    preds = pd.DataFrame(preds, index=sample.id.values, columns=sample.columns[1:])
    preds.to_csv(filename, index_label='id')
    

def plotNN(net1):
    train_loss = np.array([i["train_loss"] for i in net1.train_history_])
    valid_loss = np.array([i["valid_loss"] for i in net1.train_history_])
    plt.plot(train_loss, linewidth=3, label="train")
    plt.plot(valid_loss, linewidth=3, label="valid")
    plt.grid()
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.ylim(1e-3, 1e-2)
    plt.yscale("log")
    plt.show()
 
class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        #print 'NEW VALUE:',new_value
        getattr(nn, self.name).set_value(new_value)

def float32(k):
    return np.cast['float32'](k)


class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = [w.get_value() for w in nn.get_all_params()]
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_weights_from(self.best_weights)
            raise StopIteration()
#def buildModelMLL(clf,lX,ly,class_names,n_folds=8,trainFull=False):
  #print "Training the model..."
  #print clf
  #print class_names
  #if isinstance(lX,pd.DataFrame): lX  = lX.values

  ##cv = StratifiedShuffleSplit(ly, n_iter=n_folds, test_size=0.125)
  #cv =StratifiedKFold(ly,n_folds)
   
  #ypred = np.zeros((len(ly),))
  #yproba = np.zeros((len(ly),len(set(ly))))
  #mll = np.zeros((len(cv),1))
  #for i,(train, test) in enumerate(cv):    
      #ytrain, ytest = ly[train], ly[test]
      #clf.fit(lX[train,:], ytrain)
      #ypred[test] = clf.predict(lX[test,:])
      #yproba[test] = clf.predict_proba(lX[test,:])
      #mll[i] = multiclass_log_loss(ly[test], yproba[test])
      #acc = accuracy_score(ly[test], ypred[test])
      #print "train set: %2d samples: %5d/%5d mll: %4.3f accuracy: %4.3f"%(i,lX[train,:].shape[0],lX[test,:].shape[0],mll[i],acc)
      
  #print classification_report(ly, ypred, target_names=class_names)
  #mll_oob = multiclass_log_loss(ly, yproba)
  
  #print "oob multiclass logloss: %6.3f" %(mll_oob)
  #print "avg multiclass logloss: %6.3f +/- %6.3f" %(mll.mean(),mll.std())
  ##training on all data
  #if trainFull:
    #clf.fit(lX, ly)
  #return(clf)


#TODO check sklearn 1.6 with class_weight crossvalidation
#https://github.com/cudamat/cudamat
#TODO optimize logloss directly http://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf in xgboost_sklearn
#sparse data with LinearSVC
#https://www.kaggle.com/c/tradeshift-text-classification/forums/t/10537/beat-the-benchmark-with-less-than-400mb-of-memory
#http://blogs.technet.com/b/machinelearning/archive/2014/09/24/online-learning-and-sub-linear-debugging.aspx
#new columns: http://stackoverflow.com/questions/16139147/conditionally-combine-columns-in-pandas-data-frame
#Look at problem with SVD - need all features
#group data versus interaction!
# use MKL with numpy https://gehrcke.de/2014/02/building-numpy-and-scipy-with-intel-compilers-and-intel-mkl-on-a-64-bit-machine/
#use greedy algorithm with RF to select new features!!!
#sklearn.multiclass.OneVsOneClassifier(estimator, n_jobs=1 with logistc reg
#http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html#scipy.optimize.basinhopping
#http://pyevolve.sourceforge.net/0_6rc1/getstarted.html

if __name__=="__main__":
    """   
    MAIN PART
    """ 
    # Set a seed for consistant results
    t0 = time()
    np.random.seed(42)
    #pd.set_option('display.height', 1000)
    #pd.set_option('display.max_rows', 500)
    #pd.set_option('display.max_columns', 500)
    #pd.set_option('display.width', 1000)
    print "numpy:",np.__version__
    print "pandas:",pd.__version__
    print "scipy:",sp.__version__
    
    nsamples=-1
    standardize=True
    polynomialFeatures=2#'load'
    featureHashing=False
    OneHotEncoding=False
    analyzeIt=False
    call_group_data=False
    log_transform=False
    addNoiseColumns=None
    addFeatures=False
    doSVD=None
    
    #after linear feature selection, features are orderd according to their effect
    #all_ordered=['feat_11', 'feat_60', 'feat_34', 'feat_14', 'feat_90', 'feat_15', 'feat_62', 'feat_42', 'feat_39', 'feat_36', 'feat_75', 'feat_68', 'feat_9', 'feat_43', 'feat_40', 'feat_76', 'feat_86', 'feat_26', 'feat_35', 'feat_59', 'feat_47', 'feat_17', 'feat_48', 'feat_69', 'feat_50', 'feat_91', 'feat_92', 'feat_56', 'feat_53', 'feat_25', 'feat_84', 'feat_57', 'feat_78', 'feat_58', 'feat_41', 'feat_32', 'feat_67', 'feat_72', 'feat_77', 'feat_64', 'feat_20', 'feat_71', 'feat_83', 'feat_19', 'feat_23', 'feat_88', 'feat_33', 'feat_73', 'feat_93', 'feat_3', 'feat_81', 'feat_13', 'feat_6', 'feat_31', 'feat_52', 'feat_4', 'feat_82', 'feat_51', 'feat_28', 'feat_2', 'feat_12', 'feat_21', 'feat_80', 'feat_49', 'feat_54', 'feat_65', 'feat_5', 'feat_63', 'feat_46', 'feat_27', 'feat_44', 'feat_55', 'feat_7', 'feat_61', 'feat_70', 'feat_10', 'feat_18', 'feat_22', 'feat_38', 'feat_8', 'feat_89', 'feat_16', 'feat_66', 'feat_45', 'feat_30', 'feat_79', 'feat_1', 'feat_24', 'feat_74', 'feat_87', 'feat_37', 'feat_29']
    start_set = ['feat_11', 'feat_60', 'feat_34', 'feat_14', 'feat_90', 'feat_15', 'feat_62', 'feat_42', 'feat_39', 'feat_36', 'feat_75', 'feat_68', 'feat_9', 'feat_43', 'feat_40', 'feat_76', 'feat_86', 'feat_26', 'feat_35', 'feat_59', 'feat_47', 'feat_17', 'feat_48', 'feat_69', 'feat_50', 'feat_91', 'feat_92', 'feat_56', 'feat_53', 'feat_25', 'feat_84', 'feat_57', 'feat_78', 'feat_58', 'feat_41', 'feat_32', 'feat_67', 'feat_72', 'feat_77', 'feat_64', 'feat_20', 'feat_71', 'feat_83', 'feat_19', 'feat_23', 'feat_88', 'feat_33', 'feat_73', 'feat_93', 'feat_3', 'feat_81', 'feat_13']
    featureFilter=start_set
    
    Xtrain, ytrain, Xtest, labels  = prepareDataset(nsamples=nsamples,standardize=standardize,featureHashing=featureHashing,OneHotEncoding=OneHotEncoding,polynomialFeatures=polynomialFeatures,featureFilter=featureFilter,call_group_data=call_group_data,log_transform=log_transform,addNoiseColumns=addNoiseColumns,addFeatures=addFeatures,doSVD=doSVD,analyzeIt=analyzeIt)
    print type(Xtrain)
    
    model = LogisticRegression(C=.1,class_weight=None,penalty='L1')#0.671
    #model = Pipeline([('filter', GenericUnivariateSelect(f_classif, param=90,mode='percentile')), ('model', LogisticRegression(C=1.0))])
    #model = Pipeline([('filter', GenericUnivariateSelect(chi2, param=90,mode='percentile')), ('model', LogisticRegression(C=1.0))])
    #model = Pipeline([('pca', PCA(n_components=20)),('model', LogisticRegression(C=1.0))])
    #model = Pipeline([('pca', TruncatedSVD(n_components=25, algorithm='randomized')),('model', RandomForestClassifier())])
    #model = LDA()
    #model = QDA()
    #model = SGDClassifier(alpha=1E-6,n_iter=800,shuffle=True,loss='log',penalty='l2',n_jobs=1,learning_rate='optimal',verbose=False,class_weight='auto')#mll=0.68
    #model = SGDClassifier(alpha=1E-6,n_iter=800,shuffle=True,loss='modified_huber',penalty='l2',n_jobs=8,learning_rate='optimal',verbose=False)#mll=0.68
    #model =  RandomForestClassifier(n_estimators=500,max_depth=None,min_samples_leaf=1,n_jobs=1,criterion='gini', max_features=20,oob_score=False)
    #model = GradientBoostingClassifier(loss='deviance',n_estimators=120, learning_rate=0.1, max_depth=6,subsample=.5,verbose=False)
    #model = XgboostClassifier(booster='gblinear',n_estimators=50,alpha_L1=0.1,lambda_L2=0.1,n_jobs=2,objective='multi:softprob',eval_metric='mlogloss',silent=1)#0.63
    #basemodel = XgboostClassifier(n_estimators=400,learning_rate=0.05,max_depth=10,subsample=.5,n_jobs=4,objective='multi:softprob',eval_metric='mlogloss',booster='gbtree',silent=1)#0.45
    #model = XgboostClassifier(n_estimators=120,learning_rate=0.1,max_depth=6,subsample=.5,n_jobs=1,objective='multi:softprob',eval_metric='mlogloss',booster='gbtree',silent=1)
    #model = basemodel
    #model = BaggingClassifier(base_estimator=basemodel,n_estimators=10,n_jobs=1,verbose=1)#for some reason parallelization does not work...?estimated 12h runs 10 bagging iterations with 400 trees in 8fold crossvalidation
    #model = SVC(C=10, kernel='linear', shrinking=True, probability=True, tol=0.001, cache_size=200)
    
    #########NN- STANDARDIZE############!!!
    """
    model = NeuralNet(
    layers=[  # three layers: one hidden layer
        ('input', layers.InputLayer),      
        ('hidden1', layers.DenseLayer),
        ('dropout1', layers.DropoutLayer),
        ('hidden2', layers.DenseLayer),
        ('dropout2', layers.DropoutLayer),
        ('hidden3', layers.DenseLayer),
        ('dropout3', layers.DropoutLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape=(None,Xtrain.shape[1]),  # 96x96 input pixels per batch
    hidden1_num_units=300,  # number of units in hidden layer
    hidden1_nonlinearity=nonlinearities.rectify,
    #hidden1_nonlinearity=nonlinearities.tanh,
    #hidden1_nonlinearity=nonlinearities.leaky_rectify,
    #hidden1_nonlinearity=nonlinearities.sigmoid,
    
    #hidden1_nonlinearity=nonlinearities.linear
    dropout1_p=0.25,
    
    hidden2_num_units=300,
    hidden2_nonlinearity=nonlinearities.rectify,
    dropout2_p=0.1,
    
    hidden3_num_units=300,
    hidden3_nonlinearity=nonlinearities.rectify,
    dropout3_p=0.5,
    
    
    #hidden2_num_units=200,
    output_nonlinearity=nonlinearities.softmax,  # output layer uses identity function
    output_num_units=9,  # 30 target values

    eval_size=0.1,

    batch_iterator_train=BatchIterator(batch_size=1024),
    batch_iterator_test=BatchIterator(batch_size=1024),
    
    # optimization method:
    #update=nesterov_momentum,
    update_learning_rate=theano.shared(float32(0.005)),
    #update_learning_rate=0.002,
    update_momentum=theano.shared(float32(0.9)),
    #update_momentum=0.9,

    regression=False,  # flag to indicate we're dealing with regression problem
    max_epochs=500,  # we want to train this many epochs
    verbose=1,
    
    
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.002, stop=0.00001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        EarlyStopping(patience=200),
        ],
    
    )
    """
    
    print model
    print dir(model)
       
    scoring_func = make_scorer(multiclass_log_loss, greater_is_better=False, needs_proba=True)
    #greedyFeatureSelection(model,Xtrain,ytrain,itermax=40,itermin=30,targets=None ,start_features=start_set,verbose=True, cv=StratifiedKFold(ytrain,8), n_jobs=4,scoring_func=scoring_func)
    #analyzeLearningCurve(model,Xtrain,ytrain,cv=StratifiedShuffleSplit(ytrain,24,test_size=0.125),score_func=scoring_func)
    #iterativeFeatureSelection(model,Xtrain,Xtest,ytrain,iterations=1,nrfeats=1,scoring=scoring_func,cv=StratifiedKFold(ytrain,5),n_jobs=1)

    #model = buildClassificationModel(model,Xtrain,ytrain,list(set(labels)).sort(),trainFull=False,cv=StratifiedKFold(ytrain,8,shuffle=True))
    #model = buildModel(model,Xtrain,ytrain,cv=StratifiedKFold(ytrain,4,shuffle=True),scoring=scoring_func,n_jobs=4,trainFull=False)
    #model = buildModel(model,Xtrain,ytrain,cv=StratifiedShuffleSplit(ytrain,2,test_size=0.125),scoring=scoring_func,n_jobs=1,trainFull=False)
    #model = buildClassificationModel(model,Xtrain,ytrain,list(set(labels)).sort(),trainFull=False,cv=StratifiedShuffleSplit(ytrain,2,test_size=0.125,shuffle=True))
    #model.fit(Xtrain.values,ytrain)
    #model.fit(Xtrain,ytrain)
    #plotNN(model)
       
    model = makeGridSearch(model,Xtrain,ytrain,n_jobs=8,refit=False,cv=StratifiedKFold(ytrain,8,shuffle=True),scoring=scoring_func)
    #model = makeGridSearch(model,Xtrain,ytrain,n_jobs=1,refit=False,cv=StratifiedShuffleSplit(ytrain,2,test_size=0.125),scoring=scoring_func)
    #makePredictions(model,Xtest,filename='/home/loschen/Desktop/datamining-kaggle/otto/submissions/submission03042015a.csv')
    plt.show()
    print("Model building done in %fs" % (time() - t0))

