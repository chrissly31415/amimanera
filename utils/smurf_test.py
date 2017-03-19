# coding: utf-8
"""
smuRF: simple multithreaded Random Forest

Version: 1.0
Authors: Christoph Loschen
"""
# add path of python module
import sys
import pandas as pd

from qsprLib import *
from keras_tools import *

sys.path.append('/home/loschen/calc/smuRF/python_wrapper')
import smurf as sf



def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 7),scoring='mse'):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,scoring=scoring)
    print train_sizes
    train_scores = -1.* train_scores
    test_scores = -1 * test_scores
    train_scores_mean = np.mean(train_scores, axis=1)
    print train_scores_mean
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    print test_scores_mean
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

if __name__ == "__main__":
    # X = pd.read_csv('../data/katritzky_n_small.csv',sep=',')
    X = pd.read_csv('/home/loschen/calc/smuRF//data/mp_cdk.csv', sep=',')
    X = X._get_numeric_data()
    #X = X.iloc[:5000,:]
    #print X.describe()
    y = X['Ave °C']
    # y = X['n_exp']
    X.drop(['train', 'Ave °C'], axis=1, inplace=True)
    X.info()
    #print X.head(30)
    # X.drop(['train','n_exp'],axis=1,inplace=True)
    #print X.describe()

    scoring_func = make_scorer(root_mean_squared_error, greater_is_better=False)
    #model = sf.RandomForest(n_estimators=200, mtry=30, node_size=5, max_depth=20, n_jobs=4, verbose_level=0)
    showMemUsageGB()
    #model = LinearRegression()
    #model  = KerasNN(dims=X.shape[1],nb_classes=1,nb_epoch=50,learning_rate=0.005,validation_split=0.0,batch_size=256,verbose=1,activation='tanh', layers=[1024,1024], dropout=[0.5,0.0],loss='mse') # best
    #model = Pipeline([('scaler', StandardScaler()), ('nn',model)])
    model = RandomForestRegressor(n_estimators=200,max_depth=None,min_samples_leaf=5,n_jobs=4, max_features=X.shape[1]/3,oob_score=False)
    # rf.setDataFrame(df)
    #model.printInfo()
    cv = KFold(y.shape[0], 5, shuffle=True)
    #res = buildModel(model,X,y,cv=cv, scoring=scoring_func, n_jobs=2,verbose=False)
    #predict is not thread safe?????
    #print "Final score: ",-1*res.mean()
    #parameters = {'n_estimators': [300], 'mtry': [30], 'max_depth': [20], 'node_size': [5]}
    parameters = {'n_estimators': [300], 'max_features': [20,30], 'max_depth': [20,None], 'min_samples_leaf': [5]}
    #parameters = {'nn__nb_epoch':[50,60,70]}
    #parameters = {}
    model = makeGridSearch(model, X, y, n_jobs=2, refit=False, cv=cv, scoring=scoring_func, parameters=parameters, random_iter=-1)
    #cv = ShuffleSplit(X.shape[0], n_iter=5,test_size=0.25, random_state=0)

    #plot_learning_curve(model, "learning curve", X, y, ylim=(70.0, 15.0), cv=cv, n_jobs=1,scoring=scoring_func,train_sizes=np.linspace(.2, 1.0, 5))

    showMemUsageGB()
    #X,X_test,y,y_test = train_test_split(X, y, test_size=0.50, random_state=123)
    #print X_test.shape
    print X.shape

    #buildXvalModel(model, X, y, sample_weight=None, class_names=None, refit=False, cv= cv)
    #model.fit(X,y)
    # model.printInfo()
    #y_pred = model.predict(X_test)
    # print y_pred
    #print "RMSE test:",root_mean_squared_error(y_test,y_pred)
    #plt.scatter(y_test,y_pred)
    plt.show()
