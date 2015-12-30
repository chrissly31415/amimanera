#!/usr/bin/python
# coding: utf-8

import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

def R_var_importance():
    base = importr('base')
    ###################################################
    # load dataframe
    store = pd.HDFStore('./data/store.h5')
    print store

    #pandas2ri.activate()
    Xtrain = store['Xtrain']
    ytrain = store['ytrain']
    store.close()
    pandas2ri.activate()
    print Xtrain.info()
    Xtrain_R = pandas2ri.py2ri_pandasdataframe(Xtrain)
    ytrain_R = pandas2ri.py2ri_pandasseries(ytrain)
    print Xtrain_R


    ###################################################
    # R-code
    # http://stackoverflow.com/questions/27801409/get-field-values-from-rpy2-random-forest-object
    r = robjects.r
    r['options'](warn=-1)

    r.library('randomForest')
    rf = r.randomForest(Xtrain_R, ytrain_R,ntree=50,importance = True,do_trace=1)
    df_imp_R = rf.rx("importance")
    df_imp_R = base.as_data_frame(df_imp_R)
    df_imp = pandas2ri.ri2py(df_imp_R)
    print df_imp.sort(columns=['importance.IncNodePurity'],ascending=False)

    #print r.dimnames(rf[8])
    r.varImpPlot(rf,sort=True,n_var=30)


if __name__ == "__main__":
    R_var_importance()