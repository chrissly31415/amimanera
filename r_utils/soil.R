#!/usr/bin/Rscript

###GENERAL SETTINGS###
set.seed(1234)

if (.Platform$OS.type=='unix') {
  setwd(".")
  source("./qspr.R")
} else {
  setwd("D:/COSMOquick/mp_model")
  source("D:/data_mining/qspr.R")
}

###READ & MANIPULATE DATA###

Xtrain<-loadData("/home/loschen/Desktop/datamining-kaggle/african_soil/training.csv",separator=",")
Xtest<-loadData("/home/loschen/Desktop/datamining-kaggle/african_soil/sorted_test.csv",separator=",")

oinfo(Xtrain)
#print(summary(Xtrain))