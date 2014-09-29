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
#Xtest<-loadData("/home/loschen/Desktop/datamining-kaggle/african_soil/sorted_test.csv",separator=",")
ytrain<-Xtrain[,(length(Xtrain)-4):length(Xtrain)]
print(summary(ytrain))

Xtrain<-loadData("/home/loschen/Desktop/datamining-kaggle/african_soil/training_mod.csv",separator=",")
Xtest<-loadData("/home/loschen/Desktop/datamining-kaggle/african_soil/test_mod.csv",separator=",")

print(summary(Xtrain))
#idx<-sample(nrow(Xtrain), 50)
#Xtrain<-Xtrain[idx,]
#ytrain<-ytrain[idx,]

oinfo(Xtrain)
oinfo(Xtest)

#Xrf<-boruta_select(Xtrain,ytrain$P)