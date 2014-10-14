################################################################################
#    code collection of helper tools for data mining and machine learning      #
#    (c) by chrissly31415  2014                                                #
################################################################################

require(foreach)
require(doSNOW)

if (.Platform$OS.type=='unix') {
    cat("We are on linux.\n")
    source("./boruta_select.R")
    source("./xvalidation.R")
    source("./gaFeatureSelection.R")
} else {
    cat("We are on windows.\n")
    
    #rm(list = ls(all = TRUE))
    #in case R was aborted
    if (sink.number()>1) sink()
    args <- commandArgs(TRUE)
    options(error=recover)
    
    rootdir<-"D:/data_mining/amimanera/r_utils"
    source(paste(rootdir,"/xvalidation.R",sep=""))
    source(paste(rootdir,"/boruta_select.R",sep=""))
    source(paste(rootdir,"/rf_select.R",sep=""))
    source(paste(rootdir,"/greedySelect.R",sep=""))
    source(paste(rootdir,"/gaFeatureSelection.R",sep=""))
    source(paste(rootdir,"/bagged_net.R",sep=""))
}



loadData<-function(filename,separator=";") {
  ldata = read.csv(file=filename,sep=separator)
  #ldata = read.csv(file="acree_standard_small.csv",sep=";")  
  #ldata = read.csv(file="ONSMP13_orig2.csv")
  #ldata = read.csv(file="ons_standard_small.csv",sep=";")
  #ldata = read.csv(file="katritzky_n_small.csv",sep=";")
  #ldata = read.csv(file="ons_final.csv")
  #take only non fragmented
  
}

prepareData_standard<-function(ldata,removeZeroCols=T,useInterval=F,lowT=0,highT=1000,removeCols=TRUE) {
  #REMOVE ROWS
  if (useInterval==T) {
    ldata<-ldata[ldata$mpK<highT,]
    ldata<-ldata[ldata$mpK>lowT,]
  }
  #remove outliers
  #ldata<-ldata[ldata$SMILES!="CCCCC(CCCC)=O",]
  smiles<-ldata[,2:3]
  ldata<-ldata[,6:length(ldata)]
  #REMOVE COLS
  if (removeCols) {
    ldata<-subset(ldata,select=-fragments)
    ldata<-subset(ldata,select=-Macc4)
    ldata<-subset(ldata,select=-M4)
    #ldata<-subset(ldata,select=-M5)
    #ldata<-subset(ldata,select=-M6)
    ldata<-subset(ldata,select=-Macc3)
    ldata<-subset(ldata,select=-M3)
    ldata<-subset(ldata,select=-frag_quality)
  }
  
  #ldata<-subset(ldata,select=-zwitterion_in_water)
  #ldata<-subset(ldata,select=-similarity)
  #ldata<-subset(ldata,select=-res)
  #ldata<-subset(ldata,select=-alkane) 
  #ldata<-subset(ldata,select=-nbr3s3s)
  #ldata<-subset(ldata,select=-nbr3u1)
  #ldata<-subset(ldata,select=-nbr2s1)
  #ldata<-subset(ldata,select=-nbr3s2s)
  #ldata<-subset(ldata,select=-nbr3s1)
  #ldata<-subset(ldata,select=-nbr3u2u)
  #ldata<-subset(ldata,select=-nbr2u1)
  #ldata<-subset(ldata,select=-mpK)
  #ldata<-subset(ldata,select=-Hfus_exp.kcal.mol.)
  #ldata<-subset(ldata,select=-Sfus_exp.cal.molK.)  
  if (removeZeroCols==T) { 
    cs<-colSums(abs(ldata)==0)
    #cat(cs)
    if (0 %in% cs) {
      ldata<-ldata[,which(colSums((ldata))!=0)]
    }    
  } 
  #cat("Data after preparation:\n")
  return(list(ldata,smiles))
}

prepareData_ons<-function(ldata,lowT,highT,testSet) {
  #REMOVE ROWS
  #ldata<-ldata[ldata$Fragments<2,]
  #ldata<-ldata[ldata$Alkane>0,]
  ldata<-ldata[ldata$mpK<highT,]
  ldata<-ldata[ldata$mpK>lowT,]
  #ldata<-ldata[1:100,]
  #ldata<-ldata[ldata$Fragments<2,]
  if (testSet==T) {
    ldata<-ldata[ldata$Alkane>0,]
    #ldata<-ldata[ldata$zwitterion>0,]
    #ldata<-ldata[ldata$Fragments<2,]
  } 
  #remove outliers
  ldata<-ldata[ldata$SMILES!="CCCCC(CCCC)=O",]
  #print(summary(ldata))
  #hist(ldata$mpK,50)
  #ldata<-ldata[ldata$Si.Sn<1,]
  #define matrix
  smiles<-ldata[,2:3]
  ldata<-ldata[,6:length(ldata)]
  
  #ADD COLS
  #entropy100<-ldata$mu_self-ldata$hint_self100
  #ldata<-cbind(entropy100,ldata)
  Txentropy<-(ldata$mu_self-ldata$hint_self)*-1
  ldata<-cbind(Txentropy,ldata)
  
  #hist(ldata$mu_self_noVDW,50)
  #Txentropy_noVDW<-(ldata$mu_self_noVDW-ldata$hint_self_noVDW)*-1
  #ldata<-cbind(Txentropy_noVDW,ldata)
  
  ldata<-subset(ldata,select=-Alkane) 
  #hist(ldata$hb_self100,50) 
  hb_self_lowT<-(pmax(-5,ldata$hb_self100))
  ldata<-cbind(hb_self_lowT,ldata) 
  ldata<-subset(ldata,select=-hb_self)
  h_crs<-(ldata$hint_self-ldata$hint_self100)
  ldata<-cbind(h_crs,ldata) 
  
  #ldata<-subset(ldata,select=-hb_self100)
  #ldata<-subset(ldata,select=-hb_self_noVDW)
  #ldata<-subset(ldata,select=-Rotatable.bonds) 
  #ldata<-subset(ldata,select=-zwitterion) 
  avratio<-ldata$Area/ldata$Volume
  #avratio<-ldata$Area/(ldata$Volume^(2/3))
  ldata<-cbind(avratio,ldata) 
  #alkylatom_groups<-ldata$Alkylatoms/max(1,ldata$Alkylgroups)
  #ldata<-cbind(alkylatom_groups,ldata) 
  #termbonds<-ldata$Rotatable_bond-ldata$X.rotbonds.CDK.
  #ldata<-cbind(termbonds,ldata)
  #expbonds<-(ldata$Rotatable_bond+ldata$X.rotbonds.CDK.)/2.0
  #ldata<-cbind(expbonds,ldata)
  
  #REMOVE COLS
  ldata<-subset(ldata,select=-zwitterion)
  ldata<-subset(ldata,select=-hint_self)
  #ldata<-subset(ldata,select=-Kier2)
  #ldata<-subset(ldata,select=-hint_self_noVDW)
  #ldata<-subset(ldata,select=-mu_self_noVDW)
  #ldata<-subset(ldata,select=-hint_self100)
  #ldata<-subset(ldata,select=-mu_self100) 
  ldata<-subset(ldata,select=-X..moment6)
  ldata<-subset(ldata,select=-X..moment5)
  ldata<-subset(ldata,select=-X..moment4)
  ldata<-subset(ldata,select=-X..moment3)
  ldata<-subset(ldata,select=-X..moment2)
  ldata<-subset(ldata,select=-Fragments)
  #ldata<-subset(ldata,select=-Molweight..g.mol.)
  #ldata<-subset(ldata,select=-MWeight)
  #ldata<-subset(ldata,select=-Alkylatoms)
  #ldata<-subset(ldata,select=-Alkylgroups)
  #ldata<-subset(ldata,select=-Sumq.e.)
  ldata<-subset(ldata,select=-HB.acc_m4)
  ldata<-subset(ldata,select=-HB.acc_m3)
  ldata<-subset(ldata,select=-HB.acc_m1)
  ldata<-subset(ldata,select=-HB.don_m4)
  ldata<-subset(ldata,select=-HB.don_m3)
  ldata<-subset(ldata,select=-HB.don_m1)
  #ldata<-subset(ldata,select=-X.rotbdsmod)
  #ldata<-subset(ldata,select=-Volume)
  #ldata<-subset(ldata,select=-Area)
  if (testSet==F) { 
    cs<-colSums(abs(ldata)==0)
    cat(cs)
    if (0 %in% cs) {
      ldata<-ldata[,which(colSums((ldata))!=0)]
    }
    
  } 
  #ldata<-subset(ldata,select=-Tmult)
  #ldata<-subset(ldata,select=-shape) 
  #normalize
  return(list(ldata,smiles))
  #return(ldata)
}

prepareData_acree<-function(ldata,lowT,highT,testSet) {
  #REMOVE ROWS
  #print(summary(ldata))
  #ldata<-ldata[ldata$Fragments<2,]
  #ldata<-ldata[ldata$Alkane>0,]
  ldata<-ldata[ldata$mpK<highT,]
  ldata<-ldata[ldata$mpK>lowT,]
  #ldata<-ldata[ldata$Hfus_exp.kcal.mol.<10,]
  #ldata<-ldata[ldata$Hfus_exp.kcal.mol.>0,]
  
  
  if (testSet==T) {
    ldata<-ldata[ldata$Alkane>0,]
    #ldata<-ldata[ldata$zwitterion>0,]
    #ldata<-ldata[ldata$Fragments<2,]
  } 
  #remove outliers
  ldata<-ldata[ldata$SMILES!="CCCCC(CCCC)=O",]
  hist(ldata$mpK,50)
  
  #ldata<-ldata[ldata$Si.Sn<1,]
  #define matrix
  smiles<-ldata[,2:3]
  ldata<-ldata[,6:length(ldata)]
  #ADD COLS
  #entropy100<-ldata$mu_self-ldata$hint_self100
  #ldata<-cbind(entropy100,ldata)
  Txentropy<-(ldata$mu_self-ldata$hint_self)*-1
  ldata<-cbind(Txentropy,ldata)
  #hfus_est<-ldata$hint_self-ldata$hint_self100
  #ldata<-cbind(hfus_est,ldata)
  #hist(ldata$mu_self_noVDW,50)
  avratio<-ldata$Area/ldata$Volume
  #avratio<-ldata$Area/(ldata$Volume^(2/3))
  ldata<-cbind(avratio,ldata) 
  #REMOVE COLS
  ldata<-subset(ldata,select=-Fragments)
  #ldata<-subset(ldata,select=-Kier2)
  #ldata<-subset(ldata,select=-Kier1)
  #ldata<-subset(ldata,select=-mu_self100)
  #ldata<-subset(ldata,select=-mu_self)
  #ldata<-subset(ldata,select=-hint_self100)
  ldata<-subset(ldata,select=-hint_self)
  #ldata<-subset(ldata,select=-hb_self100)
  #ldata<-subset(ldata,select=-hb_self)
  #ldata<-subset(ldata,select=-S_crs)
  #ldata<-subset(ldata,select=-H_crs)
  #ldata<-subset(ldata,select=-H_total)
  ldata<-subset(ldata,select=-X..moment5)
  ldata<-subset(ldata,select=-X..moment4)
  #ldata<-subset(ldata,select=-X..moment2)
  ldata<-subset(ldata,select=-X..moment3)
  ldata<-subset(ldata,select=-X..moment6)
  
  ldata<-subset(ldata,select=-HB.don_m3)
  ldata<-subset(ldata,select=-HB.don_m4)
  ldata<-subset(ldata,select=-HB.acc_m3)
  ldata<-subset(ldata,select=-HB.acc_m4)
  #ldata<-subset(ldata,select=-nbr3s3s)
  #ldata<-subset(ldata,select=-nbr3u1)
  #ldata<-subset(ldata,select=-nbr2s1)
  #ldata<-subset(ldata,select=-nbr3s2s)
  #ldata<-subset(ldata,select=-nbr3s1)
  #ldata<-subset(ldata,select=-nbr3u2u)
  #ldata<-subset(ldata,select=-nbr2u1)
  ldata<-subset(ldata,select=-mpK)
  #ldata<-subset(ldata,select=-Hfus_exp.kcal.mol.)
  ldata<-subset(ldata,select=-Sfus_exp.cal.molK.)
  
  if (testSet==F) { 
    cs<-colSums(abs(ldata)==0)
    cat(cs)
    if (0 %in% cs) {
      ldata<-ldata[,which(colSums((ldata))!=0)]
    }
    
  } 
  print(summary(ldata))
  return(list(ldata,smiles))
}

prepareData_cdk<-function(ldata,lowT,highT,testSet) {
  ldata<-ldata[ldata$mpK<highT,]
  ldata<-ldata[ldata$mpK>lowT,]
  cat(nrow(ldata))
  if (testSet==T) {
    ldata<-ldata[ldata$Alkane>0,]
    #ldata<-ldata[ldata$Fragments<2,]
  } 
  smiles<-ldata[,3:4]
  ldata<-ldata[,7:length(ldata)]
  print(summary(ldata))
  if (testSet==F) { 
    cs<-colSums(abs(ldata)==0)
    if (0 %in% cs) {
      ldata<-ldata[,which(colSums((ldata))!=0)]
    }    
  } 
  #ldata<-subset(ldata,select=-Tmult)
  #ldata<-subset(ldata,select=-shape) 
  
  #normalize
  return(list(ldata,smiles))
  
}

oinfo<-function(O) {
  cat("#class:",class(O))
  cat(" #dimension:",dim(O)," ")
  cat(" #size",object.size(O)/1000000," MB\n")
}

vinfo<-function(v) {
  cat("#class:",class(v))
  cat(" #length:",length(v)," ")
  cat(" #size",object.size(v)/1000000," MB\n")
  
}

normalizeData<-function(lX) {
  fun <- function(x){ 
    a <- min(x) 
    b <- max(x) 
    (x - a)/(b - a) 
  } 
  lX<-apply(lX, 2, fun)   
  return(lX)
}

removeColVar<-function(ldata,cvalue) {
  library(caret)
  cormat<-cor(ldata)
  #print(cormat)
  c<-findCorrelation(cormat, cutoff = cvalue, verbose = FALSE)
  removed<-ldata[,c]
  cat("Removed variables according to cutoff: ",cvalue," \n")
  #print(summary(removed))
  ldata<-ldata[,-c]
  return(ldata)
}

linRegPredict<-function(fit,lX_test,exp,lid_test=NULL) {
  print(lid_test$SMILES)
  pred<-predict(fit,lX_test) 
  plot(pred,exp,col = "blue")
  abline(0,1, col = "black")
  se<-(pred-exp)^2
  cat("LINEAR MODEL TEST RMSE:",compRMSE(pred,exp),"\n")
  if (is.null(lid_test)) {
    predout<-data.frame(predicted=pred,exp=exp,se)
  } else {
    predout<-data.frame(id=lid_test$SMILES,predicted=pred,exp=exp,se=se)
  }  
  #print(predout)
  predout<-predout[with(predout, order(-se)), ]
  #print(predout)
  write.table(predout,file="pred_test.csv",sep=";",row.names=FALSE)
  
  return(pred)
}


linRegTrain<-function(lX,ly,lid=NULL,plot=T) {
  ldata=data.frame(lX,target=ly) 
  fit <- lm(target ~ ., data=ldata)
  #fit<-lm(mpK ~ Ringbonds + Alkylatoms + Conjugated.bonds + X.rotbdsmod + hb_self + Area + E_dielec +nbr11, data=ldata)
  if (plot==T) {
    print(summary(fit))
    plot(fit$fitted.values,ly,col="blue",pch=1, xlab = "predicted", ylab = "exp")
    abline(0,1, col = "black")
    #points(pred,ly_test,col="red",pch=2)
    t<-paste("QSPR wtih ",nrow(lX)," data points and ",length(fit$coefficients)-1, " variables.")
    title(main = t)
    # diagnostic plots 
    #layout(matrix(c(1,2,3,4),2,2)) # optional 4 graphs/page 
    #plot(fit)
    rmse<-compRMSE(fit$fitted.values,ly)
    cat("MLR TRAINING RMSE:",rmse,"\n")
    er<-(fit$fitted.values-ly)
    se<-(er)^2
    plot(ly,er,col="blue",pch=1, xlab = "target", ylab = "residual")
    #points(se,col="red",pch=1)  
    if (!is.null(lid)) {
	pred<-data.frame(lid,predicted=fit$fitted.values,exp=ly,se,er)
    } else {
	pred<-data.frame(predicted=fit$fitted.values,exp=ly,se,er)
    }
    pred<-pred[with(pred, order(-se)), ]
    write.table(pred,file="prediction.csv",sep=",",row.names=FALSE)
    print(colnames(lX))
  }  
  return(fit)
}

varSelectGA<-function(lX,ly) {
  require(subselect)
  ldata<-data.frame(lX,mpK=ly)
  #gamodel<-genetic(cor(ldata),kmin=20,kmax=30,popsize=50,nger=1000)
  gamodel<-anneal(mat = cor(ldata), kmin = 20, kmax = 30, nsol = 50, niter = 1000,
                  criterion = "rv")
  #gamodel<-eleaps(mat = cor(ldata), kmin = 20, kmax = 30, nsol = 4, 
  #                criterion = "rv")
  #print(summary(gamodel))
  print(gamodel)
  str(gamodel)
  vars<-gamodel$bestsets[1,]
  bestdata<-data.frame(ldata[,vars])
  print(colnames(bestdata))
  #ldata<-data.frame(bestdata,mpK=ly)
  return(bestdata)
  #str(gamodel$subsets)
}


variableSelection<-function(lX,ly,mode,nvariables,plotting=F) {
  ldata<-data.frame(lX,target=ly)
  # All Subsets Regression
  library(leaps)
  #subsets<-regsubsets(target~.,data=ldata,nbest=1,nvmax=12,method="exhaustive",force.in=match("Tmult",names(ldata))) 
  nvariables<-nvariables
  subsets<-regsubsets(target~.,data=ldata,nbest=1,nvmax=nvariables,method=mode,force.in=NULL)
  
  
  # view results 
  final<-summary(subsets)
  #str(final)
  i <-which(final$rss==min(final$rss))
  vars <- which(final$which[i,])  # id variables of best model
  #remove intercept
  vars<-vars[2:length(vars)]-1
  bestdata<-data.frame(ldata[,vars])
  #print in reusable format e.g. python
  if (plotting==T) {
      # plot a table of models showing variables in each model. models are ordered by the selection statistic.
      plot(subsets,scale="r2")
      cat("\n[")
      for (i in 1:ncol(bestdata)) {
	cat("\"",names(bestdata)[i],"\"",sep="")
	if (i!=ncol(bestdata)) {
	    cat(",",sep="")
	  } else {
	    cat("]\n\n")
	  }
      }
  }
  
  
  # plot statistic by subset size 
  #library(car)
  #subsets(leaps, statistic="rs2")
  return(bestdata)
}

genLinMod<-function(lX,ly) {
  require(lars)
  larsmodel<-cv.lars(data.matrix(lX),data.matrix(y),plot.it=T,se=T,trace=TRUE,max.steps=80)
  str(larsmodel)
  cat("Least Angle regression RMSE:",sqrt(larsmodel$cv[100]),"\n")
  #print(summary(larsmodel))
}


pc_analysis<-function(lX,ly,lX_test=NULL) {
  #mydata<-data.frame(lX,target=ly)
  mydata<-data.frame(lX)
  if (!is.null(lX_test)) {
    mydata<-data.frame(rbind(lX,lX_test))
  }
  #remove zero columns
  cs<-colSums(abs(mydata)==0)
  #cat(cs)
  if (0 %in% cs) {
    mydata<-mydata[,which(colSums((mydata))!=0)]
  } 
  fit <- princomp(mydata, cor=TRUE)
  print(summary(fit)) # print variance accounted for 
  #print(loadings(fit)) # pc loadings 
  plot(fit,type="lines") # scree plot 
  #print(fit$scores) # the principal components
  #labels <- 1:nrow(lX)
  labels <- rep("-",nrow(lX))
  biplot(fit,xlabs=labels)
  
  
  
}

matchByColumns<-function(df_orig,df_test) {
  validCols<-colnames(df_orig)
  cat("Keeping variables:",validCols,"\n")
  df_test<-subset(df_test,select=validCols)
  return(df_test)
}

compRMSE<-function(a,b) {
  mse<-sum((a-b)^2,na.rm=TRUE)
  rmse<-sqrt(mse/sum(!is.na(a)))
  #cat("RMSE:",rmse)
  return(rmse)
}

compLOGLOSS<-function(predicted,actual) {
  eps<-1e-15
  predicted <- pmin(pmax(predicted, eps), 1-eps)
  result<- -1/length(actual)*(sum((actual*log(predicted)+(1-actual)*log(1-predicted))))
  return(result)
}




gam_model<-function(lX,ly,verbose) {
  library(mgcv)
  mydata<-data.frame(lX,mpK=ly)
  #gam1<-gam(mpK ~ avratio + Rotatable.bonds+ Internal.H.bonds +Conjugated.bonds +hb_self+ X..moment4 +mu_self100 +HB.acc_m3+HB.don_m4+n_arom,family=gaussian,data=mydata)
  #gam1<-gam(mpK ~ s(avratio) + s(Rotatable.bonds)+s(Ringbonds)+s(hb_self)+s(E_dielec)+ s(Txentropy)+s(X.rotbdsmod_new),data=mydata)
  #gam1<-gam(mpK ~ s(avratio) + s(Txentropy)+s(Ringbonds)+s(Conjugated.bonds)+s(hb_self_mod)+s(E_dielec)+ s(HB.don_m2)+s(other_rotbonds)+s(n_total),data=mydata)
  #best lin model
  gam1<-gam(mpK ~ s(avratio)+s(hb_self_lowT) + s(Molweight..g.mol.) +s(Conjugated.bonds) + s(HB.don_m1) +s(E_dielec)+s(Rotatable.bonds)+s(nbr11)+s(Kier2),data=mydata)
  #gam1<-gam(mpK ~ s(avratio,k=4) + s(Txentropy)+s(Conjugated.bonds)+s(hb_self_mod)+s(E_dielec),data=mydata)
  #gam1<-gam(mpK ~ s(.),data=mydata)
  #print(summary(gam1))
  #str(gam1)  
  cat("GAM RMSE:",compRMSE(gam1$fitted.values,ly),"\n")
  if (verbose==T) {
    plot(gam1,residuals=FALSE,pch=12,shade=T, scale=0)
    gam.check(gam1,pch=19,cex=.3)
  }
  return(gam1)
}

trainResidues<-function(lX,ly,linvalues,iter,useRF) {
  cat("Train residue model:\n")
  residues<-ly-linvalues
  hist(residues,50)
  plot(residues,ly)
  if (useRF==T) {
    model<-trainRF(lX,residues,iter)    
    residue_pred<-model$predicted
  } else {
    model<-linRegTrain(lX,residues,null,F)
    residue_pred<-model$fitted.values
  }
  finalpred<-linvalues+residue_pred
  plot(finalpred,ly,col="blue",xlab = "predicted", ylab = "exp")
  abline(0,1, col = "black")
  rmse<-compRMSE(finalpred,ly)
  cat("RMSE (lin+RF):",rmse,"\n")
  results<-data.frame(exp=ly,lin=linvalues,residues=residues,rfcorr=residue_pred,finalpred=finalpred)
  write.table(results,file="residue_train.csv",sep=",",row.names=FALSE)
  return(model)
}

predictResidues<-function(lX,ly,model,linvalues) {
  cat("Predict residue model:\n")
  residues<-predict(model,lX)
  finalpred<-linvalues+residues
  points(finalpred,ly,col="red",xlab = "predicted", ylab = "exp")
  abline(0,1, col = "black")
  cat("RMSE,test (lin+RF):",compRMSE(finalpred,ly),"\n")
  cat("R^2,test (lin+RF):",(cor(finalpred,ly))^2,"\n")
  cat("slope,test (lin+RF):",getSlope(finalpred,ly),"\n")
  results<-data.frame(exp=ly,lin=linvalues,residues_true=ly-linvalues,rfcorr=residues,finalpred=finalpred)
  write.table(results,file="residue_test.csv",sep=",",row.names=FALSE)
  #cat(residues)  
}

pc_correct<-function(actual,predicted) {
  sum<-0.0
  for(i in 1:length(actual)) {
    if (actual[i]==predicted[i]) {
      sum<-sum+1
    }
  }
  cat("Sum correct predictions:",sum,"\n")
  pc<-sum/length(actual)
  cat("% correct predictions:",pc,"\n")
  cat("% wrong predictions:",(1-pc),"\n")
  return(pc)
}

computeAUC<-function(predicted,truth,verbose=F) {
  require("ROCR")
  pred<-prediction(predicted, truth)
  perf <- performance(pred,"tpr","fpr")
  if (verbose) plot(perf,col="black",lty=3, lwd=3)
  auc<-performance(pred,"auc")
  auc<-unlist(slot(auc, "y.values"))
  #str(auc)
  if (verbose) cat("AUC:",auc,"\n")
  return(auc)
}

getSlope<-function(lx,ly) {
  slopeinfo<-lm(ly~lx)
  slope<-slopeinfo$coefficients[2]
  #cat("Slope:",slope,"\n")
  return(slope)
}

#optimizing parameters for gbm
caret_train<-function(Xl,yl,classification=F,lmethod="gbm") {
  oinfo(Xl)
  #require(gbm)
  require(caret)
  #options(warn=-1)
  
  ####windows#########
  require(doSNOW)
  #cat("procs:",getDoParWorkers()," name:",getDoParName()," version:",getDoParVersion(),"\n")
  #cl<-makeCluster(nrfolds, type = "SOCK",outfile="")
  #registerDoSNOW(cl)
  
  ####Linux############
  #library(doMC)
  #registerDoMC(4)
  
  if (classification) {
    cat("classificiation\n")
    yl<-factor(yl)    
  }
  #at least 3 iter values..?
  grid<-expand.grid(.interaction.depth = c(4,5,6),.n.trees = c(5000,10000,15000),.shrinkage = c(0.01,0.05,0.001))
  #grid<-createGrid("gbm", data = tmp, len = 3)
  #print(summary(grid))
  #grid<-expand.grid(.K.prov=c(4,5))
  #grid<-expand.grid(.k=c(10,20,50))
  #grid<-expand.grid(.treesize=c(2,4))
  #fitControl <- trainControl(method = "cv",number = 5,summaryFunction = twoClassSummary,verboseIter=TRUE,classProbs=TRUE)
  #fitControl <- trainControl(method = "repeatedcv",repeats = 5,classProbs = classification,verboseIter=TRUE)
  fitControl <- trainControl(method = "repeatedcv",number = 5,repeats = 2,verboseIter=F)
  if (classification) {
    model <- train(Xl, yl,method = lmethod, trControl = fitControl,metric="ROC", tuneGrid = grid,maximize=TRUE)
    plot(model,metric="ROC",plotType="level") 
  } else {
    model <- train(Xl, yl,method = lmethod, trControl = fitControl,metric="RMSE", tuneGrid = grid,maximize=FALSE,verbose=FALSE)
    plot(model,metric="RMSE",plotType="level") 
  }
  print(model)
  print(model$results)
  #str(gbmFit1$results)
  #expred <- extractPrediction(list(gbmFit1))
  #plotObsVsPred(expred)
  resampleHist(model)
  summary(model)
  #
}

gbm_grid<-function(lX,ly,lossfn="auc",treemethod="gbm") {
  df<-data.frame(lX,ly)
  intseq<-c(3,4,5,6)#can also be mtry
  shseq<-c(0.02)
  iterseq<-c(250,300,400)
  for(i in intseq) {
    for(j in shseq) {
      for(k in iterseq) {
        if (treemethod=="gbm") {
          cat("Iterations:",k," int.depth:",i, " shrinkage:",j," ")
          xval_oob(lX,ly,numberTrees=k,nrfolds=5,intdepth=i,sh=j,minsnode=5,repeatcv=2,lossfn=lossfn,method=treemethod)
        } else {
          cat("Iterations:",k," mtry:",i," ")
          xval_oob(lX,ly,numberTrees=k,nrfolds=5,mtry=i,repeatcv=2,lossfn=lossfn,method=treemethod)
        }
        cat("\n")
      }
    }
  }
  
}

xval_oob<-function(Xl,yl,iterations=500,nrfolds=5,intdepth=2,sh=0.01,minsnode=5,repeatcv=2,lossfn="rmse",method="gbm",mtry=5,oobfile='oob_res.csv') {
  #parallel cv loop
  #outer loop
  lossdat<-foreach(j = 1:repeatcv,.combine="cbind") %do% {
      train<-createFoldIndices(Xl,k=nrfolds)
      cl<-makeCluster(nrfolds, type = "SOCK")
      registerDoSNOW(cl) 
      #inner parallel loop
      res<-foreach(i = 1:nrfolds,.packages=c('gbm','randomForest'),.combine="rbind",.export=c("compRMSE","computeAUC","oinfo","linRegTrain","variableSelection")) %dopar% {
        cat("###FOLD: ",i," Iterations: ",iterations," ") 
        idx <- which(train == i)
        Xtrain<-Xl[-idx,]
        ytrain<-yl[-idx]
        Xtest<-Xl[idx,]
        ytest<-yl[idx]
        if (lossfn=="auc") {
          if (method=="gbm") {
            gbm1<-gbm.fit(Xtrain ,ytrain,distribution="bernoulli",n.trees=iterations,interaction.depth=intdepth,shrinkage=sh,n.minobsinnode = minsnode,verbose=F)
            results<-predict.gbm(gbm1,Xtest,n.trees=iterations,type="response")
          } else if (method=='randomForest') {
            #RF
            rf1 <- randomForest(Xtrain,ytrain,ntree=iterations,mtry=mtry,importance = F,nodesize =10)
            results<-predict(rf1,Xtest,type="vote")[,2]
          } else if (method=='linear') {
	    #linear regression
	    #Xtrain<-variableSelection(Xtrain,ytrain,"forward",iterations)
	    fit<-linRegTrain(Xtrain,ytrain,NULL,F)
	    results<-predict(fit,Xtest)
          }
          score<-computeAUC(results,ytest,F)
        } else {
          if (method=="gbm") {
            gbm1<-gbm.fit(Xtrain ,ytrain,distribution="gaussian",n.trees=iterations,interaction.depth=intdepth,shrinkage=sh,n.minobsinnode = minsnode,verbose=F)
            results<-predict.gbm(gbm1,Xtest,n.trees=iterations,type="response")
            } else if (method=='randomForest') {
              #RF
              rf1 <- randomForest(Xtrain,ytrain,ntree=iterations,mtry=mtry,importance = F,nodesize =10)
              results<-predict(rf1,Xtest)             
            } else if (method=='linear') {
	      #linear regression
	      Xtrain<-variableSelection(Xtrain,ytrain,"forward",iterations)
	      fit<-linRegTrain(Xtrain,ytrain,NULL,F)
	      results<-predict(fit,Xtest)
            }
          score<-compRMSE(results,ytest)
          }        
        cat(" LOSS:",score,"\n") 
        #returning dataframe with predictions and truth
        final<-data.frame(idx,results,ytest)
        return(final)
      }#end parallel inner loop
      res<-res[order(res$idx),]
      if (lossfn=="auc") {
        auc_cv<-computeAUC(res$results,res$ytest)
        cat("AUC, CV:",auc_cv,"\n")
      } else {
        auc_cv<-compRMSE(res$results,res$ytest)
        cat("RMSE, CV:",auc_cv,"\n")
      }  
      stopCluster(cl)
      return(res$results)
  }#end outer loop
  print(summary(lossdat))
  oob_mean<-apply(lossdat, 1, function(x) mean(x))
  #oob_std<-apply(lossdat, 1, function(x) sd(x))
  #print(summary(oob_std))
  #write.table(data.frame(oob_std),file="gbm_sd.csv",sep=",",row.names=FALSE)
  if (lossfn=="auc") {
    score_iter<-apply(lossdat, 2, function(x) computeAUC(x,yl,F))
    cat("AUC of iterations:",score_iter,"\n")
    cat("<AUC,oob>:",computeAUC(oob_mean,yl,F),"\n") 
  } else {
    score_iter<-apply(lossdat, 2, function(x) compRMSE(x,yl))
    cat("RMSE of iterations:",score_iter,"\n")
    cat("<RMSE,oob>:",compRMSE(oob_mean,yl)," RMSE,mean:",mean(score_iter)," sdev:",sd(score_iter),"\n") 
  }
  res<-data.frame(prediction=oob_mean)
  if (!is.null(oobfile)) {
      write.table(res,file=oobfile,sep=",",row.names=FALSE)
  }
  return(res)
}



trainRF<-function(lX,ly,iter=500,m.try=if (!is.null(ly) && !is.factor(ly)) max(floor(ncol(lX)/3), 1) else floor(sqrt(ncol(lX))),node.size=5, verbose=T,fimportance=F) {
  cat("Training random forest...")
  require(randomForest)
  mydata.rf <- randomForest(lX,ly,ntree=iter,mtry=m.try,importance = fimportance,nodesize =node.size)
  
  if (fimportance) {
    imp<-importance(mydata.rf,type=1)
    varImpPlot(mydata.rf,type=1,main="")
    #write.table(data.frame(imp),file="importance.csv",sep=",")
    #png(file="importance.png",width=1600,height=1200,res=300)
    #par(mar=c(4,1,2,2))
    #dev.off()
  }

  if (mydata.rf$type=="regression") {  
    cat("...regression\n")
    nr.samples<-nrow(lX)
    rmse<-mean(sqrt(mydata.rf$mse)*(nr.samples-1)/nr.samples)
    stdev<-sd(mydata.rf$mse)
    if (verbose==T) {     
      cat("RF RMSE:",rmse,"stddev:",stdev," ")
      cat("R^2:",mean(mydata.rf$rsq),"\n")
      #plot
      results<-data.frame(predicted=mydata.rf$predicted,exp=mydata.rf$y)
      plot(results,col="blue",pch=1, xlab = "predicted", ylab = "exp")
      abline(0,1, col = "black")
      t<-paste("RF wtih ",nrow(lX)," data points and ",ncol(lX), " variables.")
      title(main = t) 
      #residues: pred-y 
      results<-data.frame(exp=mydata.rf$y,residue=mydata.rf$predicted-mydata.rf$y)
      plot(results,col="blue",pch=1, xlab = "exp", ylab = "residues")
      title(main = "RF residuals (y-pred)")
      se<-(mydata.rf$predicted-mydata.rf$y)^2
      ldata<-data.frame(predicted=mydata.rf$predicted,exp=mydata.rf$y,se)
    }      
  } else {
    cat(" classification\n")
    if (verbose==T) {
      print(nlevels(ly))
      cat("Random forest OOB err rate:",mydata.rf$err.rate[iter],"\n")
      #tmp<-mydata.rf$predicted
      tmp<-mydata.rf$votes[,2]
      #hist(as.numeric(mydata.rf$predicted))
      hist(as.numeric(tmp))
      #cat(mydata.rf$votes)
      ldata<-data.frame(predicted=as.numeric(as.character(tmp)),truth=as.numeric(as.character(ly)))
      #plot(ldata,col="blue",pch=1, xlab = "predicted", ylab = "exp")
      #ldata<-data.frame(predicted=tmp,truth=as.numeric(as.character(ly)))
      computeAUC(ldata$predicted,ldata$truth,T)
      #pc_correct(ldata$predicted,ldata$truth)
    }
  }   
  #write.table(ldata,file="prediction_rf.csv",sep=",",row.names=FALSE)
  return(mydata.rf)
}

saveModel<-function(lX,ly,iter) {
  require("pmml")
  require("XML")
  require("randomForest")
  mydata<-data.frame(lX,mpK=ly)
  #load("mydata_rf.RData")
  print(summary(mydata))
  #print(summary(lX))
  #model<-randomForest(Sepal.Length ~ ., data=iris)
  cat("Training Forest\n")
  model<-randomForest(mpK ~ .,data=mydata,ntree=iter,importance = FALSE,nodesize=5)
  
  nr.samples<-nrow(lX)
  rmse<-mean(sqrt(model$mse)*(nr.samples-1)/nr.samples)
  stdev<-sd(model$mse)
  cat("RMSE:",rmse,"stddev:",stdev,"\n")
  cat("R^2:",mean(model$rsq),"\n")
  
  write.table(data.frame(mydata,predicted=model$predicted),file="simple_data.csv",sep=",",row.names=FALSE)
  #mytree<-getTree(model, k=1, labelVar=FALSE)
  #str(mytree)
  print(model)
  #EXPORT ONLY WORKS FOR RF CREATED WITH FORMULA INTERFACE?
  cat("Exporting PMML\n")
  p<-pmml(model)
  saveXML(p,"model.xml")
  return(model)
}


mlr_analysis<-function(fit,lX,lid,lname) {
  n<-which(lid$SMILES==lname)
  cat("index",n," compound: ",toString(lid[n,2:2]),"\n")
  obs<-fit$model[n,]
  sum<-0.0
  cat(sprintf("%-24s%12s%12s%12s%12s\n","Variable","Value","Factor", "Incr.","Total"))
  for(i in 2:length(fit$coefficients)) {
    contrib<-obs[1,i]*fit$coefficients[i]
    sum<-sum+contrib
    if (obs[1,i]<10e-15 && obs[1,i]>-10e-15) {
      next
    }
    cat(sprintf("%-24s",names(fit$coefficients)[i]))    
    cat(sprintf("%12.2f",obs[1,i]))
    cat(sprintf("%12.2f",fit$coefficients[i]))    
    cat(sprintf("%12.2f %12.2f\n",contrib,sum))
  }
  sum<-sum+fit$coefficients[1]
  cat("Intercept:",fit$coefficients[1])
  #str(fit$model)
  cat("\n##Calc. Property:",(sum)," true: ",fit$model[n,"target"], "residual: ",fit$residuals[n] ,"##\n")
}

prepareStandard<-function(filename) {
  ldata = read.csv(filename)
  ldata<-ldata[,3:length(ldata)]  
  cs<-colSums(abs(ldata)==0)
  if (0 %in% cs) {
    ldata<-ldata[,which(colSums((ldata))!=0)]
  }
  #print(summary(ldata))
  return(ldata)
}


compareViaPrinComp<-function(lX1,lX2,ly=NULL,threshold=0.5) {
  if (!is.null(ly)) {
    #remove zero columns
    cs<-colSums(abs(lX1)==0)
    #cat(cs)
    if (0 %in% cs) {
      lX1<-lX1[,which(colSums((lX1))!=0)]
    }   
    print(summary(lX1))
    cat("Using label data...")
    idx <- which(as.numeric(as.character(ly)) < threshold)
    lX1<-lX1[idx,]
    lX2<-lX1[-idx,]
    print(summary(lX1))
    print(summary(lX2))
  } 
  pc1 <- prcomp(lX1, scale. = T)
  x1 <- pc1$x  
  pc2 <- prcomp(lX2, scale. = T)
  x2 <- pc2$x
  
  #print(summary(x))
  #plot(x1[, 1], x1[, 2], pch = 2,col=rgb(1,0,0,1/4),xlim=c(-10,20),ylim=c(-20,30))
  plot(x1[, 1], x1[, 2], pch = 2,col=rgb(1,0,0,1/4))
  points(x2[, 1], x2[, 2], pch = 1,col=rgb(0,0,1,1/4))
  legend(30, 25, c("dataset 1","dataset 2"), cex=1,pch=c(2,1),col=c("red","blue"))
  #biplot
}

#works not properly loss function too low -> use xvalid instead
xvalSVM<-function(lX,ly) {
  require(ipred)
  require(e1071)
  mydata<-data.frame(lX,target=ly)
  error.SVM <- numeric(10)
  for (i in 1:5) error.SVM[i] <-errorest(target~.,data=mydata,model = svm, cost = 10, gamma = 1.5)$error
  print(summary(error.SVM))
}

trainSVM<-function(lX,ly) {
  require(e1071)
  mydata<-data.frame(lX,target=ly)
  #mydata<-data.frame(lX,target=ly)
  m <- svm(target~ .,data=mydata,cost = 10, gamma = 1.5)
  new <- predict(m, lX)
  plot(ly, new)
  rmse<-compRMSE(ly,new)
  cat("RMSE:",rmse,"\n")
  cor.p<-cor(ly,new)
  cat("R^2:",cor.p,"\n")
  #cat("R^2:",mean(mydata.rf$rsq),"\n")  
}

#Local outlier Factor
#http://www.rdatamining.com/examples/outlier-detection
#http://www.dbs.ifi.lmu.de/~zimek/publications/KDD2010/kdd10-outlier-tutorial.pdf
outlierDetection<-function(lX,ly,nrPCA,sortOnResidues=FALSE,plot=TRUE,lfit=NULL,lsmiles=NULL,returnAll=FALSE) {
  require(DMwR)
  nout=20
  nneighbors=5
  cat("Starting outlier detection...")
  outlier.scores <- lofactor(lX, k=nneighbors)
  str(outlier.scores)
  plot(density(outlier.scores))
  
  if (sortOnResidues==TRUE) {
    res<-lfit$residuals
    outliers <- order(res, decreasing=T)[1:nout]
    cat("correlation res-outlier_scores:",cor(res,outlier.scores),"\n")
  } else {
    outliers <- order(outlier.scores, decreasing=T)[1:nout]
  }
  print(summary(outlier.scores))
  n <- nrow(lX)
  labels <- 1:n
  labels[-outliers] <- "."
  biplot(prcomp(lX), cex=.8, xlabs=labels)
  if (!is.null(lsmiles)) {
    predout<-data.frame(lX[outliers,],lsmiles[outliers,],outl_score=outlier.scores[outliers],exp=ly[outliers],predicted=lfit$fitted.values[outliers],res=res[outliers])
    write.table(predout,file="outlier.csv",sep=";",row.names=FALSE)
  }
  #write test set with labels
  
  if (returnAll==T) {
    newdf<-data.frame(lX,outl_score=outlier.scores)
  } else {
    newdf<-data.frame(outl_score=outlier.scores)
  }
  
  return(newdf)
}


#model:rf,iter,gam,svm,boost,gbm
xvalid<-function(lX,ly,nrfolds=5,modname="rf",lossfn="auc",iter=500,mtry=5) {
  require(e1071)
  require(gbm)
  ldata=data.frame(lX,target=ly)
  all_folds<-random_folds(ldata,k=nrfolds)  
  loss<-mat.or.vec(nrfolds,1)
  for(i in 1:nrfolds) {
    train<-return_fold(all_folds,i,test=F)
    Xtrain<-train[,1:ncol(train)-1]
    ytrain<-train[,length(train)]
    #TRAINING
    if (modname=="rf") {
      fit<-trainRF(Xtrain,ytrain,iter,mtry)
    }
    else if (modname=="gam") {
      fit<-gam_model(Xtrain,ytrain,F)
    }
    else if (modname=="svm") {
      cat("Training SVM\n")
      #For some reason we have to use the formula interface
      #traindata=data.frame(Xtrain,target=ytrain)
      #fit<-svm(target ~ .,data=traindata)
      fit<-svm(Xtrain,ytrain, kernel='radial',probability=T)
    } else if (modname=="boost") {
      cat("TRAIN LINMODEL+RF: ")
      fit<-linRegTrain(Xtrain,ytrain,NULL,F)
      residues<-ytrain-fit$fitted.values
      rfmodel<-trainRF(Xtrain,residues,iter)
      print(rfmodel)
      finalpred<-fit$fitted.values+rfmodel$predicted
      if (lossfn=="rmse") {
        loss<-compRMSE(finalpred,ytrain)
        cat("RMSE:",rmse,"\n")
      } else {
        loss<-computeAUC(finalpred,ytrain,F)
        cat("AUC:",loss,"\n")
      }
      
    } else if (modname=="gbm") {
      cat("TRAIN GBM: ")
      if (lossfn=="rmse") {
        fit<-gbm.fit(Xtrain,ytrain,distribution="gaussian",n.trees=iter,interaction.depth=5,shrinkage=0.01,verbose=F)
      } else {
        cat("gbm: 0-1 distribution...")
        fit<-gbm.fit(Xtrain,ytrain,distribution="bernoulli",n.trees=iter,interaction.depth=20,shrinkage=0.001,verbose=F)
      }   
    } else {       
      fit<-linRegTrain(Xtrain,ytrain,NULL,F)
    }
    #PREDICTION
    test<-return_fold(all_folds,i,test=T)
    Xtest<-test[,1:ncol(test)-1]
    ytest<-test[,length(test)]
    if (modname=="rf") {
      if (lossfn=="rmse") {
        pred<-predict(fit,Xtest)
      } else {
        pred<-predict(fit,Xtest,type="vote")[,2]
      }    
    }
    else if (modname=="gam") {
      pred<-predict(fit,Xtest)
    }
    else if (modname=="svm") {
      cat("Predicting via SVM\n")
      pred<-predict(fit,Xtest)
    } else if (modname=="boost") {
      cat("TEST LINMODEL+RF: ")
      linpred<-linRegPredict(fit,Xtest,ytest,NULL)
      residues<-predict(rfmodel,Xtest)       
      pred<-linpred+residues
    } else if (modname=="gbm") {
      cat("TEST GBM: ")
      pred<-predict(fit,Xtest,n.trees=iter,type="response")
      print(summary(data.frame(pred)))
    } else {
      pred<-linRegPredict(fit,Xtest,ytest,NULL)
    }
    cat("n(test set):",length(ytest),"\n")
    if (lossfn=="rmse") {
      lossi<-compRMSE(pred,ytest)
    } else {
      lossi<-computeAUC(pred,ytest,F)
    }
    cat("Fold ",i," - Test Set Loss:",lossi,"\n")
    loss[i]<-lossi
  }
  if (lossfn=="rmse") {
    cat("Final RMSE:",mean(loss)," stdev: ",sd(loss),"\n")
  } else {
    cat("Final AUC:",mean(loss)," stdev: ",sd(loss),"\n")
  }
  return(mean(loss))
}

trainDBN<-function(lX,ly,lXtest=NULL,lytest=NULL){
  require(h2o)
  #setup heo library : http://cran.r-project.org/web/packages/h2o/h2o.pdf
  localH2O <- h2o.init(ip = "localhost", port = 54321, startH2O = TRUE, Xmx = '2g')
  
  data<-cbind(lX,ly) 
  data_h2o <- as.h2o(localH2O, data, key = 'data')
  
  model <- 
    h2o.deeplearning(x = 1:ncol(lX),  # column numbers for predictors
                     y = ncol(data),   # column number for label
                     data = data_h2o, # data in H2O format
                     classification = FALSE,
                     activation = "MaxoutWithDropout", # or 'Tanh' TanhWithDropout
                     #activation = "RectifierWithDropout", # or 'MaxoutWithDropout'
                     input_dropout_ratio = 0.2, # % of inputs dropout
                     hidden_dropout_ratios = c(0.5,0.5), # % for nodes dropout
                     #balance_classes = TRUE, 
                     seed=42,
                     loss='MeanSquare',
                     l2=1e-10,
                     hidden = c(50,50), # three layers of 50 nodes
                     epochs = 2000) # max. no. of epochs
  
  if (is.null(lXtest) && is.null(lytest)) {
    cat("Prediction on training data:\n")
    Xtest_h2o <- as.h2o(localH2O, X, key = 'train')
  } else {   
    cat("Test set provided:\n")
    Xtest<-matchByColumns(X,lXtest)
    Xtest_h2o <- as.h2o(localH2O, Xtest, key = 'test')
    ly<-ly_test
  }
    
  pred_h2o <- h2o.predict(model, Xtest_h2o)
  finalpred <- as.data.frame(pred_h2o)[,1]

  plot(finalpred,ly,col="blue",xlab = "predicted", ylab = "exp")
  abline(0,1, col = "black")
  rmse<-compRMSE(finalpred,y)
  cat("RMSE (DBN):",rmse,"\n")
  return(rmse)
  
}



printErrors<-function(model,lX,ly,id) {
  #correlationEllipses(cor(Xrf))
  se<-(model$predicted-model$y)^2
  er<-(model$predicted-model$y)
  pred<-data.frame(id,predicted=model$predicted,exp=ly,se,er)
  pred<-pred[with(pred, order(-se,decreasing=F)), ]
  write.table(pred,file="prediction_rf.csv",sep=";",row.names=FALSE)  
}

###PLOTTING###
makeBubblePlot<-function(model,lX,ly) {
  library(ggplot2)
  require(XML)
  if (!is.null(model$predicted)) {
    pred<-model$predicted#RF 
  } else if (!is.null(model$cv.fitted)) {
    pred<-model$cv.fitted#GBM CV    
  } else if (!is.null(model$fit)) {
    pred<-model$fit#GBM.FIT
  } else {
    pred<-model$fitted.values#LinReg  
  }
  cat("RMSE:",compRMSE(ly,pred))
  mdf<-data.frame(lX,target=ly,pred=pred)  
  b<-seq(100, 600, by = 100)
  p<-ggplot(data=mdf)+
    #geom_point(aes(x=pred,y=target,colour=M2,size=molweight))+
    #geom_point(aes(x=pred,y=target,colour=M2,size=molweight), alpha = 0.5, position = "jitter")+
    geom_point(aes(x=pred,y=target,colour=M2,size=ringbonds), alpha = 0.5)+
    #geom_point(aes(x=pred,y=target,size=tmult,colour=M2), alpha = 0.5)+
    #scale_colour_gradientn(colours=c("black","white"))+
    scale_colour_gradientn(colours=c("#00A352","red"))+
    scale_x_continuous("Tm, pred. [K]",breaks = b) +
    scale_y_continuous("Tm, exp. [K]",breaks = b)+
    #scale_size(guide="none",range=c(5,20))+
    scale_size(range=c(3,15))+
    #scale_colour_gradientn(colours=rainbow(2))+
    geom_abline(intercept=0, slope=1,size=1)+
    coord_fixed()+
    #xlim(100, 700)+
    #ylim(100, 700)+
    #scale_colour_gradientn(colours=c("black", "white"))+
    #theme(axis.text.x = element_text(colour="black",size=20,angle=90,hjust=.5,vjust=.5,face="plain"),
    #      axis.text.y = element_text(colour="black",size=20,angle=0,hjust=1,vjust=0,face="plain"))+
    theme_classic(base_size=20)
  #print(p)  
  ggsave('mp2.png',  p)
  #ggsave('mp1.png',  p, width = 200, height = 200, units = "mm",dpi = 300, scale = 1)
}

#http://is-r.blogspot.de/2012/11/plotting-correlation-ellipses.html
correlationEllipses<-function(cor){
  require(ellipse)  
  #ord <- order(cor[1, ])
  #cat(ord)
  #xc <- cor[ord, ord]
  xc <- cor
  colors <- c("#A50F15","#DE2D26","#FB6A4A","#FCAE91","#FEE5D9","white",
              "#EFF3FF","#BDD7E7","#6BAED6","#3182BD","#08519C")
  colors <- c("#000000","#636363","#8A8A8A","#B0B0B0","#C9C9C9","white","#C9C9C9","#B0B0B0","#8A8A8A","#636363","000000")
  tmp<-colors[5*xc + 6]
  #   png(
  #     "corr.png",
  #     width     = 3.25,
  #     height    = 3.25,
  #     units     = "in",
  #     res       = 1200,
  #     pointsize = 4
  #   )
  #   par(
  #     mar      = c(5, 5, 2, 2),
  #     xaxs     = "i",
  #     yaxs     = "i",
  #     cex.axis = 2,
  #     cex.lab  = 1
  #   )
  plotcorr(xc,col=tmp)
  #  dev.off()
  #print(xc) 
}

plotPartialdependence<-function(rf,ldata,n=3) {
  #PARTIAL DEPENDENCE
  imp<-importance(rf)
  print(summary(imp))
  impvar<-rownames(imp)[order(imp[, 1], decreasing=TRUE)]
  cat("impvar:",impvar,"\n")
  #op <- par(mfrow=c(2, n/2+1))
  #for (i in seq_along(impvar)) {
  for (i in 1:n) {
    cat("impvar:",impvar,"\n")  
    davar<-impvar[i]
    cat("impvar:",davar,"\n")
    partialPlot(rf,ldata,massprotbond)
#     partialPlot(ozone.rf, airquality, impvar[i], xlab=impvar[i],
#                 main=paste("Partial Dependence on", impvar[i]),
#                 ylim=c(30, 70))
    
  }
  #par(op)
}



#using sparse GF matrix
errorModel<-function() {
  require(glmnet)
  require(Matrix)
  
  edata = read.csv(file="error_fg.csv",sep=";")
  lX<-edata[,1:ncol(edata)-1]
  
  lX<-lX[,-c(1,2,3,4,5)]
  #cat(cs)
  cat("ncol before:",ncol(lX),"\n")
  lX<-lX[,which(colSums(lX)!=0)]
  lX<-removeColVar(lX,0.95)
  cat("ncol after:",ncol(lX),"\n")
  print(summary(lX))
  
  ly<-edata[,ncol(edata)]
  #selcol<-c(phosphonic_acid_derivative,conjugated_double_bond,heterocyclic,carboxylic_anhydride,carboxylic_acid,nitroso,S.Se,ketone,hbond_acceptors_CKD,cyanhydrine,aldehyde,tertiary_arom_amine)
  cvglm=cv.glmnet(as.matrix(lX),ly,family="gaussian",alpha=1.0,standardize = F,type.measure = "mse",nfolds = 10,intercept = T) 
  plot(cvglm)
  cat("RMSE best:",min(sqrt(cvglm$cvm)),"\n")
  coef.fit<-coef(cvglm,s=cvglm$lambda.min)[-1]
  threshhold=7
  index <- which(coef.fit >threshhold | coef.fit< -threshhold)
  good.coef <- coef.fit[index]
  cat(colnames(lX[,index]),"\n")
  #cat("Good ones",good.coef,"\n")
  cat("n coefficients with |c|> ",threshhold,":",length(good.coef),"\n") 
  #print((cvglm))
  #lX<-variableSelection(lX,ly,"forward",10)  
  
  #fit <- lm(ly~.,data.frame(lX,ly))
  #print(summary(fit)) # show results
  #layout(matrix(c(1,2,3,4),2,2)) # optional 4 graphs/page 
  #plot(fit)
  #print(coefficients(fit))  
}

