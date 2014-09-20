#!/usr/bin/Rscript


gaFeatureSelection<-function(lX,ly) {
  require(GA)
  popsize=50
  initPop<-fillPop(lX,lY,popsize)
  GARES <- ga("binary", fitness = fitnessFNC,x=lX,y=ly,popSize=popsize, maxiter=25,seed=1000, nBits = ncol(lX), pcrossover = 0.8,pmutation = 0.1,monitor = plot,suggestions=initPop)
  plot(GARES)
  print(GARES)
  print(summary(GARES))
  cat("Solution:")
  soln<-as.vector(attr(GARES,"solution"))
  cat(soln,"\n")
  Xs<-lX[,which(soln == 1)]
  for (i in 1:ncol(Xs)) {
    cat("\"",names(Xs)[i],"\"",sep="")
    if (i!=ncol(Xs)) cat(",",sep="")
  }
  
}

fitnessFNC <- function(bitstring,x,y) {
  lossfnc="auc"
  niter=250
  #cat(bitstring,"\n")
  inc <- which(bitstring == 1)      
  Xs<-x[,inc] 
  cat("c(")
  for (i in 1:ncol(Xs)) {
    cat("\"",names(Xs)[i],"\"",sep="")
    if (i!=ncol(Xs)) cat(",",sep="")
  }
  cat(")\n")
  print(system.time(model<-trainRF(Xs,y,niter,verbose=F))) 
  #model<-trainRF(Xs,y,niter,verbose=F)
  if (lossfnc=="auc") {
    tmp<-model$votes[,2]
    tmp<-as.numeric(as.character(tmp))
    loss<-computeAUC(tmp,y,F) 
    cat("AUC:",loss,"\n")
  } else {
    loss<-compRMSE(model$predicted,model$y)
    cat("RMSE:",loss,"\n")
    #usually fitness gets MAXIMIZED
    loss<--loss
  }
  #write.table(data.frame(auc=auc,t(bitstring)),file="gabits.csv",sep=",",row.names=FALSE,col.names=F,append=T)
  #return(max(model$cvm))
  return(loss)  
}


fillPop<-function(lX,ly,popsize) {
  ##create initial population  
  initPop <- matrix(0, popsize, ncol(lX))
  initPop<-apply(initPop, c(1,2), function(x) sample(c(0,1),1))
  return(initPop)
}