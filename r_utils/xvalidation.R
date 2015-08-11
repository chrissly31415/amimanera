#!/usr/bin/Rscript

#creates stratified folds
stratified_folds<-function(dat,k=5) {
	require(plyr)
	createFolds <- function(x,k){
	    n <- nrow(x)
	    x$folds <- rep(1:k,length.out = n)[sample(n,n)]
	    return(x)
	}	  
	true.presence<-dat[,length(dat)]
	folds <- ddply(dat,.(true.presence),createFolds,k = k)
	#Proportion of true.presence in each fold:
	cat("SUMMARY OF STRATIFIED FOLD CREATION\n")
	sum<-ddply(folds,.(folds),summarise,prop = sum(true.presence)/length(true.presence))
	print(sum)
	return(folds)
}

createFoldIndices<-function(x,k){
  n <- nrow(x)
  folds <- rep(1:k,length.out = n)[sample(n,n,replace=F)]
  return(folds)
}


#creates random folds
random_folds<-function(dat,k=5) {
  require(plyr)
  createFolds <- function(x,k){
    n<-nrow(x)
    #samples n numbers from 1 to k
    x$folds <- rep(1:k,length.out = n)[sample(n,n)]
    return(x)
  }	  
  folds<-createFolds(dat,k)
  folds<-data.frame(true.presence=dat[,length(dat)],folds)
  #Proportion of true.presence in each fold:
  #cat("SUMMARY OF RANDOM FOLD CREATION\n")
  #sum<-ddply(folds,.(folds),summarise,prop_mean = round(mean(true.presence)),sdev=round(sd(true.presence)))
  #print(sum)
  return(folds)
}


#returns the samples
return_fold<-function(df,n,test,retSparse=F) {
	if(test==FALSE){
	tmp<-df[df$folds!=n,]
	} else {
	tmp<-df[df$folds==n,]
	}
	rfold<- subset(tmp, select = -c(true.presence,folds) )
  if (retSparse) {
    rfold<-sparse.model.matrix(~ . - 1, data = rfold)
  }
	return(rfold)
}

return_idx<-function(df,n,test) {
	if(test==FALSE){
	tmp<-df$folds!=n
	} else {
	tmp<-df$folds==n
	}
	return(tmp)
}

#returns list of train indices
leave_group_out_cv<-function(data,cv_labels) {
  cat("LGOCV...\n")
  data<-cbind(data,cv_labels=cv_labels)
  subs <- unique(data$cv_labels)
  train_idx_list <- vector(mode = "list", length = length(subs))
  for(i in seq_along(subs)) train_idx_list[[i]] <- which(data$cv_labels != subs[i])
  names(train_idx_list) <- paste0(subs,"")
  oinfo(train_idx_list)
  return(train_idx_list)
}

return_lgocv_idx<-function(train_idx_list,n) {
  return (train_idx_list[[n]])
}


