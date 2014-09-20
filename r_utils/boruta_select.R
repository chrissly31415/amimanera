#!/usr/bin/Rscript

boruta_select<-function(lX,ly) {
  library(Boruta)
  mydata=data.frame(lX,target=ly)
  summary(mydata)
  Boruta(target~.,data=mydata,doTrace=2)->Boruta.results
  #Nonsense attributes should be rejected
  print(Boruta.results);
  plotImpHistory(Boruta.results)
  str(Boruta.results$finalDecision);
  validCols<-getSelectedAttributes(Boruta.results);
  tmpframe<-subset(mydata,select=validCols)
  print(summary(tmpframe))
  #tmpframe<-data.frame(tmpframe,target=mydata$target)
  write.table(tmpframe,file="boruta_features.csv",sep=",",row.names=FALSE)
  return(tmpframe)
}

