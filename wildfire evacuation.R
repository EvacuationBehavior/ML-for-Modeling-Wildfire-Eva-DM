# Model Comparison and testing
# Ningzhe Xu
# 9/16/2022

rm(list = ls())
cat("\014")


####machine learning model training####
###load library
library(randomForest) #training RF
library(tree) #traning CART
library(class) #training KNN
library(e1071) #traning SVM and NB
library(xgboost) #training XGBoost
library(nnet) #traning ANN

###loading data and transfer categorical variable to factor
evacuation <- read.csv("demo.csv",header = T)

#data processing
for(i in 1:31){#13:number of adult; 29: prefire precption of safety; 30: risk perception; 31: prior awareness of threat 
  if(i %in% c(13,29:31)){
    evacuation[,i] <- evacuation[,i]
  }else{
    evacuation[,i] <- as.factor(evacuation[,i])
  }
}

####cross-validation####
#build matrix to store the results of each validation
SumvecACC <- matrix(NA, nrow = 100, ncol = 10)
SumvecFONE <- matrix(NA, nrow = 100, ncol = 10)
SumvecPrecision <- matrix(NA, nrow = 100, ncol = 10)
SumvecRecall <- matrix(NA, nrow = 100, ncol = 10)

#5 times 20-fold cross validation
for(k in 1:5){
  set.seed(k^10) #set different random seed for each time
  numHoldouts <- 20
  evacuation<-evacuation[sample(nrow(evacuation)),]
  folds <- cut(seq(from = 1, to = nrow(evacuation)),breaks=20,labels=FALSE)
  for(i in 1:numHoldouts){
    print(i)
    
    testIndexes <- which(folds==i,arr.ind=TRUE)
    testData <- evacuation[testIndexes, ]
    trainData <- evacuation[-testIndexes, ]
    
    #logit
    glm.train <- glm(Evacuation_Decision~.,data = trainData, family = 'binomial')
    prob <- predict(glm.train,newdata = testData,type = 'response')
    pred <- ifelse(prob > 0.5,"1","0")
    pred <- factor(pred,level = c("0","1"))
    CM.logit <- table(pred,testData$Evacuation_Decision)
    TP.logit <- as.numeric(CM.logit[2,2])
    FN.logit <- as.numeric(CM.logit[1,2])
    FP.logit <- as.numeric(CM.logit[2,1])
    TN.logit <- as.numeric(CM.logit[1,1])
    SumvecACC[20*(k-1)+i,1] <- (TP.logit+TN.logit)/nrow(testData)
    SumvecPrecision[20*(k-1)+i,1] <- TP.logit/(TP.logit+FP.logit)
    SumvecRecall[20*(k-1)+i,1] <- TP.logit/(TP.logit+FN.logit)
    SumvecFONE[20*(k-1)+i,1] <-2*SumvecPrecision[20*(k-1)+i,1]*SumvecRecall[20*(k-1)+i,1]/(SumvecPrecision[20*(k-1)+i,1]+SumvecRecall[20*(k-1)+i,1])
    
    #naive bayes
    nb.fit <- naiveBayes(Evacuation_Decision~.,data = trainData, laplace = 1)
    nb.pre <- predict(nb.fit, newdata = testData)
    CM.nb <- table(nb.pre, testData[,10])
    TP.nb <- as.numeric(CM.nb[2,2])
    FN.nb <- as.numeric(CM.nb[1,2])
    FP.nb <- as.numeric(CM.nb[2,1])
    TN.nb <- as.numeric(CM.nb[1,1])
    SumvecACC[20*(k-1)+i,2] <- (TP.nb+TN.nb)/nrow(testData)
    SumvecPrecision[20*(k-1)+i,2] <- TP.nb/(TP.nb+FP.nb)
    SumvecRecall[20*(k-1)+i,2] <- TP.nb/(TP.nb+FN.nb)
    SumvecFONE[20*(k-1)+i,2] <-2*SumvecPrecision[20*(k-1)+i,2]*SumvecRecall[20*(k-1)+i,2]/(SumvecPrecision[20*(k-1)+i,2]+SumvecRecall[20*(k-1)+i,2])
    
    #Decision tree
    tree00 <- tree(Evacuation_Decision~., trainData)
    tree.f1 <- prune.tree(tree00,best = 5)
    yhat <- predict(tree.f1,newdata=testData,type="class")
    mydata.test <- testData[,"Evacuation_Decision"]
    CM.dt.f1 <- table(yhat,mydata.test)
    TP.dt.f1 <- as.numeric(CM.dt.f1[2,2])
    FN.dt.f1 <- as.numeric(CM.dt.f1[1,2])
    FP.dt.f1 <- as.numeric(CM.dt.f1[2,1])
    TN.dt.f1 <- as.numeric(CM.dt.f1[1,1])
    SumvecACC[20*(k-1)+i,3] <- (TP.dt.f1+TN.dt.f1)/nrow(testData)
    SumvecPrecision[20*(k-1)+i,3] <- TP.dt.f1/(TP.dt.f1+FP.dt.f1)
    SumvecRecall[20*(k-1)+i,3] <- TP.dt.f1/(TP.dt.f1+FN.dt.f1)
    SumvecFONE[20*(k-1)+i,3] <-2*SumvecPrecision[20*(k-1)+i,3]*SumvecRecall[20*(k-1)+i,3]/(SumvecPrecision[20*(k-1)+i,3]+SumvecRecall[20*(k-1)+i,3])
    
    #random forest
    rf.train.f1 <- randomForest(Evacuation_Decision~.,data = trainData,ntree = 100,mtry = 20)
    rf.pred <- predict(rf.train.f1,newdata = testData)
    CM.rf.f1 <- table(rf.pred,testData$Evacuation_Decision)
    TP.rf.f1 <- as.numeric(CM.rf.f1[2,2])
    FN.rf.f1 <- as.numeric(CM.rf.f1[1,2])
    FP.rf.f1 <- as.numeric(CM.rf.f1[2,1])
    TN.rf.f1 <- as.numeric(CM.rf.f1[1,1])
    SumvecACC[20*(k-1)+i,4] <- (TP.rf.f1+TN.rf.f1)/nrow(testData)
    SumvecPrecision[20*(k-1)+i,4] <- TP.rf.f1/(TP.rf.f1+FP.rf.f1)
    SumvecRecall[20*(k-1)+i,4] <- TP.rf.f1/(TP.rf.f1+FN.rf.f1)
    SumvecFONE[20*(k-1)+i,4] <-2*SumvecPrecision[20*(k-1)+i,4]*SumvecRecall[20*(k-1)+i,4]/(SumvecPrecision[20*(k-1)+i,4]+SumvecRecall[20*(k-1)+i,4])
    
    #Artificial neural network
    nnet.fit.f <- nnet(Evacuation_Decision~.,data = trainData,size = 31, decay = 0.5)
    prob <- predict(nnet.fit.f,newdata = testData,type = 'class')
    pred <- ifelse(prob > 0.5,"1","0")
    nnet.pred <- factor(pred,level = c("0","1"))
    CM.nnet.f1 <- table(nnet.pred,testData$Evacuation_Decision)
    TP.nnet.f1 <- as.numeric(CM.nnet.f1[2,2])
    FN.nnet.f1 <- as.numeric(CM.nnet.f1[1,2])
    FP.nnet.f1 <- as.numeric(CM.nnet.f1[2,1])
    TN.nnet.f1 <- as.numeric(CM.nnet.f1[1,1])
    SumvecACC[20*(k-1)+i,5] <- (TP.nnet.f1+TN.nnet.f1)/nrow(testData)
    SumvecPrecision[20*(k-1)+i,5] <- TP.nnet.f1/(TP.nnet.f1+FP.nnet.f1)
    SumvecRecall[20*(k-1)+i,5] <- TP.nnet.f1/(TP.nnet.f1+FN.nnet.f1)
    SumvecFONE[20*(k-1)+i,5] <-2*SumvecPrecision[20*(k-1)+i,5]*SumvecRecall[20*(k-1)+i,5]/(SumvecPrecision[20*(k-1)+i,5]+SumvecRecall[20*(k-1)+i,5])
    
    #svm kernal:linear
    svm.linear.f <- svm(Evacuation_Decision~., data = trainData, kernel = 'linear',cost =  1.652405)
    svm.pred <- predict(svm.linear.f,newdata = testData)
    CM.svm.f1 <- table(svm.pred,testData$Evacuation_Decision)
    TP.svm.f1 <- as.numeric(CM.svm.f1[2,2])
    FN.svm.f1 <- as.numeric(CM.svm.f1[1,2])
    FP.svm.f1 <- as.numeric(CM.svm.f1[2,1])
    TN.svm.f1 <- as.numeric(CM.svm.f1[1,1])
    SumvecACC[20*(k-1)+i,6] <- (TP.svm.f1+TN.svm.f1)/nrow(testData)
    SumvecPrecision[20*(k-1)+i,6] <- TP.svm.f1/(TP.svm.f1+FP.svm.f1)
    SumvecRecall[20*(k-1)+i,6] <- TP.svm.f1/(TP.svm.f1+FN.svm.f1)
    SumvecFONE[20*(k-1)+i,6] <-2*SumvecPrecision[20*(k-1)+i,6]*SumvecRecall[20*(k-1)+i,6]/(SumvecPrecision[20*(k-1)+i,6]+SumvecRecall[20*(k-1)+i,6])
    
    #K-nearest neighbors
    train.x <- trainData[,-c(10)]
    test.x <- testData[,-c(10)]
    train.Direction <- trainData[,10]
    test.Direction <-testData[,10]
    
    #knn
    knn.pred.f <- knn (train.x,test.x,train.Direction ,k=4)
    CM.knn.f1 <- table(knn.pred.f,test.Direction)
    TP.knn.f1 <- as.numeric(CM.knn.f1[2,2])
    FN.knn.f1 <- as.numeric(CM.knn.f1[1,2])
    FP.knn.f1 <- as.numeric(CM.knn.f1[2,1])
    TN.knn.f1 <- as.numeric(CM.knn.f1[1,1])
    SumvecACC[20*(k-1)+i,7] <- (TP.knn.f1+TN.knn.f1)/nrow(testData)
    SumvecPrecision[20*(k-1)+i,7] <- TP.knn.f1/(TP.knn.f1+FP.knn.f1)
    SumvecRecall[20*(k-1)+i,7] <- TP.knn.f1/(TP.knn.f1+FN.knn.f1)
    SumvecFONE[20*(k-1)+i,7] <-2*SumvecPrecision[20*(k-1)+i,7]*SumvecRecall[20*(k-1)+i,7]/(SumvecPrecision[20*(k-1)+i,7]+SumvecRecall[20*(k-1)+i,7])
    
    #XGBoost
    data1 <- trainData[,-c(10)]
    for (z in 1:ncol(data1)){
      data1[,z]<-as.numeric(data1[,z])
    }
    data1 <- as.matrix(data1)
    label1 <- as.matrix(trainData[,10])
    xgmat <- xgb.DMatrix(data=data1,label=label1)
    param <- list("objective" = "binary:logistic","eta" = 0.4,"gamma" = 1, "max_depth"=7)
    data2 <- testData[,-c(10)]
    for (z in 1:ncol(data2)){
      data2[,z]<-as.numeric(data2[,z])
    }
    data2 <- as.matrix(data2)
    label2 <- as.matrix(testData[,10])
    xgb.f <- xgb.train(param,xgmat,nrounds = 20 )
    xgb.pre <- predict(xgb.f,newdata = data2)
    pred <- ifelse(xgb.pre > 0.5,"1","0")
    pred <- factor(pred,level = c("0","1"))
    b<-confusionMatrix(pred,testData[,10])
    CM.xgb<-b$table
    TP.xgb <- as.numeric(CM.xgb[2,2])
    FN.xgb <- as.numeric(CM.xgb[1,2])
    FP.xgb <- as.numeric(CM.xgb[2,1])
    TN.xgb <- as.numeric(CM.xgb[1,1])
    
    param <- list("objective" = "binary:logistic","eta" = 0.4,"gamma" = 0.2, "max_depth"=7,"min_child_weight"=2)
    xgb.f <- xgb.train(param,xgmat,nrounds = 2000 )
    xgb.pre <- predict(xgb.f,newdata = data2)
    pred <- ifelse(xgb.pre > 0.5,"1","0")
    pred <- factor(pred,level = c("0","1"))
    b<-confusionMatrix(pred,testData[,10])
    CM.xgb<-b$table
    TP.xgb <- as.numeric(CM.xgb[2,2])
    FN.xgb <- as.numeric(CM.xgb[1,2])
    FP.xgb <- as.numeric(CM.xgb[2,1])
    TN.xgb <- as.numeric(CM.xgb[1,1])
    SumvecACC[20*(k-1)+i,8] <- (TP.xgb+TN.xgb)/nrow(testData)
    SumvecPrecision[20*(k-1)+i,8] <- TP.xgb/(TP.xgb+FP.xgb)
    SumvecRecall[20*(k-1)+i,8] <- TP.xgb/(TP.xgb+FN.xgb)
    SumvecFONE[20*(k-1)+i,8] <-2*SumvecPrecision[20*(k-1)+i,8]*SumvecRecall[20*(k-1)+i,8]/(SumvecPrecision[20*(k-1)+i,8]+SumvecRecall[20*(k-1)+i,8])
    
    #svm kernal:poly
    svm.poly <- svm(Evacuation_Decision~., data = trainData, kernel = 'polynomial',cost =  1,degree = 6,gamma = 0.05,coef0 = 0.5)
    svm.pred <- predict(svm.poly,newdata = testData)
    CM.svm.poly <- table(svm.pred,testData$Evacuation_Decision)
    TP.svm.poly <- as.numeric(CM.svm.poly[2,2])
    FN.svm.poly <- as.numeric(CM.svm.poly[1,2])
    FP.svm.poly <- as.numeric(CM.svm.poly[2,1])
    TN.svm.poly <- as.numeric(CM.svm.poly[1,1])
    SumvecACC[20*(k-1)+i,10] <- (TP.svm.poly+TN.svm.poly)/nrow(testData)
    SumvecPrecision[20*(k-1)+i,10] <- TP.svm.poly/(TP.svm.poly+FP.svm.poly)
    SumvecRecall[20*(k-1)+i,10] <- TP.svm.poly/(TP.svm.poly+FN.svm.poly)
    SumvecFONE[20*(k-1)+i,10] <-2*SumvecPrecision[20*(k-1)+i,10]*SumvecRecall[20*(k-1)+i,10]/(SumvecPrecision[20*(k-1)+i,10]+SumvecRecall[20*(k-1)+i,10])
    
    
    #svm kernal:radio
    svm.radio <- svm(Evacuation_Decision~., data = trainData, kernel = 'radial',cost =  0.01,gamma = 1)
    svm.pred <- predict(svm.radio,newdata = testData)
    CM.svm.radio <- table(svm.pred,testData$Evacuation_Decision)
    TP.svm.radio <- as.numeric(CM.svm.radio[2,2])
    FN.svm.radio <- as.numeric(CM.svm.radio[1,2])
    FP.svm.radio <- as.numeric(CM.svm.radio[2,1])
    TN.svm.radio <- as.numeric(CM.svm.radio[1,1])
    SumvecACC[20*(k-1)+i,11] <- (TP.svm.radio+TN.svm.radio)/nrow(testData)
    SumvecPrecision[20*(k-1)+i,11] <- TP.svm.radio/(TP.svm.radio+FP.svm.radio)
    SumvecRecall[20*(k-1)+i,11] <- TP.svm.radio/(TP.svm.radio+FN.svm.radio)
    SumvecFONE[20*(k-1)+i,11] <-2*SumvecPrecision[20*(k-1)+i,11]*SumvecRecall[20*(k-1)+i,11]/(SumvecPrecision[20*(k-1)+i,11]+SumvecRecall[20*(k-1)+i,11])
    
  }
}

#for SVM, we only pick the best one here


#calculate the average for each model's performance
acc<-colMeans(SumvecACC)
f1<-colMeans(SumvecFONE)
rec<-colMeans(SumvecRecall)
pre<-colMeans(SumvecPrecision)

#get standard deviation
sdAcc<-c()
sdf1<-c()
sdpre<-c()
sdrec<-c()

for(i in 1:11){
  sdAcc<-c(sdAcc,sd(SumvecACC[,i]))
  sdf1<-c(sdf1,sd(SumvecFONE[,i]))
  sdpre<-c(sdpre,sd(SumvecPrecision[,i]))
  sdrec<-c(sdrec,sd(SumvecRecall[,i]))
}

finalresult<- data.frame(acc,sdAcc,
                         pre,sdpre,
                         rec,sdrec,
                         f1,sdf1
)

####paired t-test with bonferroni correct####
###re-built data frame to do the pairwise t-test
Model <- rep(c("Logit","NB","CART","RF","ANN","SVM","KNN","XGBoost"),each = 100)
model.acc <- c(SumvecACC[,1],SumvecACC[,2],SumvecACC[,3],SumvecACC[,4],SumvecACC[,5],SumvecACC[,9],SumvecACC[,7],SumvecACC[,8])
model.f1 <- c(SumvecFONE[,1],SumvecFONE[,2],SumvecFONE[,3],SumvecFONE[,4],SumvecFONE[,5],SumvecFONE[,9],SumvecFONE[,7],SumvecFONE[,8])
model.precision <- c(SumvecPrecision[,1],SumvecPrecision[,2],SumvecPrecision[,3],SumvecPrecision[,4],SumvecPrecision[,5],SumvecPrecision[,9],SumvecPrecision[,7],SumvecPrecision[,8])
model.recall <- c(SumvecRecall[,1],SumvecRecall[,2],SumvecRecall[,3],SumvecRecall[,4],SumvecRecall[,5],SumvecRecall[,9],SumvecRecall[,7],SumvecRecall[,8])
model.performance <- data.frame(Model,model.f1,model.acc,model.precision,model.recall)
model.performance$Model <- as.factor(model.performance$Model)

pairwise.t.test(model.performance$model.f1,model.performance$Model,p.adjust.method = "bonferroni")
pairwise.t.test(model.performance$model.acc,model.performance$Model,p.adjust.method = "bonferroni")
