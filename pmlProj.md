Prediction assignment
========================================================







```r
library(caret)
library(randomForest)
set.seed(1234)
```


The training data (downloaded from https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv) is loaded, and then the first 6 variables are removed. These variables are labels rather than predictors, so they are not useful for a prediction algorithm. Additionally the predictors that have more than half the values missing (na) are removed.

Finally, the data is split into 80% training and 20% testing sets.



```r
training<-read.csv('pml-training.csv')
#remove username etc.
train2<-training[,-c(1:6)]
#remove variables with more than hlf na
highNA<- sapply(train2, function(x) sum(is.na(x)))/dim(train2)[1]
train3<-train2[,highNA<.5]


inMyTrain<-createDataPartition(y=train3$classe,p=.8,list=F)
myTrain<-train3[inMyTrain,]
myTest<-train3[-inMyTrain,]
```

The preprocessing steps we do on the data are, firstly removing the predictors with near zero variance, using the nearZeroVar function, and secondly doing principal component analysis (PCA) to find out how many components are required for 80%, 90%, and 95% of the variance. The PCA in this chunk is diagnostic only, as the train method in caret has PCA options included.


```r
#remove  classe
classeindex<-match('classe',names(myTrain))
myTrain2<-myTrain[,-c(classeindex)]
nzv<-nearZeroVar(myTrain2)
myTrain3<-myTrain2[,-c(nzv)]

#same on my test
myTest2<-myTest[,-c(classeindex)]
myTest3<-myTest2[,-c(nzv)]
myTest3$classe<-myTest$classe

PCA80 <- preProcess(myTrain3,method="pca",thresh=.8)
PCA90 <- preProcess(myTrain3,method="pca",thresh=.9)
PCA95 <- preProcess(myTrain3,method="pca",thresh=.95) 
PCA80 #13 components
```

```
## Created from 15699 samples and 53 variables
## 
## Pre-processing:
##   - centered (53)
##   - ignored (0)
##   - principal component signal extraction (53)
##   - scaled (53)
## 
## PCA needed 13 components to capture 80 percent of the variance
```

```r
PCA90 #19 components
```

```
## Created from 15699 samples and 53 variables
## 
## Pre-processing:
##   - centered (53)
##   - ignored (0)
##   - principal component signal extraction (53)
##   - scaled (53)
## 
## PCA needed 19 components to capture 90 percent of the variance
```

```r
PCA95 #25 components
```

```
## Created from 15699 samples and 53 variables
## 
## Pre-processing:
##   - centered (53)
##   - ignored (0)
##   - principal component signal extraction (53)
##   - scaled (53)
## 
## PCA needed 25 components to capture 95 percent of the variance
```

```r
myTrain3$classe<-myTrain$classe
```

We train our model using the random forest method in the caret library. This method has the advantage that it has in-built cross validation.

The preprocessing used on this data is principal component analysis encompassing 80% of the variance. This is chosen to reduce the load on the computer, as this training is quite intensive.

The model is then used to predict the outcome on the training and test sets.



```r
modFit<-train(classe~.,myTrain3,method='rf', preProcess='pca', trControl = trainControl(preProcOptions = list(thresh = 0.8 ) ) )

testPred<- predict(modFit,myTest3)
trainPred<- predict(modFit,myTrain3)
confusionMatrix(trainPred, myTrain3$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4464    0    0    0    0
##          B    0 3038    0    0    0
##          C    0    0 2738    0    0
##          D    0    0    0 2573    0
##          E    0    0    0    0 2886
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9998, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

```r
confusionMatrix(testPred, myTest3$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1102   14    5    4    2
##          B    5  734   12    1    4
##          C    4   10  657   32    6
##          D    3    0    8  606    7
##          E    2    1    2    0  702
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9689         
##                  95% CI : (0.963, 0.9741)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9606         
##  Mcnemar's Test P-Value : 0.0006097      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9875   0.9671   0.9605   0.9425   0.9736
## Specificity            0.9911   0.9930   0.9839   0.9945   0.9984
## Pos Pred Value         0.9778   0.9709   0.9267   0.9712   0.9929
## Neg Pred Value         0.9950   0.9921   0.9916   0.9888   0.9941
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2809   0.1871   0.1675   0.1545   0.1789
## Detection Prevalence   0.2873   0.1927   0.1807   0.1591   0.1802
## Balanced Accuracy      0.9893   0.9801   0.9722   0.9685   0.9860
```

As we can see, the model has 100% accuracy on the training set, which makes us wary of overfitting, though it still achieves 97% accuracy on the test set, so it is not a bad model


Now we get the predictions from the final test set


```r
testing<-read.csv('pml-testing.csv')
#remove username etc.
test2<-testing[,-c(1:6)]

test3<-test2[,highNA<.5]

test4<-test3[,-c(nzv)]


finalTest<- predict(modFit,test4)

finalTest
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
