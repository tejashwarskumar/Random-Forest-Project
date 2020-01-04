library(caret)
library(randomForest)

FraudCheck <- read.csv(file.choose())
hist(FraudCheck$Taxable.Income)

Risky_Good = ifelse(FraudCheck$Taxable.Income<= 30000, "Risky", "Good")
FC= data.frame(FraudCheck,Risky_Good)
table(FC$Risky_Good)

set.seed(123)
ind <- sample(2, nrow(FC), replace = TRUE, prob = c(0.7,0.3))
train <- FC[ind==1,]
test  <- FC[ind==2,]

rf <- randomForest(Risky_Good~., data=train, proximity = TRUE)
plot(rf)
pred1 <- predict(rf, train)
confusionMatrix(pred1, train$Risky_Good)
pred2 <- predict(rf, test)
confusionMatrix(pred2, test$Risky_Good)
hist(treesize(rf), main = "No of Nodes for the trees")
partialPlot(rf, train, Taxable.Income, "Good")
MDSplot(rf, FC$Risky_Good)

tune <- tuneRF(train[,-6], train[,6], stepFactor = 0.5, plot = TRUE, ntreeTry = 200, trace = TRUE, improve = 0.05)

rf1 <- randomForest(Risky_Good~., data=train, ntree = 200, mtry = 2, importance = TRUE, proximity = TRUE)
plot(rf1)
pred1 <- predict(rf1, train)
confusionMatrix(pred1, train$Risky_Good)
pred2 <- predict(rf1, test)
confusionMatrix(pred2, test$Risky_Good)
hist(treesize(rf1), main = "No of Nodes for the trees")
partialPlot(rf1, train, Taxable.Income, "Good")
MDSplot(rf1, FC$Risky_Good)
