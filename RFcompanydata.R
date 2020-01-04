library(caret)
library(randomForest)

CompanyData <- read.csv(file.choose())
hist(CompanyData$Sales)

High = ifelse(CompanyData$Sales<10, "No", "Yes")
CD = data.frame(CompanyData, High)
table(CD$High)

set.seed(123)
ind <- sample(2, nrow(CD), replace = TRUE, prob = c(0.7,0.3))
train <- CD[ind==1,]
test  <- CD[ind==2,]

rf <- randomForest(High~., data=train, proximity = TRUE)
plot(rf)
pred1 <- predict(rf, train)
confusionMatrix(pred1, train$High)
pred2 <- predict(rf, test)
confusionMatrix(pred2, test$High)
hist(treesize(rf), main = "No of Nodes for the trees")
partialPlot(rf, train, Sales, "Yes")
MDSplot(rf, CD$High)

tune <- tuneRF(train[,-6], train[,6], stepFactor = 0.5, plot = TRUE, ntreeTry = 200, trace = TRUE, improve = 0.05)

rf1 <- randomForest(High~., data=train, ntree = 50, mtry = 3, importance = TRUE, proximity = TRUE)
plot(rf1)
pred1 <- predict(rf1, train)
confusionMatrix(pred1, train$High)
pred2 <- predict(rf1, test)
confusionMatrix(pred2, test$High)
hist(treesize(rf1), main = "No of Nodes for the trees")
partialPlot(rf1, train, Sales, "Yes")
MDSplot(rf1, CD$High)
