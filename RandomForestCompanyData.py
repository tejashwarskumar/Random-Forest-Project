import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
comapnyData = pd.read_csv("C:/My Files/Excelr/14 - Random Forest/Assignment/Company_Data.csv")
comapnyData.describe()

plt.hist(comapnyData['Sales'])
plt.boxplot(comapnyData['Sales'])

comapnyData = comapnyData.drop(comapnyData.index[[376]])
comapnyData = comapnyData.drop(comapnyData.index[[316]])
plt.hist(comapnyData['Sales'])
plt.boxplot(comapnyData['Sales'])

comapnyData.columns
comapnyData['Sales'].describe()
pd.set_option('display.expand_frame_repr', False) 
comapnyData['Sales'] = np.where(comapnyData['Sales'] > 7.4 ,'High','Low')
comapnyData['Sales'].value_counts()

from sklearn import preprocessing
prepocess = preprocessing.LabelEncoder()
columns = ["ShelveLoc","Urban","US"];
for i in columns:
    comapnyData[i] = prepocess.fit_transform(comapnyData[i])

from sklearn.model_selection import train_test_split
train,test = train_test_split(comapnyData,test_size=0.3)
trainX = train.iloc[:,1:]
trainY = train.iloc[:,0]
testX = test.iloc[:,1:]
testY = test.iloc[:,0]

from sklearn.ensemble import RandomForestClassifier
model1 = RandomForestClassifier().fit(trainX,trainY)
model1
model_pred = model1.predict(trainX)
accuracy1 = np.mean(model_pred == trainY)
accuracy1
model_test_predict = model1.predict(testX)
accuracy_test1 = np.mean(model_test_predict == testY)
accuracy_test1

accuracy_l_val=[];

for i in range(1,50):
    model_l = RandomForestClassifier(n_estimators=i).fit(trainX,trainY)
    model_lpredict = model_l.predict(trainX)
    accuracy_l = np.mean(model_lpredict == trainY)
    model_test_lpredict = model_l.predict(testX)
    accuracy_testl = np.mean(model_test_lpredict == testY)
    accuracy_l_val.append([accuracy_l,accuracy_testl])

plt.plot(np.arange(1,50),[i[0] for i in accuracy_l_val],"bo-")
plt.plot(np.arange(1,50),[i[1] for i in accuracy_l_val],"ro-")
plt.legend(["train","test"])

accuracy_l_val_en=[];

for i in range(1,50):
    model_l_en = RandomForestClassifier(n_estimators=i).fit(trainX,trainY)
    model_lpredict_en = model_l_en.predict(trainX)
    accuracy_l_en = np.mean(model_lpredict_en == trainY)
    model_test_lpredict_en = model_l_en.predict(testX)
    accuracy_testl_en = np.mean(model_test_lpredict_en == testY)
    accuracy_l_val_en.append([accuracy_l_en,accuracy_testl_en])

plt.plot(np.arange(1,50),[i[0] for i in accuracy_l_val_en],"bo-")
plt.plot(np.arange(1,50),[i[1] for i in accuracy_l_val_en],"ro-")
plt.legend(["train","test"])
