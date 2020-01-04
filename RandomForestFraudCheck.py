import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fraudData = pd.read_csv("C:/My Files/Excelr/14 - Random Forest/Assignment/Fraud_check.csv")
fraudData.describe()
fraudData.columns = ['Undergrad', 'MaritalStatus', 'TaxableIncome', 'CityPopulation','WorkExperience', 'Urban']

plt.hist(fraudData['TaxableIncome'])
plt.boxplot(fraudData['TaxableIncome'])

fraudData['TaxableIncome'] = np.where(fraudData['TaxableIncome'] <= 30000 , "Risky","Good")
fraudData['TaxableIncome'].value_counts()

from sklearn import preprocessing
prepocess = preprocessing.LabelEncoder()
columns = ["Undergrad","MaritalStatus","Urban"];

for i in columns:
    fraudData[i] = prepocess.fit_transform(fraudData[i])

fraudData.columns
fraudData = fraudData[['Undergrad', 'MaritalStatus', 'CityPopulation','WorkExperience', 'Urban','TaxableIncome']]

from sklearn.model_selection import train_test_split
train,test = train_test_split(fraudData,test_size=0.2)
trainX = train.iloc[:,0:5]
trainY = train.iloc[:,5]
testX = test.iloc[:,0:5]
testY = test.iloc[:,5]

from sklearn.ensemble import RandomForestClassifier
model1 = RandomForestClassifier().fit(trainX,trainY)
model1
model_pred = model1.predict(trainX)
accuracy1 = np.mean(model_pred == trainY)
accuracy1
model_test_predict = model1.predict(testX)
accuracy_test1 = np.mean(model_test_predict == testY)
accuracy_test1

model2 = RandomForestClassifier(n_estimators=15,criterion="entropy").fit(trainX,trainY)
model_pred2 = model2.predict(trainX)
accuracy2 = np.mean(model_pred2 == trainY)
accuracy2
model_test_predict2 = model2.predict(testX)
accuracy_test2 = np.mean(model_test_predict2 == testY)
accuracy_test2

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
