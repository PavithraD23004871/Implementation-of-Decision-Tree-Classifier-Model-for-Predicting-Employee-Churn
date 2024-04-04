# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the required libraries.
2.Upload and read the dataset.
3.Check for any null values using the isnull() function.
4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.


## Program:

/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Pavithra D

RegisterNumber:  212223230146

*/
```
import pandas as pd
data=pd.read_csv('/content/Employee_EX6.csv')

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level", "last_evaluation", "number_project","average_montly_hours", "time_spend_company", "Work_accident", "promotion_last_5years", "salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train, y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:

1.HEAD

![image](https://github.com/PavithraD23004871/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/138955967/bfdf1c95-695d-489c-aa99-a6f709148aa0)


Data.info():

![image](https://github.com/PavithraD23004871/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/138955967/f1945ed8-1a29-41bb-9606-a702fe7a2846)

isnull() and sum():

![image](https://github.com/PavithraD23004871/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/138955967/8bf35c1b-0d9c-432b-9b8f-6b884965edd5)

Data Value Counts():

![image](https://github.com/PavithraD23004871/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/138955967/c82fe366-4804-4b09-a4d9-c4f835a3b0cc)

Data.head() for salary:

![image](https://github.com/PavithraD23004871/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/138955967/ea4bec3f-c4dc-4cc5-88d2-11c6bd366b01)

x.head:

![image](https://github.com/PavithraD23004871/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/138955967/8718a0e6-ca6c-4353-9e6e-3c5733f12dc2)

Accuracy Value:

![image](https://github.com/PavithraD23004871/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/138955967/402b9103-ff22-4110-aef5-a44fb8a730e5)

Data Prediction:

![image](https://github.com/PavithraD23004871/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/138955967/edeaf433-a215-45dc-84e9-7009a503a27f)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
