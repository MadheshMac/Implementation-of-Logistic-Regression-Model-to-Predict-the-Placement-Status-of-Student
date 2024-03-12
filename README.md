# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values. 
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results.


## Program:
```c
## Developed by: Madheswaran E
## RegisterNumber: 212222040090

import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])


```
## Output:
## Placement Data:

![image](https://github.com/ChandrasekarS22008273/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119643845/19a2bbda-88fc-4013-97fc-9dc77e51fdd6)

## Salary Data:

![IGNORE (1)](https://github.com/ChandrasekarS22008273/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119643845/5d8bc894-164f-45a7-8165-9e522d89e327)

## Checking the null() function:


![image](https://github.com/ChandrasekarS22008273/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119643845/7cb175c5-a0e4-4499-8c6b-a2ed674cd5d6)

## Data Duplicate:


![image](https://github.com/ChandrasekarS22008273/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119643845/630ce03c-c62b-4460-99e0-ba5563106ee9)

## Print Data:

![image](https://github.com/ChandrasekarS22008273/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119643845/ebba46a5-4852-4d8a-849b-8175c2d7d3a6)


## Data-Status:

![image](https://github.com/ChandrasekarS22008273/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119643845/ac9d0b65-221c-497f-82f5-4c78d9c44b17)

## Y_prediction array:
![image](https://github.com/ChandrasekarS22008273/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119643845/4b6b6356-4145-4b71-8bc5-41b469e54fd0)

## Accuracy value:
![image](https://github.com/ChandrasekarS22008273/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119643845/9caef2a8-f7b4-421f-ae31-5526c837691e)

## Confusion array:

![image](https://github.com/ChandrasekarS22008273/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119643845/347af169-8193-4eb5-aa89-f139d59af873)

## Classification Report:

![image](https://github.com/ChandrasekarS22008273/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119643845/e76fb8ec-577e-4bbf-9e75-8fd88b22c0b9)



## Prediction of LR:
![image](https://github.com/ChandrasekarS22008273/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119643845/a02c9e1a-ad76-4822-ba9a-af7ec208d05e)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
