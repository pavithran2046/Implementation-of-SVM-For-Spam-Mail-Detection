# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.
2.Import the dataset to operate on.
3.Split the dataset.
4.Predict the required output.


## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: vinodhini k
RegisterNumber: 212223230245 
*/
```
````
import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result


import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
``````````

## Output:
![image](https://github.com/user-attachments/assets/b4a2a24c-5c8c-4ff1-af8b-d5ca42f9e85b)
````
data.head():
```````
![image](https://github.com/user-attachments/assets/23a24c0a-e309-4aec-b2c6-81e71ed61c51)
````
data.info():
``````````
![image](https://github.com/user-attachments/assets/d5a9afa1-1421-4032-bc0e-0a19ed3e9892)
``````
data.isnull().sum():
````````
![image](https://github.com/user-attachments/assets/a5b6ec99-ecf7-46cb-b291-98c1567c0e75)
````````
Y_prediction value:
`````````````
![image](https://github.com/user-attachments/assets/4966a957-6aa5-40b6-9b7e-766de9f9b526)

`````
Accuracy value:
``````````
![image](https://github.com/user-attachments/assets/e69aa8f8-5265-4d43-9f76-a7b04232d1d8)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
