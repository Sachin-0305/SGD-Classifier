# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
STEP 1.START

STEP 2.Import Necessary Libraries and Load Data

STEP 3. Dataset into Training and Testing Sets

STEP 4.Train the Model Using Stochastic Gradient Descent (SGD)

STEP 5.Make Predictions and Evaluate Accuracy

STEP 6.Generate Confusion Matrix

STEP 7.END

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: SACHIN M 
RegisterNumber:  212223040177
*/

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#load the dataset
iris = load_iris()

#create a pandas dataframe
df=pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target']=iris.target

print(df.head())

#split the data into features x and target y
x=df.drop('target',axis=1)
y= df['target']

#split the data into training and testing set
x_train, x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)

#create sgd classifier with default parameter
sgd_clf = SGDClassifier(max_iter = 1000, tol= 1e-3)

#train the classifier on the training data
sgd_clf.fit(x_train, y_train)

#make predictions on the testing data
y_pred = sgd_clf.predict(x_test)

# evaluate the classifier's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy:{accuracy:.3f}")

#calculate the confusion matrix
cf=confusion_matrix(y_test, y_pred)
print("Confusion Matrix")
print(cf)

```

## Output:
![Screenshot 2024-09-19 160012](https://github.com/user-attachments/assets/29c264e3-71a0-45a1-a812-4a38f6486fa7)

## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
