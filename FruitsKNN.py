#Importing Libraries and fruits data set

%matplotlib notebook
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

#importing CSV file
fruits = pd.read_csv(FILEPATH)

#create a dictionary to assign fruit name with label
lookup_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))

#Creating train- test split
X = fruits [["mass", "width", "height"]]
y = fruits [["fruit_label"]]
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 0)

#creating classifiers and setting KNN as "5"
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier (n_neighbors = 5)
knn.fit(X_train, y_train)

#estimate the fit of the test set with the training set 
knn.score (X_test, y_test)

#using the predictor for a new value
fruit_prediction = knn.predict([[20, 4.3, 5.5]])
lookup_fruit_name[fruit_prediction[0]]


#testing and plotting range of k values to check which would be best fit
k_range = range(1,20)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0,5,10,15,20]);


##end 
