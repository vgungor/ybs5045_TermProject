#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 13:26:21 2018

@author: mvg
"""

#PART 1 - Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Importing dataset
dataset = pd.read_csv('/Volumes/MISUSB/MIS/YBS5045 yazılım geliştirme/gitProjects/ybs5045/ann/bankaDB.csv')
#print(type(dataset))
## Selecting data by row numbers (.iloc) from PANDAS library
X = dataset.iloc[:, 3:13].values
## Alternative to row selection is selecting all necessary rows by name
#X = dataset[['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']]
y = dataset.iloc[:, 13].values

## Dealing with categorical data
## Encoding the Independent Variable - 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#will change text data to numbers for calculations
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
## categorical data are not ordinal - create dummy variables for columns
## if categorical data has more than 2 categories, data needed to be encoded 
## not to have higher effect on calculations
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
## remove one dummy variable to avod dummy variable trap(DVT)
X = X[:, 1:]

## Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
testSize = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= testSize, random_state = 0)

## Feature Scaling - independent variables are scaled
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


## PART 2 - ANN
## importing keras library and packages
import keras
## Sequential module - for initializing ANN
from keras.models import Sequential
## Dense - to create layers in ANN
from keras.layers import Dense

## Initializing ANN - graph or sequence of layers
## ANN named as classifier as it will be classification
classifier = Sequential()

## Adding layers of ANN

## input layer and first hidden layer - adds hidden layer and input layer is evaluated from input value
## number of hidden layer node is calculated by average of input + output nodes
## old version classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11 ))
## adding inuut layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11 ))

## adding 1st hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

## adding output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

## compiling ANN - applying sthocastic gradient descent
classifier.compile(optimizer = 'adam', loss ='binary_crossentropy', metrics = ['accuracy'] )

## running ANN
batch_size = 10
epochs = 100
classifier.fit(X_train, y_train, batch_size, epochs)


## PART 3 - Predictions and evaluating model
## Predicting the Test set results - probabilities of output
'''
False - bankayi terk etmez
True - bankayi terk eder
'''
## olasilikli sonuc
y_pred = classifier.predict(X_test)
y_pred_olasiliklar = y_pred
## true/false durumu
y_pred_state = (y_pred > 0.5)

## Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
'''
yazdirirken accuracy ve test size verisine gore yazdir
'''
accuracy = (cm[0][0]+cm[1][1]) / len(y_pred)
realTestSize = testSize * len(X)
print('_'*50)
print(str(int(realTestSize)) + " test verisinde;" + str(cm[0][0]+cm[1][1]) + " dogru sonuc")
print('accuracy: ', accuracy)


"""
uniform testini yap
"""


"""
write the each result of model training accuracy to a database and graph
accuracy, test size, date(directly to DB), batch_size, epoch
"""
import sqlite3 as sq
connection = sq.connect('/Volumes/MISUSB/MIS/YBS5045 yazılım geliştirme/gitProjects/ybs5045/ann/anntest.db')
cursor = connection.cursor()

# Create table on result.db
cursor.execute('''CREATE TABLE IF NOT EXISTS results(
        id INTEGER PRIMARY KEY,
        date DATETIME NOT NULL DEFAULT (datetime(CURRENT_TIMESTAMP, 'localtime')),
        accuracy FLOAT,
        test_size INTEGER, 
        batch_size INTEGER, 
        epoch INTEGER)''')

# Insert  test results to DB after each test
cursor.execute('''INSERT INTO results(accuracy, test_size, batch_size, epoch) VALUES(%f,%d,%d,%d)''' % (accuracy, realTestSize, batch_size, epochs))
connection.commit()
connection.close()

#"""
#Kullanici olasiliklarina gore farkli siniflar olustur. 
#Normalde binary classification sorununu cozuyor ama olasilik degerlerine gore 
#bir siniflama yapilabilir, bunu icin softmax kullanilmali ama veri seti uygun degil
#"""
#def siniflamisVeri(y_pred):
#    ## Anlamsiz ama sonucu manuel olarak birkac sinif yapmak istersek
#    y_pred_4categories = np.empty(shape=(len(y_pred),1), dtype = str)
#    for i in range(len(y_pred)):
#        if(y_pred[i] > 0.75):
#            y_pred_4categories[i] = 'g' #'gidici'
#        elif(y_pred[i] > 0.5):
#            y_pred_4categories[i] = 't' #'takip'
#        elif(y_pred[i] > 0.25):
#            y_pred_4categories[i] = 'i' #'ikna'
#        else:
#            y_pred_4categories[i] = 'k' #'kalici'
#    return y_pred_4categories
#
#y_pred_categorized = siniflamisVeri(y_pred_olasiliklar)

"""
kullanici girisine gore tahminde bulun, test sonucuna gore verileri - tahmin tablosuna yaz
"""
# array should be one row thus 2D array is used
customer_data = np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])
#perdictTestdata = np.array([[600,'France','Male',40,3,60000,2,1,1,50000]])
new_prediction = classifier.predict(sc.transform(customer_data))
new_prediction = (new_prediction > 0.5)
if(new_prediction == True):
    print('Musteri banka ile calismaya devam etmeyecek!!!')
else:
    print('Musteri banka ile calismaya devam edecek :) ')

## Prediction for more than one customers from a csv file
#perdictTestcsv = pd.read_csv('/Volumes/MISUSB/MIS/YBS5045 yazılım geliştirme/gitProjects/ybs5045/ann/predictTestDB.csv')
#perdictTestdata = perdictTestcsv.iloc[:, 3:13].values
#perdictTestdata[:, 1] = labelencoder_X_1.fit_transform(perdictTestdata[:, 1])
#perdictTestdata[:, 2] = labelencoder_X_2.fit_transform(perdictTestdata[:, 2])
#perdictTestdata = onehotencoder.fit_transform(perdictTestdata).toarray()
#perdictTestdata = perdictTestdata[:, 1:]
#perdictTestdata = sc.transform(perdictTestdata)
#predictions = classifier.predict(perdictTestdata)
#predictions = (predictions > 0.5)