#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 14:09:58 2018

@author: mvg
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#import os.path
import sqlite3
import csv
from ann_ybs5045 import y_pred_olasiliklar

#print(os.getcwd())
global connection
connection = sqlite3.connect('/Volumes/MISUSB/MIS/YBS5045 yazılım geliştirme/gitProjects/ybs5045/ann/anntest.db')

global filePath
filePath = '/Volumes/MISUSB/MIS/YBS5045 yazılım geliştirme/gitProjects/ybs5045/ann'

"""
Kullanici olasiliklarina gore farkli siniflar olustur. 
Normalde binary classification sorununu cozuyor ama olasilik degerlerine gore 
bir siniflama yapilabilir, bunu icin softmax kullanilmali ama veri seti uygun degil
"""
def siniflamisVeri(y_pred):
    ## Anlamsiz ama sonucu manuel olarak birkac sinif yapmak istersek
    y_pred_4categories = np.empty(shape=(len(y_pred),1), dtype = str)
    for i in range(len(y_pred)):
        if(y_pred[i] > 0.75):
            y_pred_4categories[i] = 'g' #'gidici'
        elif(y_pred[i] > 0.5):
            y_pred_4categories[i] = 't' #'takip'
        elif(y_pred[i] > 0.25):
            y_pred_4categories[i] = 'i' #'ikna'
        else:
            y_pred_4categories[i] = 'k' #'kalici'
    return y_pred_4categories

global y_pred_categorized
y_pred_categorized = siniflamisVeri(y_pred_olasiliklar)

#def dbOlustur(dbName):
#    cursor = connection.cursor()
#    ## Create table on result.db
#    cursor.execute('''CREATE TABLE IF NOT EXISTS %s(
#            id INTEGER PRIMARY KEY,
#            date DATETIME NOT NULL DEFAULT (datetime(CURRENT_TIMESTAMP, 'localtime')),
#            username TEXT,
#            password TEXT)''' %dbName)
#    connection.commit()
    
#def kullaniciEkle(username, password):
#    ## arayuz icin kullanici ekleme
#    cursor = connection.cursor()
#    cursor.execute('INSERT INTO kullanici(username, password) VALUES(?, ?)',(username, password))
#    connection.commit()

def dbErisim(tableName):
    cursor = connection.cursor()
    #sorgu = "SELECT * FROM results"
    sonuc = cursor.execute("SELECT * FROM %s" %tableName)
    #return sonuc
    return [row for row in sonuc]

def tabloHeaderAlma(tableName):
    ## Tablonun header bilgilerini alma
    cursor = connection.cursor()
    sorgu = 'PRAGMA table_info('+ tableName + ')'
    sonuc = cursor.execute(sorgu)
    return [row[1] for row in sonuc]

def csvOlustur(tableName):
    ## Creates a csv file from the data of given TABLENAME on DB for graph
#    print(dbErisim(tableName))
    fileName = filePath + '/' + tableName + 'Data.csv'
    with open(fileName, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',quotechar=',', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(tabloHeaderAlma(tableName))
        for item in dbErisim(tableName):
            spamwriter.writerow(item)

def veriCekme(columnName, value):
    cursor = connection.cursor()
    sonuc = cursor.execute('SELECT * FROM results WHERE %s = %s' % (columnName, str(value))).fetchall()
    return (sonuc)
    
def butunVeri():
    cursor = connection.cursor()
    sonuc = cursor.execute('SELECT * FROM results').fetchall()
    return (sonuc)
    

def AccuracyDagilim():
    accAll = []
    testAll = []
    for item in butunVeri():
        accAll.append(item[2])
        testAll.append(item[3])
#    print('accuracy for all: ',accAll )
#    print('test for all: ',testAll )
    
    ## Accuracy dagilim grafikleri
    plt.scatter(testAll, accAll, label = 'accuracy', color = 'blue')
    plt.scatter(testAll[len(testAll)-1], accAll[len(accAll)-1], label = 'Last training accuracy', color = 'red')
    plt.ylim(bottom = min(accAll)-0.001, top = max(accAll)+0.001)
    plt.xlim(min(testAll)-200, max(testAll)+200)
    plt.xticks([1000,1500,2000,2500,3000])
    plt.xlabel('Test size')
    plt.ylabel('Accuracy')
    plt.title('Test Size - Accuracy')
    plt.rc('grid', linestyle = 'dashed', color = 'black')
    plt.grid(True)
    plt.legend()
    plt.show()


def tekliAccuracy():
    ## CSV dosyasinda veri cekerek grafik
    dataPath = filePath + '/resultsData.csv'
#    print(dataPath)
    #data = pd.read_csv(dataPath)
    data = pd.read_table(dataPath, sep=',')
#    print(data)
#    print(data['accuracy'])
#    print(data['test_size'])
    
    ## Boolean deger ile pandas dataframeden veri cekme
    ## test_size 2000
    t2000 = data['test_size'] == 2000
    ## accuracy
    #a2000 = data['accuracy'] > 0
    test2000 = data[t2000]
    
    ## Tek tek accuracy grafikleri
    #plt.plot(test, acc, label='accuracy')
    plt.ylim(0.831,0.889)
    #plt.bar(test, acc, label='accuracy')
    plt.plot(test2000['accuracy'], label = 'accuracy', color = 'blue')
    plt.xlabel('Test')
    plt.ylabel('Accuracy')
    plt.title('Tekli Accuracy Test Size 2000 - Accuracy')
    plt.legend()
    plt.show()

def siniflamaGrafigi():
    ## Siniflama grafikleri:
    ## Called a variable from another script
#    print(y_pred_categorized)
    category_values= {}
    for item in y_pred_categorized[:]:
        if(item[0] in category_values):
            category_values[item[0]] += 1
        else:
            category_values[item[0]] = 1
#    print(category_values)
    
    categories = []
    for item in category_values.keys():
        if item == 'k':
            categories.append('kalici')
        if item == 'g':
            categories.append('dondurulemez')
        if item == 'i':
            categories.append('kalma egiliminde')
        if item == 't':
            categories.append('gitme egiliminde')
        
    values = list(category_values.values())
#    print(categories)
#    print(values)
    
    ### Dikey 
    #plt.bar(categories, values)
    #for i,v in enumerate(values):
    #    plt.text(str(v))
    #plt.legend()
    #plt.show()
    
    ## Yatay  - daha basarili
    fig, ax = plt.subplots()    
    width = 0.75 # the width of the bars 
    ind = np.arange(len(values))  # the x locations for the groups
    ax.barh(ind, values, width, color="blue")
    ax.set_yticks(ind+width/2)
    ax.set_yticklabels(categories, minor=False)
    for i, v in enumerate(values):
        ax.text(v + 3, i + .25, str(v), color='red')
    plt.title('2000 Musteri Egilim Grafigi')
    plt.xlabel('Musteri Sayisi')
    plt.ylabel('Egilim')      
    plt.show()

def tekliAccuracyVeriCekme():
    acc = []
    testDate = []
    for item in veriCekme('test_size', 2000):
        acc.append(item[2])
        testDate.append(item[1])
#    print('accuracy for test size 2000: ',acc )
#    print('testDate: ',testDate )

## Csv file olusturma
csvOlustur('results')

## Butun dataset Accuracy dagilim grafigi
AccuracyDagilim()

## Tekli Accuracy dagilim
tekliAccuracy()

## Siniflama Grafigi
siniflamaGrafigi()