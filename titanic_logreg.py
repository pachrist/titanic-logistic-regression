#titanic logistic regression

import numpy as np
from sklearn.linear_model import LogisticRegression
import csv

def gender(s): #vectorizing gender (I know sklearn has functions for this but I know exactly what this does which is safer for now)
    if s == 'male': return 0
    elif s == 'female': return 1
    else: 
        print "there are empty genders" #apparently no empty genders so not a problem here
        return .5                       #if there were this could be a decent lazy mean/median imputation though

def median(L): #for imputing ages, to compare with imputating from mean
    l = len(L)
    L.sort()
    if l%2 != 0: return L[l//2]
    else: return (L[l/2] + L[l/2 + 1])/2

#reading the train/test csv files and getting the data into nested lists
with open('C:\\Users\\Paul\\Downloads\\train.csv','r') as csvfile:
    titanicReader = csv.reader(csvfile, delimiter=',')
    simpleData = []
    data = []
    titanicReader.next()
    for row in titanicReader:
        new_row = []
        for item in row:
            if item.isdigit() == True:
                item = float(item)  
            new_row.append(item)
        data.append(new_row)
    observations = [float(row[5]) for row in data if row[5] != '']
    dummy_age = median(observations) 
    for row in data:
        if row[5] == '': row[5] = dummy_age
        new_row = row[1:3] + [gender(row[4]),float(row[5])]
        simpleData.append(new_row)

with open('C:\\Users\\Paul\\Downloads\\test.csv','r') as csvfile:
    testReader = csv.reader(csvfile, delimiter=',') 
    test = []
    simpleTest = []
    testReader.next()
    for row in testReader:
        new_row = []
        for item in row:
            if item.isdigit() == True:
                item = float(item) 
            new_row.append(item)
        test.append(new_row)
    observations = [float(row[4]) for row in test if row[4] != '']
    dummy_age = median(observations) 
    for row in test:
        if row[4] == '': row[4] = dummy_age
        new_row = [row[1],gender(row[3]),float(row[4])]
        simpleTest.append(new_row)
                
#moving data into numpy matrices
vectorVictor = np.matrix([row[1:4] for row in simpleData])
supervision = np.transpose(np.matrix([row[0] for row in simpleData]).ravel()) #do I need tanspose for 1D array or just for targets?
simpleTest = np.matrix(simpleTest)

#training and testing model
classifier = LogisticRegression()
classifier.fit(vectorVictor, supervision)
predictions = classifier.predict(simpleTest)

#writing results for kaggle upload
with open('C:\\Python\\titanic_cga2_logreg.csv', 'wb') as csvfile:
    titanicWriter = csv.writer(csvfile, delimiter=',')
    titanicWriter.writerow(['PassengerId','Survived'])
    for i,boolean in enumerate(predictions):
        titanicWriter.writerow([i+892,int(boolean)])