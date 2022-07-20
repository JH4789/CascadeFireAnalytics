#TF + Keras
import csv
import statistics
import numpy as np
from array import array
import matplotlib.pyplot as plt
import sys
import math
import tensorflow as tf
from sklearn import preprocessing
from sklearn import datasets, linear_model
from sklearn.neural_network  import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import datasets
from tensorflow.keras import Sequential
from keras.layers.core import Dense
import datetime as DT
import matplotlib.dates as mdates
"""
Note index 71 is for flame length
Index 73 is fire size?
"""
def main():
    np.random.seed(0)
    maximum = sys.maxsize
    masterdata = []
    count = 0
    with open('ODF_Fire.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        for row in csv_reader:            
            if count != 0:
                newaddition = []
                for column in row:
                    newaddition.append(column)
                masterdata.append(newaddition)
            count +=1
    filler,filler,filler,unit,ignition, serial,  = filterInt(masterdata,0,0,0,8,54,2)
    #All ignition time entries after 2008 go boom using the below line of code
    #flamelength, firesize,longitude,latitude,ignition,serial = filterInt(masterdata, 63,73,23,24,54,2)
    ignition = processTimes(ignition)
    print(ignition)
    newarray = []
    counter = 0
    print(webScrape('2012/4/6','2020/5/6'))
    """
    for row in flamelength:
        newarray.append([int(row)])
        counter+=1
    max = 0
    count = 0
    
    for x in range(1,16):
        newint = linearreg(newarray,list2,x*1000)
        
    if newint > max:
            max = newint
    """
    print(dataValidate(masterdata,serial,ignition,54))
    #newint = linearreg(newarray, firesize, 16000)
    #perceptron(newarray, list2)
    #keras(list1,list2)
def dataValidate(masterdata, serial, inputlist, index):
    count = 0
    valid = True
    for row in masterdata:
        if serial == row[0]:
            if inputlist[count] == row[index]:
                valid = valid
            else:
                valid = False
            count +=1
    return valid

# fit the model

#Linear reg seems to be working better, further testing needed
def linearreg(list1, list2,numeral):

    X_train = list1[0:numeral]
    y_train = list2[0:numeral]
    X_test = list1[numeral:numeral+200]
    y_test = list2[numeral:numeral+200]
    regr = linear_model.BayesianRidge()
    regr.fit(X_train, y_train)
    results = regr.predict(X_test)
    listresult = results.tolist()
    #print(listresult)
    plt.scatter(X_test, y_test, color =  "blue")
    plt.scatter(X_test, results, color = "red")
    plt.show()
    print("Coeffiencet: %.2f" % r2_score(y_test,results))
    return r2_score(y_test,results)
def scale_data(inputdata):
    newarray = []
    for row in inputdata:
        newarray.append(row*100)
    return newarray
#Somethign is working here, have to keep messing around remember to add other types of input data, write new function to make sure that all the data stays aligned
#MLP might not be the most optimal thing, sklearn may not deliver on the functionalities I need
def perceptron(newarray, list2):
    X= newarray
    testsize = 0.3
    random_state = 0    
    y = list2
    y = np.asarray(y)
    print(y[0:20])
    np.expand_dims(y, -1)
    #X = np.asarray(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    y_train = scaler.fit_transform(y_train.reshape(-1,1)).flatten()
    print(y_train[0:20])
    X_train = X_train[0:1000]
    y_train = y_train[0:1000]
    
    y_test = scaler.transform(y_test.reshape(-1, 1)).flatten()
    ppn = MLPClassifier(hidden_layer_sizes = (3000), activation = "logistic",max_iter = 200,learning_rate_init = 0.001, solver = "lbfgs")
    ppn.fit(X_train, y_train.astype(int))

    
    y_pred = ppn.predict(X_test)
    print(y_pred)
    
def getLongitude(masterdata):
    newarray = []
    for row in masterdata:
        if row[23] != 'Longitude' and row[23] != '':
            newarray.append(float(row[23]))
    return newarray
def getLatitude(masterdata):
    newarray = []
    for row in masterdata:
        if row[24] != 'Latitude' and row[24] != '':
            newarray.append(float(row[24]))
    return newarray
def findMin(array):
    final = 100000000
    for number in array:
        if number < final:
            final = number
    return final
#Returns good data
def webScrape(startyear, endyear):
    starting = startyear.split('/')
    ending = endyear.split('/')
    dates = mdates.num2date(mdates.drange(DT.datetime(int(starting[0]), int(starting[1]), int(starting[2])),
                                          DT.datetime(int(ending[0]), int(ending[1]), int(ending[2])),
                                      DT.timedelta(days=1)))
    newdates = []
    for row in dates:
        temp = row.split(' ')
        newdates.append(temp)
    return newdates
def filterInt(masterdata, index1, index2,index3,index4,index5,index6):
    newarray1 = []
    newarray2 = []
    newarray3 = []
    newarray4 = []
    newarray5 = []
    newarray6 = []
    
    for row in masterdata:
        if row[index1] != '' and row[index2] != '' and row[index3] != '' and row[index4] == "51" and row[index5] != '' and row[index6] != '':
            newarray1.append(float(row[index1]))
            newarray2.append(float(row[index2]))
            newarray3.append(float(row[index3]))
            newarray4.append(float(row[index4]))
            newarray5.append(row[index5])
            newarray6.append(row[index6])
        
    return newarray1 , newarray2, newarray3, newarray4, newarray5,newarray6
def graphRange(masterdata, minsize, maxsize, inputcolor):
    acres_long = filterSize(masterdata, minsize, maxsize, 23)
    acres_lat = filterSize(masterdata, minsize, maxsize, 24)
    plt.scatter(acres_long, acres_lat, color = inputcolor)    
def processTimes(inputlist):
    newlist = []
    for row in inputlist:
        temp = row.split(" ")
        processed = temp[0].split('+')
        newlist.append(processed[0])
    return newlist
#Cleanup happened here
def keras(X,y):
    y = np.asarray(y)
    np.expand_dims(y, -1)
    X = np.asarray(X)
    np.expand_dims(X,-1)
    X_train = X[:-200]
    X_test = X[-200:]
    y_train = y[:-200]
    y_test = y[:-200]
    print(len(X_train))
    model = Sequential()
    model.add(Dense(512, input_shape=(16552,), activation='relu'))
    model.add(Dense(768, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=100)

main()

