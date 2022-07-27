#TF + Keras
#NEED UPDATE HERE PLEASE!!!!!!!!! UPDATE MEDIUM
#RANDOM FOREST (ONE POSSIBLE METHOD)
#FIX ERRORS IN WEB SCRAPE (TRY-CATCH)
#Joint Probability Mass Distribution?, this may be promising to actually generate a risk index
#Look into SciPy
import timeit
import csv
import urllib.request
import statistics
import numpy as np
from array import array
import matplotlib.pyplot as plt
import sys
import math
from statistics import mean
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
import statistics
from scipy.stats import multivariate_normal
from sklearn.ensemble import RandomForestClassifier
"""
Note index 71 is for flame length
Index 73 is fire size?
"""
def main():
    start = timeit.default_timer()
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
    
    #All ignition time entries after 2008 go boom using the below line of code
    #flamelength, firesize,longitude,latitude,ignition,serial = filterInt(masterdata, 63,73,23,24,54,2)
    
    
    bigWrapper(masterdata, '2012/2/10','2020/9/1', 52, 'Astoria','astoria.csv')
    #linearreg(environment,finalcount,50 )
    stop = timeit.default_timer()
    print('Time: ', stop - start)  
    #newint = linearreg(newarray, firesize, 16000)
    
    #keras(list1,list2)
def bigWrapper(masterdata, start , end , citynum, cityname, filename):
    filler,filler,filler,unit,ignition, serial,  = filterInt(masterdata,0,0,0,8,54,2,citynum)
    ignition = processTimes(ignition, start,end)
    loaded = loadData(filename)
    
    finalcount = []
    environment = []
    counter = 0
    #scraped = webScrape(start,end, cityname)
    countarray = countFires(ignition, loaded)
    
    for row in loaded:
        temp = []
        temp.append(row[1])
        temp.append(row[2])
        environment.append(temp)
    
    for row in countarray:
        finalcount.append(row[1])
    """
    with open( 'astoria.csv', 'w') as filewrite:
        writer = csv.writer(filewrite)
        writer.writerow(['Date','Temperature' ,'Wind Speed'])
        for row in scraped:
            data = []
            data.append(row[0])
            data.append(row[1])
            data.append(row[2])
            writer.writerow(data)
    """
    randomForest(environment, finalcount)
def countFires(ignition, scraped):
    newarray = []
    for row in scraped:
        temp = []
        temp.append(row[0])
        count = ignition.count(row[0])
        if count >= 1:
            count = True
        else:
            count = False
        temp.append(count)
        newarray.append(temp)
    return newarray
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

def randomForest(environment, finalcount):
    X = np.asarray(environment)
    print(X)
 
    y = np.asarray(finalcount)
    print(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    clf = RandomForestClassifier(n_estimators = 100)
    clf.fit(X_train, y_train)
    print(clf.predict(X_test))
    print(clf.score(X_test, y_test))
#Linear reg seems to be working better, further testing needed
def linearreg(list1, list2,numeral):

    X_train = list1[0:numeral]
    y_train = list2[0:numeral]
    X_test = list1[numeral:numeral+20]
    y_test = list2[numeral:numeral+20]
    regr = linear_model.BayesianRidge()
    regr.fit(X_train, y_train)
    results = regr.predict(X_test)
    listresult = results.tolist()
    #print(listresult)
    """
    plt.scatter(X_test, y_test, color =  "blue")
    plt.scatter(X_test, results, color = "red")
    plt.show()
    """
    print("Coeffiencet: %.2f" % r2_score(y_test,results))
    return r2_score(y_test,results)
def scale_data(inputdata):
    newarray = []
    for row in inputdata:
        newarray.append(row*100)
    return newarray
def loadData(filename):
    newarray = []
    count = 0
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        for row in csv_reader:
            if count % 2 == 0 and count != 0:
                temp = []
                temp.append(row[0])
                temp.append(float(row[1]))
                temp.append(float(row[2]))
                newarray.append(temp)
            count +=1
    return newarray
#Somethign is working here, have to keep messing around remember to add other types of input data, write new function to make sure that all the data stays aligned
#MLP might not be the most optimal thing, sklearn may not deliver on the functionalities I need
def perceptron(newarray, list2):
    X= newarray
    testsize = 0.3
    random_state = 0    
    y = list2
    y = np.asarray(y)
    np.expand_dims(y, -1)
    #X = np.asarray(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    y_train = scaler.fit_transform(y_train.reshape(-1,1)).flatten()
    
    y_test = scaler.transform(y_test.reshape(-1, 1)).flatten()
    ppn = MLPClassifier(hidden_layer_sizes = (3000), activation = "logistic",max_iter = 400,learning_rate_init = 0.01, solver = "lbfgs")
    ppn.fit(X_train, y_train.astype(int))
    y_pred = ppn.predict(X_test)
    newarr = []
    #print(ppn.score(X_test,y_test))
    print(ppn.predict([[60.8, 7.48]]))
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
def webScrape(startyear, endyear, cityname):
    starting = startyear.split('/')
    ending = endyear.split('/')
    dates = mdates.num2date(mdates.drange(DT.datetime(int(starting[0]), int(starting[1]), int(starting[2])),
                                          DT.datetime(int(ending[0]), int(ending[1]), int(ending[2])),
                                      DT.timedelta(days=1)))
    newdates = []
    tempdates = []
    for row in dates:
        newarr = []
        newarr.append(row.year)
        newarr.append(row.month)
        newarr.append(row.day)
        tempdates.append(newarr)
    for row in tempdates:
        temp = []
        tempyear = row[0]
        tempmonth = row[1]
        tempday = row[2]
        if tempmonth < 10:
            tempmonth = "0"+str(tempmonth)
        if tempday < 10:
            tempday = "0" + str(tempday)
            datestr = str(tempyear) + "-" + str(tempmonth) + "-" + str(tempday)
        else:
            datestr = str(tempyear) + "-" + str(tempmonth) + "-" + str(tempday)
        urlstr = 'https://www.almanac.com/weather/history/OR/' + cityname + '/' + datestr 
        print("Iter")
        print(datestr)
        page_html = urllib.request.urlopen(urlstr).read()
        temp.append(datestr)
        try:
            temp.append(float(findTemp(str(page_html))))
            temp.append(float(windSpeed(str(page_html))))
        except:
            temp.append(float(findTemp(str(page_html))))
        newdates.append(temp)
        
    return newdates
def findTemp(page_html):
    temp_marker = page_html.find("Maximum")
    temp_marker = page_html.find("value", temp_marker+1)
    new_temp_marker  = page_html.find("<", temp_marker+3)
    mean_temp = page_html[temp_marker:new_temp_marker+3]
    mean_temp = mean_temp.split('>')
    mean_temp = mean_temp[1].split('<')
    return mean_temp[0]
def windSpeed(page_html):
    temp_marker = page_html.find("Speed")
    temp_marker = page_html.find("Maximum", temp_marker+3)
    temp_marker = page_html.find("<span class=\"value\">", temp_marker+3)
    
    if temp_marker == -1:
        print("Huh?")
    new_temp_marker = page_html.find("<",temp_marker+3)
    mean_temp = page_html[temp_marker:new_temp_marker]
    mean_temp = mean_temp.split('>')
    return mean_temp[1]
def filterInt(masterdata, index1, index2,index3,index4,index5,index6,locationnum):
    newarray1 = []
    newarray2 = []
    newarray3 = []
    newarray4 = []
    newarray5 = []
    newarray6 = []
    for row in masterdata:
        if row[index1] != '' and row[index2] != '' and row[index3] != '' and int(row[index4]) == locationnum and row[index5] != '' and row[index6] != '':
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
def processTimes(inputlist,range1, range2):
    newlist = []
    starting = range1.split('/')
    ending = range2.split('/')
    dates = mdates.num2date(mdates.drange(DT.datetime(int(starting[0]), int(starting[1]), int(starting[2])),
                                          DT.datetime(int(ending[0]), int(ending[1]), int(ending[2])),
                                      DT.timedelta(days=1)))
    
    tempdates = []
    for row in dates:
        year = str(row.year)
        month = str(row.month)
        day = str(row.day)
        if row.month < 10:
            month = "0" + month
        if row.day < 10:
            day = "0" + day
        newstr = year + "/" + month + "/" + day
        tempdates.append(newstr)
    for row in inputlist:
        temp = row.split(" ")
        processed = temp[0].split('+')
        finaldate = processed[0]
        if finaldate in tempdates:
            finaldate = finaldate.replace("/", "-")
            newlist.append(finaldate)
 
    return newlist
#Cleanup happened here
#If not used in next 2 days delete
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

