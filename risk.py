#Joint Probability Mass Distribution?, this may be promising to actually generate a risk index
#Look into SciPy
#New dataset? Caps downloads at 1000 records a day doing initial testing with tillamook
#Legacy code being phased out, maybe delete some stuff at the bottom
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
from sklearn import preprocessing
from sklearn import datasets, linear_model
from sklearn.neural_network  import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import datasets
import datetime as DT
import matplotlib.dates as mdates
import statistics
from scipy.stats import multivariate_normal
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.tree import export_graphviz
from subprocess import check_call
import pandas as pd
#import pydot
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
    masterdata = pd.read_csv('ODF_Fire.csv')
    testmaster = pd.read_csv('tillamook_total.csv')
    #All ignition time entries after 2008 go boom using the below line of code
    #flamelength, firesize,longitude,latitude,ignition,serial = filterInt(masterdata, 63,73,23,24,54,2)
    masterdata =process_master(masterdata,52)
    
    testmaster = testmaster.dropna(subset = ["humidity"])
    
    newWrapper(masterdata, '2015/5/1','2019/5/1', 51, 'Tillamook', testmaster)
    #writeFile(masterdata, '2012/2/10', '2020/9/1', 51, 'Tillamook', 'tillamook.csv')
    #bigWrapper(masterdata, '2015/5/1','2019/5/1', 51, 'Tillamook','tillamook.csv')
    #print(returnDates('2018/5/1','2020/5/1'))
    stop = timeit.default_timer()
    print('Time: ', stop - start)  
def writeFile(masterdata, start, end, citynum, cityname, filename):
    filler,filler,filler,unit,ignition, serial,  = filterInt(masterdata,0,0,0,8,54,2,citynum)
    ignition = processTimes(ignition, start,end)
    scraped = webScrape(start, end ,cityname)
    with open( filename, 'w') as filewrite:
        writer = csv.writer(filewrite)
        writer.writerow(['Date','Temperature' ,'Wind Speed'])
        for row in scraped:
            data = []
            data.append(row[0])
            data.append(row[1])
            data.append(row[2])
            writer.writerow(data)
def newWrapper(masterdata, start, end, citynum, cityname, environmentdata):
    ignition = masterdata[['Ign_DateTime']]
    dates = returnDates(start, end)
    
    ignition[['Ign_DateTime', 'Filler']] = ignition['Ign_DateTime'].str.split(' ',1,expand=True)
    del ignition['Filler']
    ignition['Ign_DateTime'] = ignition['Ign_DateTime'].str.replace('/','-')
    
    ignition = ignition[ignition['Ign_DateTime'].isin(dates)]
    
    #print(ignition)
    
    loaded = loadData('astoria.csv',start,end)
    
    finalcount = []
    environment = []
    counter = 0

    count = 0
    print(environmentdata)
    environmentdata = environmentdata[environmentdata.datetime.isin(dates)]
    environmentdata = environmentdata[["humidity","windspeed","tempmax"]]
    print(environmentdata)
    #scraped = webScrape(start,end, cityname)
    #countarray = countFires(ignition, loaded)
    randomForest(environmentdata, countFires(ignition,dates))
def bigWrapper(masterdata, start , end , citynum, cityname, filename):
    #filler,filler,filler,unit,ignition, serial,  = filterInt(masterdata,0,0,0,8,54,2,citynum)
    ignition = masterdata[['Ign_DateTime']]
    dates = returnDates(start, end)
    ignition[['Ign_DateTime', 'Filler']] = ignition['Ign_DateTime'].str.split(' ',1,expand=True)
    del ignition['Filler']
    ignition['Ign_DateTime'] = ignition['Ign_DateTime'].str.replace('/','-')
    print(ignition)
    ignition = ignition[ignition['Ign_DateTime'].isin(dates)]
    loaded = loadData(filename,start,end)
    print(ignition)
    finalcount = []
    environment = []
    counter = 0
    count = 0
    #scraped = webScrape(start,end, cityname)
    countarray = countFires(ignition, dates)
    processedarray = []
    for row in loaded:
        if row[2] < 500:
            temp = []
            temp.append(row[1])
            temp.append(row[2])
            environment.append(temp)
        count +=1
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
    randomForest(environment, countFires(ignition,dates))
def countFires(ignition, scraped):
    newarray = []
    for row in scraped:
        temp = []
        
        count = len(ignition[ignition['Ign_DateTime'] == row])
        #print(count)
        #print(count)
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
    y = np.asarray(finalcount)
    truecount = 0
    falsecount = 0
    for i in y:
        if i == True:
            truecount +=1
        else:
            falsecount +=1
    print(truecount)
    print(falsecount)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    #scaler = StandardScaler()
    #scaler.fit(X_train)
    #X_train = scaler.transform(X_train)
    #X_test = scaler.transform(X_test)
    clf = RandomForestClassifier(max_depth = 75,min_samples_split = 100,n_estimators = 100,class_weight = {0:1,1:40})
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cv_results = cross_validate(clf, X, y, cv=3)
    print(cv_results)
    print(clf.decision_path(X_test))
    #clf.plot_tree(clf, features_names = X, class_names = y,filled = True)
    estimator = clf.estimators_[1]
    """
    export_graphviz(estimator, out_file='tree.dot', 
                feature_names = ['Humidity', 'Temperature','Wind Speed'],
                class_names = 'Fire Prediction',
                rounded = True, proportion = False, 
                precision = 2, filled = True)
    """
    #check_call(['dot','-Tpng','tree.dot',' -o',' test.png'])
    count = 0
    print(y_pred)
    falsecount = 0
    truecount = 0
    truetotal = 0

    fig = plt.figure(figsize = plt.figaspect(0.5))
    ax1 = fig.add_subplot(1,2,1,projection = "3d")
    
    maxtemp = [item[0] for item in X_test]
    windspeed = [item[1] for item in X_test]
    humidity = [item[2] for item in X_test]
    count = 0
    for i in maxtemp:
        if y_test[count] == True:
            ax1.scatter(i,windspeed[count],maxtemp[count],c="red")
            truetotal +=1
        else:
            ax1.scatter(i,windspeed[count],maxtemp[count],c="green")
        count +=1
    ax1 = fig.add_subplot(1,2,2,projection = "3d")
    count = 0
    for i in maxtemp:
        if y_pred[count] == True:
            ax1.scatter(i,windspeed[count],maxtemp[count],c="red")
            truetotal +=1
        else:
            ax1.scatter(i,windspeed[count],maxtemp[count],c="green")
        count +=1
    """
    for i in X_test:
        if y_test[count] == True:
            plt.scatter(i[0], i[1], color = "blue")
            truetotal +=1
        if y_pred[count] == True:
            #print("Predicted")
            falsecount +=1
            plt.scatter(i[0],i[1], color = "green")
        if y_test[count] == True and y_pred[count] == True:
            #print("Hoepfully happned once")
            truecount +=1
            #print(X_test[count])
        count +=1
    """
    print(len(X_test))
    print(falsecount)
    print(truecount)
    print(truetotal)
    """
    for i in X_train:
        if y_train[count] == False:
            #plt.scatter(i[0],i[1],color = "red")
            plt.scatter(i[0],i[1],color = "blue")
        count +=1
    count = 0
    for i in X_train:
        if y_train[count] == True:
            plt.scatter(i[0],i[1],color = "red")
            
        count +=1
        """
    plt.show()
def calculateRisk(range1, range2, environment, countdata, datapoint):
    
    for i in bigdata:
        if datapoint[0] - range1 <= i[0] <= datapoint[0] + range1 and datapoint[1]-range2 <= i[1] <= datapoint[1]+range2:
            
            if countdata[count] == True:
                total +=1
        count +=1
#Linear reg might need to be retired soon
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
def returnDates(start,end):
    starting = start.split('/')
    ending = end.split('/')
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
        newdates.append(datestr)
    return newdates
def loadData(filename,start,end):
    newarray = []
    count = 0
    starting = start.split('/')
    ending = end.split('/')
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
        newdates.append(datestr)
        #Filler for now 
        
#    print(tempdates)
    newdata = []
    print(newdates)
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')

        for row in csv_reader:
            
            if count != 0 and len(row) != 0:
                temp = []
                temp.append(row[0])
                temp.append(float(row[1]))
                temp.append(float(row[2]))
                newdata.append(temp)
            count +=1
    for row in newdata:
        #print(row[0])
        if row[0] in newdates:
            
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
        except:
            temp.append(-1)
        try:
            temp.append(float(windSpeed(str(page_html))))
        except:
            temp.append(-1)
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
def process_master(masterdata,locationnum):
    masterdata = masterdata.dropna(subset = ["Ign_DateTime","Current_District", "Current_Unit"])
    print(masterdata["Current_District"])
    masterdata = masterdata[masterdata.Current_District == locationnum]
    return masterdata
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

main()

