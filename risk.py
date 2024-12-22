#Project that builds random forest classifier out of environmental and fire occurence data
#Use cases are very specific, exception management is currently WIP
#Inputs for the classifier are based upon the MAXIMUM temperature and wind speed
import warnings
warnings.simplefilter("ignore")

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
from sklearn.metrics import classification_report
import datetime as DT
import matplotlib.dates as mdates
import statistics
from scipy.stats import multivariate_normal
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.tree import export_graphviz
from subprocess import check_call
import pandas as pd
import numpy.ma as ma
from sklearn.metrics import confusion_matrix
import seaborn as sb
#Main function
def main():
    start = timeit.default_timer()
    np.random.seed(0)
    maximum = sys.maxsize
    masterdata = []
    count = 0
    masterfilename = input("Enter the name of the master file: ")
    #Has to be taken from Visual Crossing or an analog
    environmentaldataname = input('Enter the name of the environment data file: ')
    areacode = input("Enter the code recognized by the master file: ")
    startingrange = input("Enter the starting date range (Formatted YYYY/MM/DD): ")
    endrange = input("Enter the end date range (Formatted YYYY/MM/DD): ")
    masterdata = pd.read_csv("ODF_Fire.csv")
    environmentaldata = pd.read_csv("tillamook_total.csv")
    masterdata =process_master(masterdata,int(51))
    environmentaldata = environmentaldata.dropna(subset = ["humidity"])
    wrapper(masterdata, "2016/10/23","2019/10/23", int(51), environmentaldata)
    stop = timeit.default_timer()
    print('Time: ', stop - start)
#Functions to convert units (Visual Crossing Dataset is not in metric)
def convert(x):
    return (x-32)/1.8
def convertmiles(x):
    return x*1.6093435
#Wrapper function that is a flexible solve
def wrapper(masterdata, start, end, citynum, environmentdata):
    ignition = masterdata[['Ign_DateTime']]
    dates = returnDates(start, end)
    test = ignition['Ign_DateTime'].str.split()
    
    ignition['Ign_DateTime'] = [x[0] for x in test]
    
    ignition['Ign_DateTime'] = ignition['Ign_DateTime'].str.replace('/','-')
    ignition = ignition[ignition['Ign_DateTime'].isin(dates)]
    finalcount = []
    environment = []
    counter = 0
    count = 0    
    environmentdata = environmentdata[environmentdata.datetime.isin(dates)]
    environmentdata = environmentdata[["tempmax","windspeed","humidity"]]
    environmentdata["tempmax"] = environmentdata["tempmax"].apply(convert)
    environmentdata["windspeed"] = environmentdata["windspeed"].apply(convertmiles)
    randomForest(environmentdata, countFires(ignition,dates))
#Function to create the binary dataset from the ODF dataset
def countFires(ignition, scraped):
    newarray = []
    for row in scraped:
        temp = []
        
        count = len(ignition[ignition['Ign_DateTime'] == row])
        if count >= 1:
            count = True
        else:
            count = False
        temp.append(count)
        newarray.append(temp)
    return newarray
#Function to fit the classifier
def randomForest(environment, finalcount):
    X = np.asarray(environment)

    y = np.asarray(finalcount)
    fires = []
    nofires = []
    firestotal = 0
    nofirestotal = 0
    notemp = 0
    averagetemp = 0
    finalarr = np.concatenate((X,y), axis=1)
    for i in finalarr:
        if i[3] == 1:
            averagetemp += i[2]
            firestotal +=1
        else:
            nofires.append(i)
            notemp += i[2]
            nofirestotal +=1
    print(averagetemp/firestotal)
    print(notemp/nofirestotal)
    #y = y.ravel()
    #testmaster = np.concatenate((X,y),axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    #scaler = StandardScaler()
    #scaler.fit(X_train)
    #X_train = scaler.transform(X_train)
    #X_test = scaler.transform(X_test)
    clf = RandomForestClassifier(max_depth = 75,min_samples_split = 50,n_estimators = 100,class_weight = "balanced")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cv_results = cross_validate(clf, X, y, cv=3)
    #clf.plot_tree(clf, features_names = X, class_names = y,filled = True)
    estimator = clf.estimators_[1]
    print(classification_report(y_pred, y_test))
    #Used for generating decision path
    """
    export_graphviz(estimator, out_file='tree.dot', 
                feature_names = ['Maximum Temperature', 'Windspeed','Humidity'],
                class_names = 'Fire Prediction',
                rounded = True, proportion = False, 
                precision = 2, filled = True)
    
    count = 0

    falsecount = 0
    truecount = 0
    truetotal = 0
    cm = confusion_matrix(y_test, y_pred)
    #Used for generating the confusion matrix and graphing
    
    ax = sb.heatmap(cm, annot = True,fmt = 'g',cmap='Blues')
    ax.set_title("   Random Forest Predictions on Wildfire Occurence Confusion Matrix\n\n") 
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values ')
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])
    newfig = ax.get_figure()
    newfig.savefig("Confusion_Matrix.png")
    
    fig = plt.figure(figsize = plt.figaspect(0.5))
    ax1 = fig.add_subplot(1,2,1,projection = "3d")
    ax1.title.set_text("Actual Fires")
    ax1.set_xlabel("Temperature (Celsius)")
    ax1.set_ylabel("Wind Speed (KPH)")
    ax1.set_zlabel("Humidity (relative)")
    maxtemp = [item[0] for item in X_test]
   
    windspeed = [item[1] for item in X_test]
    humidity = [item[2] for item in X_test]
    count = 0
    humiditytotal = []
    maxtemptotal = []
    windspeedtotal = []
    
    for i in maxtemp:
        if y_test[count] == True:
            humiditytotal.append(humidity[count])
            maxtemptotal.append(maxtemp[count])
            windspeedtotal.append(windspeed[count])
            ax1.scatter(i,windspeed[count],humidity[count],c="red")
            truetotal +=1
        else:
            ax1.scatter(i,windspeed[count],humidity[count],c="green")
        count +=1
    ax1 = fig.add_subplot(1,2,2,projection = "3d")
    ax1.title.set_text("Predicted Fires")
    ax1.set_xlabel("Temperature (Celsius)")
    ax1.set_ylabel("Wind Speed (KPH)")
    ax1.set_zlabel("Humidity (relative)")
    
   
    truetotal = 0
    count = 0
    for i in maxtemp:
        if y_pred[count] == True:
            ax1.scatter(i,windspeed[count],humidity[count],c="red")

            truetotal +=1
        else:
            ax1.scatter(i,windspeed[count],humidity[count],c="green")
        count +=1
    
    #fig.savefig("3dtest.png")
    plt.show()
    """
    
    inputLoop(clf)
#Generates dates to track with the main datasets
def inputLoop(clf):
    print(clf.predict([[29.44,28.96,18]]))
    userinput = input("Enter P to enter a prediction, enter Q to quit?\n")
    while userinput != 'P' and userinput != "Q":
        print(userinput)
        userinput = input("Please try again. Enter P to enter a prediction, enter Q to quit?\n")
    if userinput == "Q":
        return
    else:
        environ = input("Please enter the environmental data as decimals separated by spaces. Ex: Temperature WindSpeed Humidity?\n")
        split = environ.split(' ')
        #A problem with the current implementation of the model is that extremeley high values in
        #any of the categories results in a return of false
        split = [float(x) for x in split]
        print(split)
        
        try:
            print(clf.predict([split]))
            inputLoop(clf)
        except:
            print("Please try again")
            inputLoop(clf)
        
    
#Generates dates
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
#Filters the master data set
def process_master(masterdata,locationnum):
    masterdata = masterdata.dropna(subset = ["Ign_DateTime","Current_District", "Current_Unit"])
   
    masterdata = masterdata[masterdata.Current_District == locationnum]
    return masterdata
main()

