import csv
import numpy as np
from array import array
import matplotlib.pyplot as plt
import sys
def main():
    maximum = sys.maxsize
    masterdata = []
    count = 0
    with open('ODF_Fire.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        for row in csv_reader:
            count +=1
            if row[23] != '' and row[24] != '':
                newaddition = []
                for column in row:
                    newaddition.append(column)
                masterdata.append(newaddition)
    """
    master_long = getLongitude(masterdata)
    master_lat = getLatitude(masterdata)

    acres_100_long = filterSize(masterdata, 100,200, 23)
    acres_100_lat = filterSize(masterdata, 100 ,200, 24)
    
    plt.scatter(master_long, master_lat, color = "black")
    plt.scatter(acres_100_long, acres_100_lat, color = "red")
    """
    graphRange(masterdata, 0, maximum, "green")
    #graphRange(masterdata, 300, 1000, "red")
    print(count)
    
    plt.show()
    
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
def printYears(masterdata):
    for index in masterdata:
        print(index[4])
def filterSize(masterdata, minsize, maxsize, index):
    newarray = []
    for row in masterdata:
        if row[72] != 'Size_acres' and row[72] != '':
            if minsize < float(row[72]) < maxsize:
                newarray.append(float(row[index]))
    return newarray
def graphRange(masterdata, minsize, maxsize, inputcolor):
    acres_long = filterSize(masterdata, minsize, maxsize, 23)
    acres_lat = filterSize(masterdata, minsize, maxsize, 24)
    plt.scatter(acres_long, acres_lat, color = inputcolor)
main()

