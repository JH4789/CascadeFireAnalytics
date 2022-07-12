import csv
import statistics
import numpy as np
from array import array
import matplotlib.pyplot as plt
import sys
import math
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
    
    graphRange(masterdata, 0, maximum, "green")
    #graphRange(masterdata, 300, 1000, "red")
    print(count)
    """
    list1, list2 = filterData(masterdata, 4,72)
    testing = linearReg(list1, list2)
    for x in range(1,2000):
        testing.train(x)
    print (testing.adjuster)
    plt.scatter(list1, list2, color = "black")
    for x in range(1960, 2022):
        plt.scatter(x, x*testing.adjuster)
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
#Returns good data
def filterData(masterdata, index1, index2):
    newarray1 = []
    newarray2 = []
    count = 0
    for row in masterdata:
        if count != 0 and row[index1] != '' and row[index2] != '':
            newarray1.append(float(row[index1]))
            newarray2.append(float(row[index2]))
        count +=1
    return newarray1 , newarray2
def graphRange(masterdata, minsize, maxsize, inputcolor):
    acres_long = filterSize(masterdata, minsize, maxsize, 23)
    acres_lat = filterSize(masterdata, minsize, maxsize, 24)
    plt.scatter(acres_long, acres_lat, color = inputcolor)
def testMLP(masterdata, inputcolor):
    longitude = getLongitude(masterdata)
    latitude = getLatitude(masterdata)
    test1 = []
    test2 = []
    count = 0
    for data in longitude:
        temp = []
        temp.append(data)
        temp.append(latitude[count])
        temp +=1
        test1.append(temp)
        test2.append(count)
    
#Function to do STD and calculate normal distribution for data
#Probably more proof of concept than anything else
def normalDist(masterdata, count):
    longitude = getLongitude(masterdata)
    latitude = getLatitude(masterdata)
    std_long = statistics.stdev(longitude)
    std_lat = statistics.stdev(latitude)
    new_long = []
    
    #for x <
class linearReg:
    def __init__(self, x, y):
        self.input = x
        self.labels = y
        self.adjuster = 1
    def train(self, index):
        newoutput = self.input[index] * self.adjuster
        if newoutput > self.labels[index]:
            self.adjuster -= 0.01
        else:
            self.adjuster += 0.01
    
class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],4) 
        self.weights2   = np.random.rand(4,1)                 
        self.y          = y
        self.output     = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2
    def sigmoid_derivative(value):
         return (math.exp(float(value))/((1+math.exp(float(value)))**2))  
main()

