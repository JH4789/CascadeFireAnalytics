#TF + Keras
import csv
import statistics
import numpy as np
from array import array
import matplotlib.pyplot as plt
import sys
import math
from sklearn import preprocessing
from sklearn import datasets, linear_model
from sklearn.neural_network  import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn import datasets
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
    list1, list2,longitude,latitude = filterInt(masterdata, 63,73,23,24)
    newarray = []
    counter = 0
    for row in list1:
        newarray.append([int(row),longitude[counter], latitude[counter]])
        counter+=1
    linearreg(newarray,list2)
    #perceptron(newarray, list2)
#Linear reg seems to be working better, further testing needed
def linearreg(list1, list2):
    regr = linear_model.LinearRegression()
    regr.fit(list1, list2)
    print(regr.predict([[6,-123,45]]))
def scale_data(inputdata):
    newarray = []
    for row in inputdata:
        newarray.append(row*100)
    return newarray
#Somethign is working here, have to keep messing around remember to add other types of input data, write new function to make sure that all the data stays aligned
#MLP might not be the most optimal thing, sklearn may not deliver on the functionalities I need
def perceptron(newarray, list2):
    iris = datasets.load_iris()
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
def printYears(masterdata):
    for index in masterdata:
        print(index[4])
#Returns good data
def filterInt(masterdata, index1, index2,index3,index4):
    newarray1 = []
    newarray2 = []
    newarray3 = []
    newarray4 = []
    count = 0
    for row in masterdata:
        if count != 0 and row[index1] != '' and row[index2] != '' and row[index3] != '' and row[index4] != '' and float(row[index2]) >5 :
            newarray1.append(float(row[index1]))
            newarray2.append(float(row[index2]))
            newarray3.append(float(row[index3]))
            newarray4.append(float(row[index4]))
        count +=1
    return newarray1 , newarray2, newarray3, newarray4
def graphRange(masterdata, minsize, maxsize, inputcolor):
    acres_long = filterSize(masterdata, minsize, maxsize, 23)
    acres_lat = filterSize(masterdata, minsize, maxsize, 24)
    plt.scatter(acres_long, acres_lat, color = inputcolor)    
#Anything below this point is preserved for a once in a blue scenario, will probably not be used
def normalDist(masterdata, count):
    longitude = getLongitude(masterdata)
    latitude = getLatitude(masterdata)
    std_long = statistics.stdev(longitude)
    std_lat = statistics.stdev(latitude)
    new_long = []
    
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
#Does not work, will not be used
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

