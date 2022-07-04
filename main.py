"""
Archived for now use as refernece later
"""
from sklearn.neural_network import MLPClassifier
from collections import Counter
import numpy as np
import csv
import matplotlib.pyplot as plt
from array import array
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
def main():
     longitude = []
     latitude = []
     discoveredtime = []
     ignitiontime = []
     with open('ODF_Fire.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        line_count = 0;
        for row in csv_reader:
            longitude.append(row[23])
            latitude.append(row[24])
            ignitiontime.append(row[54])
            discoveredtime.append(row[55])
     """print(longitude)
     print(latitude)
     print(time)"""
     clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
     #print(discoveredtime)
     discovered_date = []
     discovered_time = []
     discovered_year = []
     #print(type(discoveredtime[0]))
     index = 0
     for test in discoveredtime:
          if index != 0:
              test1 = test.split(' ')
              test2 = test1[0].split('/')
              if test2[0] == '':
                   discovered_year.append(0)
              else:
                   inttest = int(test2[0])
              #discovered_date.append(test1[0])
                   discovered_year.append(inttest)
          index = index + 1
#     print(discovered_year)
 #    print(type(discovered_year[0]))
     yearindex = []
     countindex = []
     index = 0
     twodarray = []
     
     for x in range(1960, 2021):
          d = Counter(discovered_year)
          countindex.append([0, x])
          yearindex.append(d[x])
     print(countindex)
     countindex_train = countindex[:-20]
     countindex_test = countindex[-20:]
     yearindex_train = yearindex[:-20]
     yearindex_test = yearindex[-20:]
     reg = linear_model.BayesianRidge()
     reg.fit(countindex_train, yearindex_train)
     print(reg.predict([[1,2018]]))
     
     regr = linear_model.LinearRegression()
     regr.fit(countindex_train, yearindex_train)
     predicted = regr.predict([[1, 2018]])
     print(predicted)
     if len(countindex_test) == len(yearindex_test):
          print('TRUE')
     """
     #plt.scatter(countindex_test, yearindex_test, color="black")
     #plt.plot(countindex_test, predicted, color="blue", linewidth=3)
     
     plt.xticks(())
     plt.yticks(())

     plt.show()
     """
     clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(4, 4), random_state=100)
     clf.fit(countindex_train, yearindex_train)
     print(clf.predict([[0, 2050]]))
main()
