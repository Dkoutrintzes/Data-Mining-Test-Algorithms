import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from statistics import mean
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix , precision_score

def creating_knn(X, Y, k_range):
    # Separating the X,Y to train and test values
    accuracy = []
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)
    # creating the knn classifier
    for k in range(k_range[0],k_range[1]+1):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, Y_train)

         # Use the X_test values to take some predictions
        Y_pred = knn.predict(X_test)

        # Printing and saving the results of the accuracy of the knn classifier\
        accuracy.append(metrics.accuracy_score(Y_test, Y_pred))

        #print(classification_report(Y_test, Y_pred))
    return accuracy
names = ['Pre', 'Glu', 'Blood', 'Skin', 'Insu', 'BMI', 'DPF', 'Age', 'Outcome']
datasetA = pd.read_csv('diabetes.csv', names = names)
data = pd.DataFrame()
print("------------------------------Question A------------------------------")
# Question A 1-5
# On the preprocessing we will replace all the wrong registries with the mean value of the equal characteristic
for i in range(1,8):
    datasetA[names[i]] = datasetA[names[i]].replace(0,mean(datasetA[names[i]]))
datasetA.to_csv('diabetes_missing_values_replace.csv')
# Separating the registries into X for the characteristics and Y for the classes-outcome

X = datasetA.drop(['Outcome'],axis = 1)
Y = datasetA['Outcome']

# Creating an knn classifier whit k = 5 nearest neighbors and we running for 10 times to see if it's stable
accuracies_with_k_5 = []
for i in range(10):
    accuracies_with_k_5.append(creating_knn(X,Y,[5,5]))
print("")
print("The accuracies of the 10 runs of the knn classifier is: ")
for i in range(10):
    print('Accuracy of',i+1,'Run is {:1.2f}'.format(accuracies_with_k_5[i]))
print("")
print('The maximum error of accuracy between runs is: {:1.3f}'.format((max(accuracies_with_k_5)-min(accuracies_with_k_5))*100),"%")
if (max(accuracies_with_k_5)-min(accuracies_with_k_5))*100 < 10:
    print('This error is acceptable')
else:
    print('This error is unacceptable and the knn os not stable')

df = pd.DataFrame(accuracies_with_k_5).transpose()
data = data.append(df)
print("")
# Question A 6
# Run the algorithm again but this time with 1-15 k nearest neighbors

accuracyA = []
for i in range(1,16):
    accuracyA.append(creating_knn(X, Y, [i,i], ))

print("")
print("The accuracies of the 15 runs of the knn classifier is: ")
for i in range(15):
    print('Accuracy of',i+1,'Run is {:1.2f}'.format(accuracyA[i]))

print("------------------------------Question B------------------------------")
# Question B 1-4
datasetB = pd.read_csv('diabetes.csv', names = names)
# On the preprocessing we will remove all the wrong registries
for i in ['Glu', 'Blood', 'Skin', 'Insu', 'BMI', 'DPF', 'Age']:
    datasetB[i] = datasetB[i].replace(0,np.nan)
datasetB = datasetB.dropna()

# Separating the registries into X for the characteristics and Y for the classes-outcome

X = datasetB.drop(['Outcome'],axis = 1)
Y = datasetB['Outcome']

# Creating an knn classifier whit k = 5 nearest neighbors and we running for 10 times to see if it's stable
accuracies_with_k_5 = []
for i in range(10):
    accuracies_with_k_5.append(creating_knn(X,Y,[5,5]))
print("")
print("The accuracies of the 10 runs of the knn classifier is: ")
for i in range(10):
    print('Accuracy of',i+1,'Run is {:1.2f}'.format(accuracies_with_k_5[i]))
print("")
print('The maximum error of accuracy between runs is: {:1.3f}'.format((max(accuracies_with_k_5)-min(accuracies_with_k_5))*100),"%")
if (max(accuracies_with_k_5)-min(accuracies_with_k_5))*100 < 10:
    print('This error is acceptable')
else:
    print('This error is unacceptable and the knn is not stable')
df = pd.DataFrame(accuracies_with_k_5).transpose()
data = data.append(df)
print("")
# Question B 5
# Run the algorithm again but this time with 1-15 k nearest neighbors

accuracyB = []
for i in range(0,10):
    temp = mean(creating_knn(X, Y, [1,15]))
    print(mean(creating_knn(X, Y, [1,15])))
    accuracyB.append(temp)

print("The accuracies of the 15 runs of the knn classifier is: ")
for i in range(10):
    print('Accuracy of',i+1,'Run is {:1.2f}'.format(mean(accuracyB)))
print("")
print("------------------------------Question C------------------------------")
# Question C
df = pd.DataFrame(accuracyA).transpose()
data = data.append(df)
print("Mean of Accuracy A: {:1.2f}\nThe error between runs is: {:1.2f}".format(mean(accuracyA),(max(accuracyA)-min(accuracyA))*100), "%")
df = pd.DataFrame(accuracyB).transpose()
data = data.append(df)
print("Mean of Accuracy B: {:1.2f}\nThe error between runs is: {:1.2f}".format(mean(accuracyB),(max(accuracyB)-min(accuracyB))*100), "%")

data.to_csv("Result.csv")