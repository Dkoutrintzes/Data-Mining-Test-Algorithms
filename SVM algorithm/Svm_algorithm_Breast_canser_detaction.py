import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from statistics import mean
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix , precision_score


print("Μάθημα: Εργαστήριο Εξόρυξη γνώσης από δεδομένα\nΟνομα: Κουτριντζες Δημητρης\nΑΜ: 3998\nΤμημα: 12-2\nMail: dkoutrintzes@gmail.com")

names = ['ID','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']
dataset = pd.read_csv('breast-cancer-wisconsin.csv', names= names)
#deleting "?"
dataset = dataset.replace('?',np.nan)
dataset = dataset.dropna()
#Check for values out of range base on the panel beneath
"""
#  Attribute                     Domain
   -- -----------------------------------------
   1. Sample code number            id number
   2. Clump Thickness               1 - 10
   3. Uniformity of Cell Size       1 - 10
   4. Uniformity of Cell Shape      1 - 10
   5. Marginal Adhesion             1 - 10
   6. Single Epithelial Cell Size   1 - 10
   7. Bare Nuclei                   1 - 10
   8. Bland Chromatin               1 - 10
   9. Normal Nucleoli               1 - 10
  10. Mitoses                       1 - 10
  11. Class:                        (2 for benign, 4 for malignant)
"""
out_of_range_values = False
χ = dataset.index
print(χ)
for i in dataset.index:
    for j in range(1,len(dataset.columns)-1):
        if int(dataset[names[j]][i]) < 1 or int(dataset[names[j]][i]) >10:
            dataset = dataset.drop(i)
            out_of_range_values = True
            break
        if int(dataset['Class'][i]) !=2 and int(dataset['Class'][i]) !=4:
            dataset = dataset.drop(i)
            out_of_range_values = True
if out_of_range_values is False:
    print("There are no wrong registries")
elif out_of_range_values is True:
    print("The wrong registries have been deleted. The now size of the dataset is ",len(dataset))

#Spit the dataset to X,Y to train and test the SVM

X = dataset.drop(['ID','Class'],axis = 1)
Y = dataset['Class']
data = pd.DataFrame()
#Creating the traing and test X,Y
for i in range(10):
    X_train,X_test,Y_train,Y_test = train_test_split(X , Y , test_size = 0.20)

#Creation of the linear separation



    svcclassifier = SVC(kernel='linear')
    svcclassifier.fit(X_train , Y_train)
    print('svc',svcclassifier)

#trying the SVM wiht the test subjects that we rezerved for testing

    Y_preb = svcclassifier.predict(X_test)
    print(confusion_matrix(Y_test , Y_preb))
    a = classification_report(Y_test,Y_preb,output_dict=True)
    print(classification_report(Y_test,Y_preb))
    df = pd.DataFrame(a).transpose()
    data = data.append(df)
data.to_csv('Final_result_2.csv')






