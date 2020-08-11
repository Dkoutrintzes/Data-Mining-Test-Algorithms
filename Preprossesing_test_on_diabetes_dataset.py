import pandas as pd
import numpy as np
from sklearn import preprocessing
def del_values(dataset):
    values = []
    for i in range(len(dataset)):
        if dataset.iloc[i,1:8].count()!=7:
            values.append(i)
    return values
def count_zero(dataset):
    counter = []

    for i in range(len(dataset.columns)):
            counter.append(len(dataset) - dataset.iloc[:,i].count())

    return counter
print("Μάθημα: Εργαστήριο Εξόρυξη γνώσης από δεδομένα\nΟνομα: Κουτριντζες Δημητρης\nΑΜ: 3998\nΤμημα: 12-2\nMail: dkoutrintzes@gmail.com")
print("\n")
dataset = pd.read_csv('diabetes.csv', names = ['Pre', 'Glu', 'Blood', 'Skin', 'Insu', 'BMI', 'DPF', 'Age', 'Outcome'])

for column in ['Glu', 'Blood', 'Skin', 'Insu', 'BMI', 'DPF', 'Age']:
    dataset[column] = dataset[column].replace(0,np.nan)

print("---------------------------------------Question 1---------------------------------------\n\n")

new_dataset = dataset.dropna(axis = 0)

remove_dataset = del_values(dataset)
print("The New Dataset after the deleting of the wrong registries is:\n",new_dataset)
print("\n")

print("The initial size of the dataset was {} registries and the new dataset has size of {} registries, {} registries was deleted.".format(len(dataset),len(new_dataset),len(dataset)-len(new_dataset)))
print("\n")

print("The registries that was deleted is: ",remove_dataset)

print("---------------------------------------Question 2---------------------------------------\n\n")

zeros = count_zero(dataset)
for i in range(1,len(dataset.columns)-1):
    print(dataset.columns[i], 'Column has',zeros[i],'zeros')

new_dataset_2 = dataset.dropna(axis = 1,thresh = len(dataset) - 40)
new_dataset_2 = new_dataset_2.fillna(0)


print("\n")
print(new_dataset_2)

new_dataset_2.to_csv('final_dataset.csv')
positive = 0
negative = 0

for i in range(len(new_dataset_2)):
    if new_dataset_2['Outcome'][i] == 0:
        negative += 1
    else:
        positive += 1
print("There are ",positive," people Positeve in diavetes and ",negative,"people that they dont have diabetes.")