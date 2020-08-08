import numpy as np
from time import time
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn import preprocessing
from adaboost import Adaboost

# define the accuracy calculation function
def accuracy(true_label, prediction):
    accuracy = np.sum(true_label == prediction) / len(true_label)
    return accuracy 
    
## file location is subject to change
data_set    =  pd.read_csv(r'/home/aaron/Desktop/car_2class.data',header=0)
#data_set    =  pd.read_csv(r'/home/aaron/Desktop/car.data',header=0)
#print (data_set.head())

# header info (reference)
# buying,maint,door,persons,lug_boot,safety,target_label

# convert all the non-int value to numeric value using sklearn preprocessing labelencoder
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data_set["buying"]))
maint = le.fit_transform(list(data_set["maint"]))
door = le.fit_transform(list(data_set["door"]))
persons = le.fit_transform(list(data_set["persons"]))
lug_boot = le.fit_transform(list(data_set["lug_boot"]))
safety = le.fit_transform(list(data_set["safety"]))
target_label = le.fit_transform(list(data_set["target_label"]))

label      =   np.array(target_label)
attributes =   np.array(list(zip(buying,maint,door,persons,lug_boot,safety)))

for i in range (2,7):
    
# cross validation using k fold from sklearn
    kf = KFold(n_splits=i,shuffle=True)
    counter =0 
    print("When the number of spilt is", i)
    for train_index, test_index in kf.split(attributes): 
#print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = attributes[train_index], attributes[test_index]
        y_train, y_test = label[train_index], label[test_index]
#print("train set size: %d,test set size: %d" %(len(x_train),len(x_test)))
        t0 = time()
        counter=counter+1
        Ada_clf = Adaboost(n_clf=5)
        Ada_clf.fit(x_train, y_train)  
        t1 = time()
        Ada_prediction = Ada_clf.predict(x_test)
         
        
        Ada_acc = accuracy(y_test, Ada_prediction)
            
        print ("At %d number of fold with the Accuracy: %f" %(counter,Ada_acc))
    print("The training time is %f ms" % ((t1 - t0)*1000))
            