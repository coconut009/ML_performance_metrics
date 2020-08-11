import numpy as np
from time import time
import pandas as pd
from sklearn.model_selection import  KFold
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn. ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


## file location is subject to change
data_set_1    =  pd.read_csv(r'car_2class.data',header=0)
#data_set_2    =   pd.read_csv(r'/home/aaron/Desktop/page-blocks.data',header=None)
data_set_2    = np.loadtxt('/home/aaron/Desktop/page-blocks.data')
#data_set_2    =   pd.read_csv(r'/home/aaron/Desktop/page-blocks.csv',header=0)
label_2     =   data_set_2[:,-1]
attributes_2=   data_set_2[:,:-1]
#print(data_set_2)
#print(attributes_2)
#print(label_2)

#print(data_set_2.head())
## File location for Jin
#data_set_1 = pd.read_csv(r'car_2class.data',header=0)
#data_set_2 = pd.DataFrame.to_numpy(pd.read_csv(r'heart_failure_clinical_records_dataset.csv',delimiter=',',header=0))
def accuracy(true_label, prediction):
    accuracy = np.sum(true_label == prediction) / len(true_label)
    return accuracy

def knn(x_train,y_train,x_test,y_test,k):
    acc_list=[]
    for element in k:
        knn = KNeighborsClassifier(n_neighbors=element)
        knn_pred = knn.fit(x_train,y_train)
        acc=knn.score(x_test,y_test)
        acc_list.append(acc)
    return  acc_list



    

  

k=[3,5,7,9]

#############################################################################################
# first data set header info (reference)
# buying,maint,door,persons,lug_boot,safety,target_label
# convert all the non-int value to numeric value using sklearn preprocessing labelencoder
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data_set_1["buying"]))
maint = le.fit_transform(list(data_set_1["maint"]))
door = le.fit_transform(list(data_set_1["door"]))
persons = le.fit_transform(list(data_set_1["persons"]))
lug_boot = le.fit_transform(list(data_set_1["lug_boot"]))
safety = le.fit_transform(list(data_set_1["safety"]))
target_label = list(data_set_1["target_label"])

label_1     =   np.array(target_label)
attributes_1=   np.array(list(zip(buying,maint,door,persons,lug_boot,safety)))
label_2     =   data_set_2[:,-1]
attributes_2=   data_set_2[:,:-1]
#print("====Data Set 1====")
# cross validation using k fold from sklearn
for i in range (2,7):
    kf = KFold(n_splits=i,shuffle= True)
    counter =0 
    train_t_avg = 0
    total_t_avg = 0
    for train_index, test_index in kf.split(attributes_2):
      counter=counter+1
      #print("\nData Set 1 with the number of spilt is", i, "( Training:",train_index,"Testing:",test_index,")")
      x_1_train, x_1_test = attributes_2[train_index], attributes_2[test_index]
      y_1_train, y_1_test = label_2[train_index], label_2[test_index]
      #print( x_1_test.shape)
      knn_1 = knn(x_1_train,y_1_train,x_1_test,y_1_test,k)
      print("At %d number of fold,When the k value is %d, it has the hightest accuracy value %f" %(counter,k[knn_1.index(max(knn_1))], max(knn_1)))
      #print(y_1_test.shape[1])
      #print("correct:", len(y_1_test)*knn_1)
      #print("error:", len(y_1_test)*(1-knn_1))

 