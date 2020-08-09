import numpy as np
from time import time
import pandas as pd
from sklearn.model_selection import  KFold
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn. ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# define the accuracy calculation function
def accuracy(true_label, prediction):
    accuracy = np.sum(true_label == prediction) / len(true_label)
    return accuracy 
    
## file location is subject to change
data_set_1    =  pd.read_csv(r'/home/aaron/Desktop/car_2class.data',header=0)
data_set_2    =   pd.DataFrame.to_numpy(pd.read_csv(r'/home/aaron/Desktop/heart_failure_clinical_records_dataset.csv',delimiter=',',header=0))
 
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

# define the classifiers for data set 1
Ada_1_clf = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators = 5, learning_rate = 1)

#############################################################################################
# second data set 
label_2     =   data_set_2[:,-1]
attributes_2=   data_set_2[:,:-1]
# define the classifiers for data set 2
Ada_2_clf = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators = 15, learning_rate = 1)

# cross validation using k fold from sklearn
for i in range (2,7):
    kf = KFold(n_splits=i,shuffle=True)
    counter =0 
    print("Data Set 1 with the number of spilt is", i)
    for train_index, test_index in kf.split(attributes_1): 
        t0 = time()
        x_1_train, x_1_test = attributes_1[train_index], attributes_1[test_index]
        y_1_train, y_1_test = label_1[train_index], label_1[test_index]        
        t1 = time()     
        Ada_1_clf.fit(x_1_train, y_1_train)  
        t2 = time()
        Ada_1_prediction = Ada_1_clf.predict(x_1_test)       
        counter=counter+1
        Ada_1_acc = accuracy(y_1_test, Ada_1_prediction)
        t3 = time()       
# #############################################################################
# Compute a PCA (eigenvalues) on the dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
        n_components = attributes_1.shape[1] 
        t4 = time()
        pca_1 = PCA(n_components=n_components, svd_solver='randomized',whiten=True).fit(x_1_train)
        x_1_train_pca = pca_1.transform(x_1_train)
        x_1_test_pca = pca_1.transform(x_1_test) 
        t5 = time()
        svc_clf = SVC(C=1000, class_weight='balanced', gamma=0.01)
        svc_clf = svc_clf.fit(x_1_train_pca, y_1_train)
        print ("At %d number of fold, the accuracy of adaboost classifier on data set 1: %f" %(counter,Ada_1_acc))
    print("The Adaboost training time on data set 1 is %f ms" % ((t2 - t1)*1000))
    print("The total computation time on data set 1 is %f ms" % ((t3 - t0)*1000))

# cross validation using k fold from sklearn
for i in range (2,7):
    kf = KFold(n_splits=i,shuffle=True)
    counter =0 
    print("Data Set 2 with the number of spilt is", i)
    for train_index, test_index in kf.split(attributes_2): 
        t0 = time()
        x_2_train, x_2_test = attributes_2[train_index], attributes_2[test_index]
        y_2_train, y_2_test = label_2[train_index], label_2[test_index]        
        t1 = time()       
        Ada_2_clf.fit(x_2_train, y_2_train)  
        t2 = time()
        Ada_2_prediction = Ada_2_clf.predict(x_2_test)        
        counter=counter+1
        Ada_2_acc = accuracy(y_2_test, Ada_2_prediction)
        t3 = time()        
        print ("At %d number of fold, the accuracy of adaboost classifier on data set 2: %f" %(counter,Ada_2_acc))
    print("The Adaboost training time on data set 2 is %f ms" % ((t2 - t1)*1000))
    print("The total computation time on data set 2 is %f ms" % ((t3 - t0)*1000))           