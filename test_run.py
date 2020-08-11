import numpy as np
from time import time
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import  KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn. ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# define the accuracy calculation function
def accuracy(true_label, prediction):
    accuracy = np.sum(true_label == prediction) / len(true_label)
    return accuracy
# define the KNN algorithm call:
def knn(x_train,y_train,x_test,y_test,k):
  for element in k:
    knn = KNeighborsClassifier(n_neighbors=element)
    knn_pred = knn.fit(x_train,y_train)
    #knn_acc=accuracy(y_test,knn_pred)
    acc=knn.score(x_test,y_test)

  return acc

## file location is subject to change
data_set_1    =  pd.read_csv(r'/home/aaron/Desktop/car_2class.data',header=0)
data_set_2    =   pd.DataFrame.to_numpy(pd.read_csv(r'/home/aaron/Desktop/heart_failure_clinical_records_dataset.csv',delimiter=',',header=0))

## File location for Jin
#data_set_1 = pd.read_csv(r'car_2class.data',header=0)
#data_set_2 = pd.DataFrame.to_numpy(pd.read_csv(r'heart_failure_clinical_records_dataset.csv',delimiter=',',header=0))

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
# zip all the transformed data into numpy array
label_1     =   np.array(target_label)
attributes_1=   np.array(list(zip(buying,maint,door,persons,lug_boot,safety)))

# second data set
label_2     =   data_set_2[:,-1]
attributes_2=   data_set_2[:,:-1]

#############################################################################################

# define the classifiers for data set 1
Ada_1_clf = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators = 5, learning_rate = 1)

# define the classifiers for data set 2
Ada_2_clf = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators = 15, learning_rate = 1)

#define the value of K for KNN algorithm
k=[3,5,7,9]

print("====Data Set 1====")
# cross validation using k fold from sklearn
for i in range (2,7):
    kf = KFold(n_splits=i,shuffle=True)
    counter =0
    ada_train_t_avg = 0
    ada_total_t_avg = 0
    svc_train_t_avg = 0
    svc_total_t_avg = 0   
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
        X_1_test_pca = pca_1.transform(x_1_test)        
        svc_1_clf = SVC(C=100, class_weight='balanced', gamma=0.05)
        svc_1_clf = svc_1_clf.fit(x_1_train_pca, y_1_train)
        t5 = time()
        svc_1_pred = svc_1_clf.predict(X_1_test_pca)
        svc_1_acc = accuracy(y_1_test,svc_1_pred)
        t6 = time()
        print ("At %d number of fold, the accuracy of adaboost classifier on data set 1: %.2f%%" %(counter,Ada_1_acc*100))
        print ("At %d number of fold, the accuracy of SVM classifier on data set 1: %.2f%%" %(counter,svc_1_acc*100))
        ada_train_t_avg = ada_train_t_avg + t2-t1
        ada_total_t_avg = ada_total_t_avg + t3-t0
        svc_train_t_avg = svc_train_t_avg + t5-t4
        svc_total_t_avg = svc_total_t_avg + t6-t4 +t1-t0
    ada_train_t_avg = ada_train_t_avg / counter
    ada_total_t_avg = ada_total_t_avg / counter
    svc_train_t_avg = svc_train_t_avg / counter
    svc_total_t_avg = svc_total_t_avg / counter
    print("\nThe Adaboost training time on data set 1 is %f ms" % (ada_train_t_avg*1000))
    print("The total computation time on data set 1 is %f ms" % (ada_total_t_avg*1000))
    print("The SVM training time on data set 1 is %f ms" % (svc_train_t_avg*1000))
    print("The total computation time on data set 1 is %f ms" % (svc_total_t_avg*1000))

print("====Data Set 2====")
# cross validation using k fold from sklearn
for i in range (2,7):
    kf = KFold(n_splits=i,shuffle=True)
    counter =0
    ada_train_t_avg = 0
    ada_total_t_avg = 0
    svc_train_t_avg = 0
    svc_total_t_avg = 0  
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
# #############################################################################
# Compute a PCA (eigenvalues) on the dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
        n_components = attributes_2.shape[1]
        t4 = time()
        pca_2 = PCA(n_components=n_components, svd_solver='randomized',whiten=True).fit(x_2_train)
        x_2_train_pca = pca_2.transform(x_2_train) 
        X_2_test_pca = pca_2.transform(x_2_test)
        svc_2_clf = SVC(C=1000, class_weight='balanced', gamma=0.01)
        svc_2_clf = svc_2_clf.fit(x_2_train_pca, y_2_train)
        t5 = time()
        svc_2_pred = svc_2_clf.predict(X_2_test_pca)
        svc_2_acc = accuracy(y_2_test,svc_2_pred)
        t6 = time()
        print ("At %d number of fold, the accuracy of adaboost classifier on data set 2: %.2f%%" %(counter,Ada_2_acc*100))
        print ("At %d number of fold, the accuracy of SVM classifier on data set 2: %.2f%%" %(counter,svc_2_acc*100))
        ada_train_t_avg = ada_train_t_avg + t2-t1
        ada_total_t_avg = ada_total_t_avg + t3-t0
        svc_train_t_avg = svc_train_t_avg + t5-t4
        svc_total_t_avg = svc_total_t_avg + t6-t4 +t1-t0
    ada_train_t_avg = ada_train_t_avg / counter
    ada_total_t_avg = ada_total_t_avg / counter
    svc_train_t_avg = svc_train_t_avg / counter
    svc_total_t_avg = svc_total_t_avg / counter
    print("\nThe Adaboost training time on data set 1 is %f ms" % (ada_train_t_avg*1000))
    print("The total computation time on data set 1 is %f ms" % (ada_total_t_avg*1000))
    print("The SVM training time on data set 1 is %f ms" % (svc_train_t_avg*1000))
    print("The total computation time on data set 1 is %f ms" % (svc_total_t_avg*1000))
