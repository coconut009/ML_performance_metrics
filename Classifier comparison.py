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

###############################################################################
# define the accuracy calculation function
def accuracy(true_label, prediction):
    accuracy = np.sum(true_label == prediction) / len(true_label)
    return accuracy

# define the True Positive Counter function
def TP_cnt(true_label, prediction):
    TP_cnt = np.sum(true_label+prediction==2)
    return TP_cnt

# define the False Positive Counter function (if Falce == -1)
def FP_cnt_minusOne(true_label, prediction):
    FP_cnt = np.sum(prediction-true_label==2)
    return FP_cnt

# define the False Positive Counter function (if False == 0)
def FP_cnt_zero(true_label, prediction):
    FP_cnt = np.sum(prediction-true_label==1)
    return FP_cnt

    
###############################################################################
# define the KNN algorithm call:
k=[3,5,7,9]
def knn(x_train,y_train,x_test,y_test,k):
    acc_list=[]
    for element in k:
        knn = KNeighborsClassifier(n_neighbors=element)
        knn_pred = knn.fit(x_train,y_train)
        acc=knn.score(x_test,y_test)
        acc_list.append(acc)
    return  acc_list

# #############################################################################
# Define the Adaboost classifier call fot the data sets need to be tested
# return the acc value with training and testing time

Ada_clf =   AdaBoostClassifier(DecisionTreeClassifier(),n_estimators = 15, learning_rate = 1)
def ada_boost(x_train,y_train,x_test,y_test):
    t0 = time()
    #Ada_clf = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators = 5, learning_rate = 1)
    Ada_clf.fit(x_train, y_train)
    t1 = time()
    Ada_prediction = Ada_clf.predict(x_test)
    acc = accuracy(y_test, Ada_prediction)
    t2 = time()
    trainT  =   t1-t0
    runT    =   t2-t0
    return acc, trainT, runT

# #############################################################################
# Define the SVM classifier call for the data sets
# return the acc value with training and testing time
# Compute a PCA (eigenvalues) on the dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction

def svm_call (x_train,y_train,x_test,y_test,n_components):
    t0 = time()
    pca = PCA(n_components=n_components, svd_solver='randomized',whiten=True).fit(x_train)
    x_train_pca = pca.transform(x_train)
    X_test_pca = pca.transform(x_test)
    svc_clf = SVC(C=1000, class_weight='balanced', gamma=0.05)
    svc_clf = svc_clf.fit(x_train_pca, y_train)
    t1 = time()
    svc_pred = svc_clf.predict(X_test_pca)
    acc = accuracy(y_test,svc_pred)
    t2 = time()
    trainT  =   t1-t0
    runT    =   t2-t0
    return acc, trainT, runT

###############################################################################
k=[3,5,7,9]
def knn(x_train,y_train,x_test,y_test,k):
    acc_list=[]
    for element in k:
        knn = KNeighborsClassifier(n_neighbors=element)
        knn_pred = knn.fit(x_train,y_train)
        acc=knn.score(x_test,y_test)
        acc_list.append(acc)
    return  acc_list

###############################################################################
## file location is subject to change
data_set_1    =  pd.read_csv(r'car_2class.data',header=0)
data_set_2    = np.loadtxt('/home/aaron/Desktop/page-blocks.txt')

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
#define the value of K for KNN algorithm


print("====Data Set 1====")
# cross validation using k fold from sklearn
for i in range (2,7):
    kf = KFold(n_splits=i,shuffle=True)
    counter =0
    ada_train_t_avg = 0
    ada_total_t_avg = 0
    svc_train_t_avg = 0
    svc_total_t_avg = 0
    knn_total_t_avg = 0
    print("Data Set 1 with the number of spilt is", i)
    for train_index, test_index in kf.split(attributes_1):
        t0 = time()
        x_1_train, x_1_test = attributes_1[train_index], attributes_1[test_index]
        y_1_train, y_1_test = label_1[train_index], label_1[test_index]
        t1 = time()
        counter=counter+1
        ada_boost_1 = ada_boost(x_1_train,y_1_train,x_1_test,y_1_test)
        n_components = attributes_1.shape[1]
        svm_1 = svm_call(x_1_train,y_1_train,x_1_test,y_1_test,n_components)
        t2 = time()
        knn_1 = knn(x_1_train,y_1_train,x_1_test,y_1_test,k)
        t3 = time()
        print ("At %d number of fold, the accuracy of adaboost classifier on data set 1: %.2f%%" %(counter,ada_boost_1[0]*100))
        print ("At %d number of fold, the accuracy of SVM classifier on data set 1: %.2f%%" %(counter,svm_1[0]*100))
        print("At %d number of fold, on data set 1 when the k value is %d, it has the hightest accuracy value %.2f%%" %(counter,k[knn_1.index(max(knn_1))], 100*max(knn_1)))
        ada_train_t_avg = ada_train_t_avg + ada_boost_1[1]
        ada_total_t_avg = ada_total_t_avg + ada_boost_1[2]+t1-t0
        svc_train_t_avg = svc_train_t_avg + svm_1[1]
        svc_total_t_avg = svc_total_t_avg + svm_1[2] +t1-t0
        knn_total_t_avg = svc_total_t_avg + t3-t2 +t1-t0
    ada_train_t_avg = ada_train_t_avg / counter
    ada_total_t_avg = ada_total_t_avg / counter
    svc_train_t_avg = svc_train_t_avg / counter
    svc_total_t_avg = svc_total_t_avg / counter
    knn_total_t_avg = knn_total_t_avg / counter
    print("\nThe Adaboost training time on data set 1 is %f ms" % (ada_train_t_avg*1000))
    print("The total computation time of Adaboost classifier on data set 1 is %f ms" % (ada_total_t_avg*1000))
    print("The SVM training time on data set 1 is %f ms" % (svc_train_t_avg*1000))
    print("The total computation time of SVM classifier on data set 1 is %f ms" % (svc_total_t_avg*1000))
    print("The KNN classifier total computation time on data set 2 is %f ms" % (knn_total_t_avg*1000))


print("====Data Set 2====")
# cross validation using k fold from sklearn
for i in range (2,7):
    kf = KFold(n_splits=i,shuffle= True)
    counter =0
    ada_train_t_avg = 0
    ada_total_t_avg = 0
    svc_train_t_avg = 0
    svc_total_t_avg = 0
    knn_total_t_avg = 0
    print("Data Set 2 with the number of spilt is", i)
    for train_index, test_index in kf.split(attributes_2):
        t0 = time()
        x_2_train, x_2_test = attributes_2[train_index], attributes_2[test_index]
        y_2_train, y_2_test = label_2[train_index], label_2[test_index]
        t1 = time()
        counter=counter+1
        ada_boost_2 = ada_boost(x_2_train,y_2_train,x_2_test,y_2_test)
# #############################################################################
# Compute a PCA (eigenvalues) on the dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
        n_components = attributes_2.shape[1]
        svm_2 = svm_call(x_2_train,y_2_train,x_2_test,y_2_test,n_components)
        t2 = time()
        knn_2 = knn(x_2_train,y_2_train,x_2_test,y_2_test,k)
        t3 = time()
        print ("At %d number of fold, the accuracy of adaboost classifier on data set 2: %.2f%%" %(counter,ada_boost_2[0]*100))
        print ("At %d number of fold, the accuracy of SVM classifier on data set 2: %.2f%%" %(counter,svm_2[0]*100))
        print ("At %d number of fold, on data set 2 when the k value is %d, it has the hightest accuracy value %.2f%%" %(counter,k[knn_2.index(max(knn_2))], 100*max(knn_2)))
        ada_train_t_avg = ada_train_t_avg + ada_boost_2[1]+t1-t0
        ada_total_t_avg = ada_total_t_avg + ada_boost_2[2]+t1-t0
        svc_train_t_avg = svc_train_t_avg + svm_2[1]
        svc_total_t_avg = svc_total_t_avg + svm_2[2] +t1-t0
        knn_total_t_avg = svc_total_t_avg + t3-t2 +t1-t0
    ada_train_t_avg = ada_train_t_avg / counter
    ada_total_t_avg = ada_total_t_avg / counter
    svc_train_t_avg = svc_train_t_avg / counter
    svc_total_t_avg = svc_total_t_avg / counter
    knn_total_t_avg = knn_total_t_avg / counter
    print("\nThe Adaboost training time on data set 1 is %f ms" % (ada_train_t_avg*1000))
    print("The total computation time on data set 1 is %f ms" % (ada_total_t_avg*1000))
    print("The SVM training time on data set 2 is %f ms" % (svc_train_t_avg*1000))
    print("The total computation time on data set 2 is %f ms" % (svc_total_t_avg*1000))
    print("The KNN classifier total computation time on data set 2 is %f ms" % (knn_total_t_avg*1000))
