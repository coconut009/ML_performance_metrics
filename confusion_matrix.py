import numpy as np
from time import time
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import  KFold
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn. ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


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
    tn, fp, fn, tp  = confusion_matrix(y_test, Ada_prediction).ravel()
    t2 = time()
    trainT  =   t1-t0
    testT    =   t2-t1
    return tn, fp, fn, tp , trainT, testT

# #############################################################################
# Define the svm classifier call for the data sets
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
    tn, fp, fn, tp  = confusion_matrix(y_test,svc_pred).ravel()
    t2 = time()
    trainT  =   t1-t0
    testT    =   t2-t1
    return tn, fp, fn, tp , trainT, testT

###############################################################################
k=[3,5,7,9]
def knn(x_train,y_train,x_test,y_test,k):
    tf_list=[]
    for element in k:
        t0=time()
        task=[]
        knn = KNeighborsClassifier(n_neighbors=element)
        knn_pred = knn.fit(x_train,y_train)
        predit = knn.predict(x_test)
        tn, fp, fn, tp = confusion_matrix(y_test, predit).ravel()
        t1=time()
        print("KNN k=%d: \t %d \t %d \t %d \t %d \t %.2f%% \t Total: %.2fms" %(element,tn, fp, fn, tp, ((tn+tp)/(tn+fp+fn+tp)*100), (t1-t0)*1000 ))
        # print("KNN: k=%d: \t tn:%d fp:%d fn:%d tp:%d \t Accuracy: %.2f%% \t Total time:%.2fms" %(element,tn, fp, fn, tp, ((tn+tp)/(tn+fp+fn+tp)*100), (t1-t0)*1000 ))


#def knn(x_train,y_train,x_test,y_test,k):
#    acc_list=[]
#    for element in k:
#        knn = KNeighborsClassifier(n_neighbors=element)
#        knn = knn.fit(x_train,y_train)
#        knn_pred = knn.predict(x_test)
#        tn, fp, fn, tp  = confusion_matrix(y_test,knn_pred).ravel()
#    return  acc_list
###############################################################################
## file location is subject to change
data_set_1    =  pd.read_csv(r'car_2class.data',header=0)
# data_set_2    = np.loadtxt('/home/aaron/Desktop/page-blocks.txt') # For Aaron
data_set_2    = np.loadtxt('page-blocks.txt') # For Jin

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


print("===========Data Set 1===========")
# cross validation using k fold from sklearn
for i in range (2,7):
    kf = KFold(n_splits=i,shuffle=True)
    counter =0
    # ada_train_t_avg = 0
    # ada_total_t_avg = 0
    # svc_train_t_avg = 0
    # svc_total_t_avg = 0
    # knn_total_t_avg = 0
    print("\n\n\n=====Data Set 1 with the number of spilt is %d======" %i)
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
        print ("\n===Split: %d, Fold: %d ===" % (i, counter))
        print ("[Algorithm] \t [TN] \t [FP] \t [FN] \t [TP] \t [Accuracy] \t -----[Time]-----")
        print ("Adaboost \t %d \t %d \t %d \t %d \t %.2f%% \t Training: %.2fms \t Testing: %.2fms" %(ada_boost_1[0],ada_boost_1[1],ada_boost_1[2],ada_boost_1[3],((ada_boost_1[0]+ada_boost_1[2])/(ada_boost_1[0]+ada_boost_1[1]+ada_boost_1[2]+ada_boost_1[3])*100), (ada_boost_1[4]*1000),(ada_boost_1[5]*1000)))
        print ("SVM \t \t %d \t %d \t %d \t %d \t %.2f%% \t Training: %.2fms \t Testing: %.2fms" %(svm_1[0],svm_1[1],svm_1[2],svm_1[3],((svm_1[0]+svm_1[3])/(svm_1[0]+svm_1[1]+svm_1[2]+svm_1[3])*100), (svm_1[4]*1000),(svm_1[5]*1000)))
        # print ("Adaboost: \t tn:%d fp:%d fn:%d tp:%d \t Accuracy: %.2f%% \t Training time: %.2fms \t Testing time: %.2fms" %(ada_boost_1[0],ada_boost_1[1],ada_boost_1[2],ada_boost_1[3],((ada_boost_1[0]+ada_boost_1[2])/(ada_boost_1[0]+ada_boost_1[1]+ada_boost_1[2]+ada_boost_1[3])*100), (ada_boost_1[4]*1000),(ada_boost_1[5]*1000)))
        # print ("SVM: \t\t tn:%d fp:%d fn:%d tp:%d \t Accuracy: %.2f%% \t Training time: %.2fms \t Testing time: %.2fms" %(svm_1[0],svm_1[1],svm_1[2],svm_1[3],((svm_1[0]+svm_1[3])/(svm_1[0]+svm_1[1]+svm_1[2]+svm_1[3])*100), (svm_1[4]*1000),(svm_1[5]*1000)))
        # print ("KNN:")
        t3 = time()
        knn_1 = knn(x_1_train,y_1_train,x_1_test,y_1_test,k)
        t4 = time()
        # print("\nThe KNN total time is %.2f ms\n\n" % (t4-t3))
        # ada_train_t_avg = ada_train_t_avg + ada_boost_1[4]
        # ada_total_t_avg = ada_total_t_avg + ada_boost_1[5]+t1-t0
        # svc_train_t_avg = svc_train_t_avg + svm_1[4]
        # svc_total_t_avg = svc_total_t_avg + svm_1[5] +t1-t0
        # knn_total_t_avg = knn_total_t_avg + t4-t3 +t1-t0
    # ada_train_t_avg = ada_train_t_avg / counter
    # ada_total_t_avg = ada_total_t_avg / counter
    # svc_train_t_avg = svc_train_t_avg / counter
    # svc_total_t_avg = svc_total_t_avg / counter
    # knn_total_t_avg = knn_total_t_avg / counter
    # print("\nThe Adaboost training time on data set 1 is %.2f ms" % (ada_train_t_avg*1000))
    # print("The total computation time of Adaboost classifier on data set 1 is %.2f ms" % (ada_total_t_avg*1000))
    # print("The svm training time on data set 1 is %.2f ms" % (svc_train_t_avg*1000))
    # print("The total computation time of svm classifier on data set 1 is %.2f ms" % (svc_total_t_avg*1000))
    # print("The KNN classifier total computation time on data set 1 is %.2f ms" % (knn_total_t_avg*1000))


print("\n\n\n\n===========Data Set 2===========")
# cross validation using k fold from sklearn
for i in range (2,7):
    kf = KFold(n_splits=i,shuffle= True)
    counter =0
    ada_train_t_avg = 0
    ada_total_t_avg = 0
    svc_train_t_avg = 0
    svc_total_t_avg = 0
    knn_total_t_avg = 0
    print("\n\n\n=====Data Set 2 with the number of spilt is %d======" %i)
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
        print ("\n===Split: %d, Fold: %d ===" % (i, counter))
        print ("[Algorithm] \t [TN] \t [FP] \t [FN] \t [TP] \t [Accuracy] \t -----[Time]-----")
        print ("Adaboost \t %d \t %d \t %d \t %d \t %.2f%% \t Training: %.2fms \t Testing: %.2fms" %(ada_boost_2[0],ada_boost_2[1],ada_boost_2[2],ada_boost_2[3],((ada_boost_2[0]+ada_boost_2[2])/(ada_boost_2[0]+ada_boost_2[1]+ada_boost_2[2]+ada_boost_2[3])*100), (ada_boost_2[4]*1000),(ada_boost_2[5]*1000)))
        print ("SVM \t \t %d \t %d \t %d \t %d \t %.2f%% \t Training: %.2fms \t Testing: %.2fms" %(svm_2[0],svm_2[1],svm_2[2],svm_2[3],((svm_2[0]+svm_2[3])/(svm_2[0]+svm_2[1]+svm_2[2]+svm_2[3])*100), (svm_2[4]*1000),(svm_2[5]*1000)))
        t3 = time()
        knn_2 = knn(x_2_train,y_2_train,x_2_test,y_2_test,k)
        t4 = time()
        # print("\nThe Adaboost training time is %.2f ms" % (ada_boost_2[4]*1000))
        # print("The Adaboost testing time is %.2f ms" % (ada_boost_2[5]*1000))
        # print("\nThe svm training time is %.2f ms" % (svm_2[4]*1000))
        # print("The svm testing time is %.2f ms\n\n" % (svm_2[5]*1000))
        # print("\nThe KNN total time is %.2f ms\n\n" % (t4-t3))
    #     ada_train_t_avg = ada_train_t_avg + ada_boost_2[4]+t1-t0
    #     ada_total_t_avg = ada_total_t_avg + ada_boost_2[5]+t1-t0
    #     svc_train_t_avg = svc_train_t_avg + svm_2[4]
    #     svc_total_t_avg = svc_total_t_avg + svm_2[5] +t1-t0
    #     knn_total_t_avg = knn_total_t_avg + t4-t3 +t1-t0
    # ada_train_t_avg = ada_train_t_avg / counter
    # ada_total_t_avg = ada_total_t_avg / counter
    # svc_train_t_avg = svc_train_t_avg / counter
    # svc_total_t_avg = svc_total_t_avg / counter
    # knn_total_t_avg = knn_total_t_avg / counter
    # print("\nThe Adaboost training time on data set 2 is %.2f ms" % (ada_train_t_avg*1000))
    # print("The total computation time on data set 2 is %.2f ms" % (ada_total_t_avg*1000))
    # print("The svm training time on data set 2 is %.2f ms" % (svc_train_t_avg*1000))
    # print("The total computation time on data set 2 is %.2f ms" % (svc_total_t_avg*1000))
    # print("The KNN classifier total computation time on data set 2 is %.2f ms" % (knn_total_t_avg*1000))
