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

import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve

###############################################################################
# define the accuracy calculation function
def accuracy(true_label, prediction):
    accuracy = np.sum(true_label == prediction) / len(true_label)
    return accuracy

###############################################################################
## file location is subject to change
data_set_1    =  pd.read_csv(r'car_2class.data',header=0)
data_set_2    = np.loadtxt('/home/aaron/Desktop/page-blocks.txt')

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

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots()

for i in range (2,7):
    kf = KFold(n_splits=i,shuffle=True)
    print("Data Set 1 with the number of spilt is", i)
    for train_index, test_index in kf.split(attributes_1):
        x_1_train, x_1_test = attributes_1[train_index], attributes_1[test_index]
        y_1_train, y_1_test = label_1[train_index], label_1[test_index]
        n_components = attributes_1.shape[1]
        pca = PCA(n_components=n_components, svd_solver='randomized',whiten=True).fit(x_1_train)
        x_train_pca = pca.transform(x_1_train)
        X_test_pca = pca.transform(x_1_test)
        svc_clf = SVC(C=1000, class_weight='balanced', gamma=0.05)
        svc_clf = svc_clf.fit(x_train_pca, y_train)
        svc_pred = svc_clf.predict(X_test_pca)
        acc = accuracy(y_test,svc_pred)
        viz = plot_roc_curve(svc_clf,x_1_test,y_1_test,name='ROC fold {}'.format(i),alpha=0.3, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Chance', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)

mean_tpr[-1:] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="Receiver operating characteristic example")
ax.legend(loc="lower right")
plt.show() 
         