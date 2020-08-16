import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import  KFold
from sklearn.neighbors import KNeighborsClassifier



import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve



###############################################################################
## file location is subject to change
data_set    =  pd.read_csv(r'car_2class.data',header=0)


le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data_set["buying"]))
maint = le.fit_transform(list(data_set["maint"]))
door = le.fit_transform(list(data_set["door"]))
persons = le.fit_transform(list(data_set["persons"]))
lug_boot = le.fit_transform(list(data_set["lug_boot"]))
safety = le.fit_transform(list(data_set["safety"]))
target_label = list(data_set["target_label"])
# zip all the transformed data into numpy array
label     =   np.array(target_label)
attributes=   np.array(list(zip(buying,maint,door,persons,lug_boot,safety)))


tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots()

# change the k_value to get the different plot
k_value = 9 

for i in range (2,7):
    kf = KFold(n_splits=i,shuffle=True)
    for train_index, test_index in kf.split(attributes):
        x_train, x_test = attributes[train_index], attributes[test_index]
        y_train, y_test = label[train_index], label[test_index] 
        knn = KNeighborsClassifier(n_neighbors=k_value)
        knn = knn.fit(x_train,y_train)
        viz = plot_roc_curve(knn,x_test,y_test,name='ROC @ fold {} times'.format(i),alpha=0.8, lw=1, ax=ax)
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
        label=r'Mean ROC (AUC = %0.4f $\pm$ %0.4f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="KNN Classifier (when k = %d) ROC curve of Data Set 1" %(k_value))
ax.legend(loc="lower right")
plt.show()  