import numpy as np 
from sklearn import preprocessing
from sklearn.model_selection import  KFold
from sklearn.decomposition import PCA
from sklearn.svm import SVC


import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve



###############################################################################
## file location is subject to change
data_set    =  np.loadtxt('/home/aaron/Desktop/page-blocks.txt')

label     =   data_set[:,-1]
attributes=   data_set[:,:-1]


tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots()
random_state = np.random.RandomState(0)
svc_clf = SVC(kernel='linear', probability=True,random_state=random_state)

for i in range (2,7):
    kf = KFold(n_splits=i,shuffle=True)
    for train_index, test_index in kf.split(attributes):
        x_train, x_test = attributes[train_index], attributes[test_index]
        y_train, y_test = label[train_index], label[test_index] 
        svc_clf = svc_clf.fit(x_train, y_train) 
        viz = plot_roc_curve(svc_clf,x_test,y_test,name='ROC @ fold {} times'.format(i),alpha=0.8, lw=1, ax=ax)
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
       title="SVM Classifier (Without PCA) ROC curve of Data Set 2")
ax.legend(loc="lower right")
plt.show() 
         