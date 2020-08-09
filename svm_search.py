import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn. ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.svm import SVC
from prettytable import PrettyTable
from sklearn.decomposition import PCA

#data_set_2    =   pd.DataFrame.to_numpy(pd.read_csv(r'/home/aaron/Desktop/heart_failure_clinical_records_dataset.csv',header=0))
#label_2     =   data_set_2[:,-1]
#attributes_2=   data_set_2[:,:-1]
def accuracy(true_label, prediction):
    accuracy = np.sum(true_label == prediction) / len(true_label)
    return accuracy 


#data_set_1    =  pd.read_csv(r'/home/aaron/Desktop/car_2class.data',header=0)
data_set_1    =  pd.read_csv(r'/home/aaron/Desktop/car_2class.data',header=0)

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data_set_1["buying"]))
maint = le.fit_transform(list(data_set_1["maint"]))
door = le.fit_transform(list(data_set_1["door"]))
persons = le.fit_transform(list(data_set_1["persons"]))
lug_boot = le.fit_transform(list(data_set_1["lug_boot"]))
safety = le.fit_transform(list(data_set_1["safety"]))
target_label = list(data_set_1["target_label"])

label_1     =   np.array(target_label)
#label_1     =   np.array(data_set_1["target_label"])
attributes_1=   np.array(list(zip(buying,maint,door,persons,lug_boot,safety)))

X_train, X_test, y_train, y_test = train_test_split(attributes_1, label_1, test_size=0.2, random_state=5)

#adb = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators = 5, learning_rate = 1)
#adb.fit(X_train,y_train)
#prediction = adb.predict(X_test)
results_List=[]
results_List_c=[]
results_List_g=[]
component_list=[]
run_time = 6
c_list = [1, 100, 1000, 10000]
g_list = [0.001, 0.005, 0.01, 0.05]
for element_g in g_list:
    for element_C in c_list:
        for i in range  (run_time):
            # #############################################################################
        # Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
        # dataset): unsupervised feature extraction / dimensionality reduction
            n_components = i+1
            pca = PCA(n_components=n_components, svd_solver='randomized',
                    whiten=True).fit(X_train)

            X_train_pca = pca.transform(X_train)
            X_test_pca = pca.transform(X_test)

            # #############################################################################
            # Train a SVM classification model
            clf = SVC(C=element_C, class_weight='balanced', gamma=element_g)
            clf = clf.fit(X_train_pca, y_train)
            
            # #############################################################################
            # Quantitative evaluation of the model quality on the test set

            y_pred = clf.predict(X_test_pca)

            #print(classification_report(y_test, y_pred, target_names=target_names))
            #print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

            
            acc = accuracy(y_test,y_pred)
            component_list.append(n_components)
            results_List.append(acc)
            results_List_c.append(element_C)
            results_List_g.append(element_g)
x= PrettyTable()
x.field_names = ["Number of Components", "C Value","gamma Value","Value"]
for i in range (run_time*len(g_list)*len(c_list)):
    x.add_row([component_list[i],results_List_c[i],results_List_g[i],results_List[i]])
print(x)

