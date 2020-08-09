import numpy as np
from time import time
import pandas as pd
from sklearn.model_selection import  KFold
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn. ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC


## file location is subject to change
# data_set_1    =  pd.read_csv(r'/home/aaron/Desktop/car_2class.data',header=0)
# data_set_2    =   pd.DataFrame.to_numpy(pd.read_csv(r'/home/aaron/Desktop/heart_failure_clinical_records_dataset.csv',delimiter=',',header=0))

## File location for Jin
data_set_1 = pd.read_csv(r'car_2class.data',header=0)
data_set_2 = pd.DataFrame.to_numpy(pd.read_csv(r'heart_failure_clinical_records_dataset.csv',delimiter=',',header=0))

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

print("====Data Set 1====")
# cross validation using k fold from sklearn
# for i in range (2,7):
for i in range (2,7):
    test_index = int(label_1.shape[0]/i)
    train_index = label_1.shape[0] - test_index
    train_t_avg = 0
    total_t_avg = 0
    print("\nData Set 1 with the number of spilt is", i, "( Training:",train_index,"Testing:",test_index,")")

    x_1_train, x_1_test = attributes_1[0:train_index], attributes_1[train_index:label_1.shape[0]]
    y_1_train, y_1_test = label_1[0:train_index], label_1[train_index:label_1.shape[0]]

    dist = np.zeros(x_1_train.shape[0])
    errors = np.zeros(x_1_test.shape[0])

    # print("number of trains:", x_1_train.shape[0])
    # print("number of tests:", x_1_test.shape[0])

    y_1_train = (y_1_train + 1) / 2
    y_1_test = (y_1_test + 1) / 2

    for k in (3,5,7):
        t1 = time()
        for l in range(test_index):
            dist = np.sum((x_1_train - x_1_test[l,:])**2, axis=1)**0.5
            sortIndex = np.argsort(dist)
            bestLabels = y_1_train[sortIndex[0:k]]
            prediction = (sum(bestLabels) > k/2.0)*1
            # print("Prediction vs Actual:", prediction,"vs",y_1_test[l])
            errors[l] = (y_1_test[l] != prediction)
        t2 = time()

        TotalErrors = np.sum(errors)
        Accuracy = (1-(TotalErrors) / test_index )*100
        print("\nIf k =", k, "==> Total Errors:", TotalErrors, "(Accuracy: %.2f%%), (Computation time: %.0f ms)" % (Accuracy,((t2-t1)*1000)))
        # print("The total computation time on data set 1 is %.0f ms\n" % ((t2-t1)*1000))
    print("\n")




#############################################################################################
# second data set
label_2     =   data_set_2[:,-1]
attributes_2=   data_set_2[:,:-1]






# kNN Classifier with Python
def classify0(inX, dataSet, labels, k, option):
	# number of rows in the dataSet. Each attribute is a column of the dataSet matrix
  dataSetSize = dataSet.shape[0]

	# the diffMat is then squared and Euclidean distances matrix is calculated by summing the squares of the differences over all attributes and taking the square root
  if(option == "Euclidean"):
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5

  elif(option == "Nominal"):
    newInX = np.tile(inX, (dataSetSize, 1))
    diffMat = np.zeros((dataSet.shape[0], dataSet.shape[1]))
    for x in range(dataSet.shape[0]):
      for y in range(dataSet.shape[1]):
        if(newInX[x,y] != dataSet[x,y]):
          diffMat[x,y] = 1
    distances = diffMat.sum(axis=1)/dataSet.shape[1]

	# Sort and index
  sortedDistIndicies = distances.argsort()
  classCount = {-1:0, 1:0} # This is a dictionary. We can create an entry as classCount[‘M’] = 1. M is key
  for i in range(k):
    # Keep track of the class label for each of kNN
    voteIlabel = labels[sortedDistIndicies[i]]

    # increment the item in the dictionary referenced by the key
    classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

	# Find the item in the dictionary with the most votes
  sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
  return sortedClassCount[0][0]
