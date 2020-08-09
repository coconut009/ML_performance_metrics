# ML_performance_metrics
## This is the classifier comparison 
### By comparing three different algorithms' accuracy and computation time
All the source data file is download from __[UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php "UCI ML Respository Home Page")__ <br><br>
The first data set: __[Car Evaluation Data Set](https://archive.ics.uci.edu/ml/datasets/Car+Evaluation)__
<br><br> 
The data Attribute Information:<br>
__Class Values:__<br>
unacc, acc, good, vgood<br>
__Attributes:__<br>
buying: vhigh, high, med, low.<br>
maint: vhigh, high, med, low.<br>
doors: 2, 3, 4, 5more.<br>
persons: 2, 4, more.<br>
lug_boot: small, med, big.<br>
safety: low, med, high.<br>
 
#### __Adaboost__
The adaboost code in this repository is only good to classify two classes.
<br>
The data set downloaded from UCI ML repository has been modified to fit the adaboost classifier 
<br>
The target label has been changed from 4 classes to 2 classes:
<br>
unacc => -1
<br>
acc, good , v_good => 1
<br>
If the data sets have more than two classes(labels) the accuracy will be significant reduced
<br><br> 
#### __Support Vector Machine__
The Support vector machine code in this repository is implemented from sklearn directly for best performance
