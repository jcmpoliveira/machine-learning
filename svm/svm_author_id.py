#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###




from sklearn.svm import SVC

classif = SVC(kernel="rbf", C=10000)

## smaller data set
#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 


t0 = time()
classif.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"	#### training time: 192.599 s

t0 = time()
prediction = classif.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"	#### prediction time: 19.135 s


accu = classif.score(features_test,labels_test)		#### accuracy: 0.9840
print accu


#print "prediction 10:", prediction[10]
#print "prediction 26:", prediction[26]
#print "prediction 50:", prediction[50]

print sum(prediction)




#########################################################


