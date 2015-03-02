# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# ##Imports

# <codecell>

import os
from collections import Counter
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
from scipy import sparse

import util
import pandas as pd

# <markdowncell>

# ##Map prediction/target values to their corresponding numbers

# <codecell>

classMap = {0: 'Agent',
1: 'AutoRun',
2: 'FraudLoad',
3: 'FraudPack',
4: 'Hupigon',
5: 'Krap',
6: 'Lipler',
7: 'Magania',
8: 'None',
9: 'Poison',
10: 'Swizzor',
11: 'Tdss',
12: 'VB',
13: 'Virut',
14: 'Zbot'}
classMap = dict([[v,k] for k,v in classMap.items()])

# <markdowncell>

# ##Function Definitions for Feature Extraction

# <codecell>

#Converts an xml malware file to a vector of frequencies of each all_section token
def xml2vec(filename, readPath = "train/"):
    tokens = []
    # extract id and true class (if available) from filename
    id_str, clas = filename.split('.')[:2]
    
    # parse file as an xml document
    tree = ET.parse(readPath + filename)
    root = tree.getroot()
    for section in root.iter('all_section'):
        for token in section:
            tokens.append(token.tag)
    return tokens

# <codecell>

def createDesignMatrix(readPath = "train/"):
    '''Returns the design matrix and target values if readPath = "/train", else returns just the design matrix.
    The design matrix has a row for each xml file and the columns are counts of all of the distinct all_section tokens.
    '''
    indices = []
    targets = []   
    all_rows = []
    
    for filename in os.listdir(readPath):
        #print ('Working on file:', filename)
        row = pd.Series(xml2vec(filename)).value_counts()
        all_rows.append(row)
        id_str, clas = filename.split('.')[:2]
        indices.append(id_str)
        if readPath == "test/":
            target.append(classMap[clas])
            
    df = pd.concat(all_rows, axis = 1, join = 'outer')
    df = df.transpose()
    df = df.replace(to_replace=NaN, value = 0)
    df.index = indices
    target = pd.Series(target)
    
    return (df, target) if readPath == "/train" else df

# <codecell>

##REMOVE ALL FEATURES IN TRAINING DATA THAT ARE NOT IN TEST DATA (AND VICE VERSA IF NECESSARY)
for column in testdf.columns:
    if column not in df.columns:
        del testdf[column]
        
for column in df.columns:
    if column not in testdf.columns:
        del df[column]        

# <codecell>

len(testdf.columns)

# <markdowncell>

# ##Creating our classifier, Discrete performed better than real on our test set

# <codecell>

bdt_real = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2),
    n_estimators=600,
    learning_rate=1)

X, t = createDesignMatrix(readPath = "train/")

classifier = bdt_real.fit(X, t)
predicted = classifier.predict(testdf) #predict on test values, not on X
accuracy = sum(pd.Series(predicted) & t) / len(testdf)
print (accuracy)

# <codecell>

bdt_discrete = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2),
    n_estimators=600,
    learning_rate=1.5,
    algorithm="SAMME")

X, t = createDesignMatrix(readPath = "train/")

classif_discrete = bdt_discrete.fit(X, t)
discrete_predicted = classif_discrete.predict(testdf) #predict on test values, not on X
sum(pd.Series(discrete_predicted) & t) / len(testdf)

# <markdowncell>

# ##Trying a normalized matrix as the input

# <codecell>

#normalizedX = (X - X.mean()) / X.std()
normalizedX = X / X.sum()
classif_discrete = bdt_discrete.fit(X, t)
discrete_predicted = classif_discrete.predict(testdf) #predict on test values, not on X
sum(pd.Series(discrete_predicted) & t) / len(testdf)

# <markdowncell>

# ##Generate csv submission file for Kaggle Competition

# <codecell>

filename = "submission3.csv"
pd.DataFrame(discrete_predicted, testdf.index).to_csv(filename, index_label = 'Id', header = ['Prediction'])

# <markdowncell>

# ##Example of using AdaBoost in Sci-Kit Learn

# <codecell>

from sklearn.externals.six.moves import zip

import matplotlib.pyplot as plt

from sklearn.datasets import make_gaussian_quantiles
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


X, y = make_gaussian_quantiles(n_samples=13000, n_features=10,
                               n_classes=3, random_state=1)

n_split = 3000

X_train, X_test = X[:n_split], X[n_split:]
y_train, y_test = y[:n_split], y[n_split:]

bdt_real = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2),
    n_estimators=600,
    learning_rate=1)

bdt_discrete = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2),
    n_estimators=600,
    learning_rate=1.5,
    algorithm="SAMME")

bdt_real.fit(X_train, y_train)
bdt_discrete.fit(X_train, y_train)

real_test_errors = []
discrete_test_errors = []

real_test_predict = 0
for real_test_predict, discrete_train_predict in zip(
        bdt_real.staged_predict(X_test), bdt_discrete.staged_predict(X_test)):
    real_test_errors.append(
        1. - accuracy_score(real_test_predict, y_test))
    discrete_test_errors.append(
        1. - accuracy_score(discrete_train_predict, y_test))

n_trees_discrete = len(bdt_discrete)
n_trees_real = len(bdt_real)

# Boosting might terminate early, but the following arrays are always
# n_estimators long. We crop them to the actual number of trees here:
discrete_estimator_errors = bdt_discrete.estimator_errors_[:n_trees_discrete]
real_estimator_errors = bdt_real.estimator_errors_[:n_trees_real]
discrete_estimator_weights = bdt_discrete.estimator_weights_[:n_trees_discrete]

plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.plot(range(1, n_trees_discrete + 1),
         discrete_test_errors, c='black', label='SAMME')
plt.plot(range(1, n_trees_real + 1),
         real_test_errors, c='black',
         linestyle='dashed', label='SAMME.R')
plt.legend()
plt.ylim(0.18, 0.62)
plt.ylabel('Test Error')
plt.xlabel('Number of Trees')

plt.subplot(132)
plt.plot(range(1, n_trees_discrete + 1), discrete_estimator_errors,
         "b", label='SAMME', alpha=.5)
plt.plot(range(1, n_trees_real + 1), real_estimator_errors,
         "r", label='SAMME.R', alpha=.5)
plt.legend()
plt.ylabel('Error')
plt.xlabel('Number of Trees')
plt.ylim((.2,
         max(real_estimator_errors.max(),
             discrete_estimator_errors.max()) * 1.2))
plt.xlim((-20, len(bdt_discrete) + 20))

plt.subplot(133)
plt.plot(range(1, n_trees_discrete + 1), discrete_estimator_weights,
         "b", label='SAMME')
plt.legend()
plt.ylabel('Weight')
plt.xlabel('Number of Trees')
plt.ylim((0, discrete_estimator_weights.max() * 1.2))
plt.xlim((-20, n_trees_discrete + 20))

# prevent overlapping y-axis labels
plt.subplots_adjust(wspace=0.25)
plt.show()

# <codecell>

lol = pd.DataFrame([[1,2,3], [4, 8, 12]])

# <codecell>

lol / lol.sum()

# <codecell>


