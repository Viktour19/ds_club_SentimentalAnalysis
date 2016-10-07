# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 08:07:57 2016

@author: amuhebwa
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

train_data = pd.read_csv('insults/train.csv')
train_data = train_data.drop('Date', axis=1)
test_data = pd.read_csv('insults/test_with_solutions.csv')
test_data = test_data[['Insult','Comment']]
y_train = np.array(train_data.Insult)
comments_train = np.array(train_data.Comment)

assert comments_train.shape[0] == y_train.shape[0]

cv = CountVectorizer()
cv.fit(comments_train)

X_train = cv.transform(comments_train).tocsr()

svm = LinearSVC()
svm.fit(X_train, y_train)

comments_test = np.array(test_data.Comment)
X_test = cv.transform(comments_test)
y_test = np.array(test_data.Insult)

score = svm.score(X_test, y_test)
print(comments_test[8])
predicted = svm.predict(X_test.tocsr()[8])[0];
print("Target: %d, prediction: %d" % (y_test[8], predicted))
