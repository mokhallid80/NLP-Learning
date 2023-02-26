import numpy as np
import pandas as pd # to read csv files.. etc

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from sklearn import metrics


df = pd.read_csv('../../UPDATED_NLP_COURSE/TextFiles/smsspamcollection.tsv',sep='\t')

# show first 5 rows
print(df.head())

# df.isnull() checks if any of our data items is missing. True => missing

print(df.isnull().sum())


# we will make a model that predict depending on the length and punct only :)

# X is our feature data
X = df[['length','punct']]
# y is our label
y = df['label']


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 42)

lr_model = LogisticRegression(solver='lbfgs')
lr_model.fit(X_train, y_train)


predictions = lr_model.predict(X_test)


df = pd.DataFrame(metrics.confusion_matrix(y_test,predictions), index=['ham','spam'], columns=['ham','spam'])
print(df)

print(metrics.classification_report(y_test, predictions))



nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)


predictions = nb_model.predict(X_test)


df = pd.DataFrame(metrics.confusion_matrix(y_test,predictions), index=['ham','spam'], columns=['ham','spam'])
print(df)

print(metrics.classification_report(y_test, predictions))



svc_model = SVC(gamma='auto')
svc_model.fit(X_train, y_train)


predictions = svc_model.predict(X_test)


df = pd.DataFrame(metrics.confusion_matrix(y_test,predictions), index=['ham','spam'], columns=['ham','spam'])
print(df)

print(metrics.classification_report(y_test, predictions))


