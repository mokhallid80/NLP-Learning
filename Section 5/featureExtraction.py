import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.svm import LinearSVC

from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv('../../UPDATED_NLP_COURSE/TextFiles/smsspamcollection.tsv', sep='\t')


print(df.isnull().sum())
# we might as well check for empty strings

X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.33, random_state = 42)

count_vect = CountVectorizer()


# Fit
count_vect.fit(X_train)
# Then, Transform
# X_train_counts = count_vect.transform(X_train)

# Do them together
# step 1
X_train_counts = count_vect.fit_transform(X_train)

# Trying to do it without tfidf


X_test_counts = count_vect.transform(X_test)
# clf = classifier
clf = LinearSVC()

clf.fit(X_train_counts, y_train)

clf.predict(X_test_counts)

print(X_train_counts.shape)

print(X_test_counts.shape)
print("WITHOUT tfidf")
predictions = clf.predict(X_test_counts)


df = pd.DataFrame(confusion_matrix(y_test,predictions), index=['ham','spam'], columns=['ham','spam'])
print(df)
print(classification_report(y_test, predictions))


# Now, transform using tfidf
# step 2
tfidf_transformer = TfidfTransformer()

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

X_test_tfidf = tfidf_transformer.fit_transform(X_test_counts)




# tfidf vectorization does both of the steps together 



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
text_clf = Pipeline([('tfidf',TfidfVectorizer()),('clf',LinearSVC())])

text_clf.fit(X_train, y_train)


predictions = text_clf.predict(X_test)

print("WITH tfidf")
df = pd.DataFrame(confusion_matrix(y_test,predictions), index=['ham','spam'], columns=['ham','spam'])
print(df)
print(classification_report(y_test, predictions))
