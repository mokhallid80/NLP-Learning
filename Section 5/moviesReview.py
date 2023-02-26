import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

df = pd.read_csv("../../UPDATED_NLP_COURSE/TextFiles/moviereviews.tsv", sep="\t")

# print(df['review'][0])

print(df.isnull().sum())
# some reviews are missing
# we can delete these missing reviews
#df.dropna(inplace=True)

blanks = []

#(index, label, review)
for i,lb, rv in df.itertuples():

    try:
        if rv.isspace():
            blanks.append(i)
    except:
            blanks.append(i)


df.drop(blanks, inplace=True)

print(df.shape)



X = df['review']
y = df['label']


# splitting the data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 42)


# building the pipeline
text_clf = Pipeline([('tfidf',TfidfVectorizer()), ('Classifier', LinearSVC())])

# fit the pipeline
text_clf.fit(X_train, y_train)

# prediction
predictions = text_clf.predict(X_test)


# checking the results 
df = pd.DataFrame(confusion_matrix(y_test,predictions), index=['Pos','Negative'], columns=['Pos','Negative'])
print(df)
print(classification_report(y_test, predictions))

print(accuracy_score(y_test, predictions))


# print(text_clf.predict(["opened my eyes the whole movie"]))

print(text_clf.predict(["I couldn'\t watch this movie. So bad"]))
