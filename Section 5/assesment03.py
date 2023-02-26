import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


df = pd.read_csv("../../UPDATED_NLP_COURSE/TextFiles/moviereviews2.tsv", sep="\t")


print("Checking for empty values")
print(df.isnull().sum())


# Removing nulls
blanks = []

for i, lb, rv in df.itertuples():
    try:
        if(rv.isspace()):
            blanks.append(i)
    except:
            blanks.append(i)


df.drop(blanks, inplace=True)



# splitting the data
X = df['review']
y = df['label']


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 42)


# pipeline
text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('linearSVC', LinearSVC())])


# fit the model
text_clf.fit(X_train, y_train)

# predict 
predictions = text_clf.predict(X_test)

# checking the accuracy
# checking the results 
df = pd.DataFrame(confusion_matrix(y_test,predictions), index=['Pos','Negative'], columns=['Pos','Negative'])
print(df)
print(classification_report(y_test, predictions))

print(accuracy_score(y_test, predictions))


# print(text_clf.predict(["opened my eyes the whole movie"]))

print(text_clf.predict(["I couldn'\t watch this movie. So bad"]))
print(text_clf.predict(["So good"]))

