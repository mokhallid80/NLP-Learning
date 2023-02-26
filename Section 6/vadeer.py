import nltk
import pandas as pd

nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# returns a dict of +, -, 0 or compound score by normalizing the -, + and 0 scores
sid = SentimentIntensityAnalyzer()

def analyze(text):
    print(text)
    print(sid.polarity_scores(text))
    print('\n')


analyze("This is a great movie")
analyze("This was the best. most awesome movie ever made!!!")
analyze("This was the WORST movie that has ever disgraced the screen")

analyze("This is a GOOD movie!!!!!")
analyze("This is a good movie")


df = pd.read_csv('../../UPDATED_NLP_COURSE/TextFiles/amazonreviews.tsv', sep='\t')


df.dropna(inplace=True)

# blanks = []
# for i, lb, rv in df.itertuples():
#     if(type(rv) == str):
#         if rv.isspace():
#             blanks.append(i)
# df.drop(blanks, inplace=True)

# df.iloc array of values read from the CSV file

analyze(df.iloc[0]['review'])


df['scores'] = df['review'].apply(lambda review: sid.polarity_scores(review))

df['compound'] = df['scores'].apply(lambda d: d['compound'])

df['comp_score'] = df['compound'].apply(lambda score:'pos' if score >= 0 else 'neg')



print(accuracy_score(df['label'], df['comp_score']))
print(classification_report(df['label'], df['comp_score']))