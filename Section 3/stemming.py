import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer


p_stemmer = PorterStemmer()

words = ['run','runs', 'runner', 'ran', 'running', 'easily', 'fairly', 'fairness']

words2 = ['generous','generation','generously','generate']

print("Stemming using PorterStemmer")
for word in words2:
    print(word + " ----> " + p_stemmer.stem(word))

print("\n")

print("Stemming using SnowballStemmer")
s_stemmer = SnowballStemmer(language="english")
for word in words2:
    print(word + " ----> " + s_stemmer.stem(word))
