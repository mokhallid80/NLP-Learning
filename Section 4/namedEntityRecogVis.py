import spacy
from spacy import displacy

nlp = spacy.load('en_core_web_sm')

doc = nlp(u'Over the last decade Apple sold nearly 10 million iPhones for a profit of $6 millions.By contrast, Samsung only sold 3 million.')


displacy.serve(doc, style='ent')


