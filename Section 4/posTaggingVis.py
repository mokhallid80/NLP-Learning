import spacy
from spacy import displacy

nlp = spacy.load('en_core_web_sm')


doc = nlp(u'The quick brown fox jumped over the lazy dog\'s back')


options = {'distance':110,'color':'blue'}
# displacy.serve(doc, style='dep',options=options)

# when given a list of spans, it will draw each one of them alone.
doc2 = nlp(u'This is a sentence. This is another sentence. This is the very last long sentece')
spans = list(doc2.sents)
displacy.serve(spans, style='dep',options=options)