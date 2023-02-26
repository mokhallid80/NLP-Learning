import spacy
from spacy.tokens import Span
from spacy.matcher import PhraseMatcher


nlp = spacy.load('en_core_web_sm')

def show_ents(doc):
    if(doc.ents):
        for ent in doc.ents:
            print(ent.text + ' - '+ent.label_ + ' - '+str(spacy.explain(ent.label_)))
    else:
        print('No entities found')
    print('\n')
    

doc = nlp(u'Hi how are you?')
show_ents(doc)

doc = nlp(u'May I go to New York, DC next May to see the New York Mounment?')
show_ents(doc)

doc = nlp(u'Apple made a new iPhone')
show_ents(doc)

doc = nlp(u'I ate an apple')
show_ents(doc)

doc = nlp(u'Tesla to build a U.K. factory for $6 million')
show_ents(doc)
# It didn't detect Tesla as an Org


# Adding a new entity
ORG = doc.vocab.strings[u"ORG"]
print(ORG)
# from 0 to before 1
new_ent = Span(doc, 0, 1, label=ORG)
doc.ents = list(doc.ents) + [new_ent]


show_ents(doc)
# it did detect it here

# why it didn't work here?
# because in this way, we are only detecting Tesla in this span 'the doc we provided to the Span'
doc = nlp(u'I drive a Tesla')
show_ents(doc)


# Adding multiple phrases as a named entities 

doc = nlp(u'Our company created a brand new vacuum cleaner.'
          u'This new vacuum-cleaner is the best.')

matcher = PhraseMatcher(nlp.vocab)

phrase_list = ['vacuum cleaner','vacuum-cleaner']

phrase_patterns = [nlp(text) for text in phrase_list]

matcher.add('newProduct',None, *phrase_patterns)

found_matches = matcher(doc)

PROD = doc.vocab.strings[u'PRODUCT']

new_ents = []
for match in found_matches:
    new_ents.append(Span(doc, match[1], match[2], label=PROD) )
doc.ents = list(doc.ents) + new_ents

show_ents(doc)

doc = nlp(u'Our company created a brand new vacuum cleaner.')
show_ents(doc)
# why does it only finds in the first doc ?


# How to count occurrences of a label
doc = nlp(u'Originally I paid $29.99 for this car toy, but now it is marked down by $10.00')

c = [ent for ent in doc.ents if ent.label_ == "MONEY"]
print(len(c))