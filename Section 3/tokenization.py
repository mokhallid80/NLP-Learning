import spacy
from spacy import displacy


nlp = spacy.load('en_core_web_sm')


myString = '"We\'re moving to L.A.!"'
#used \ so that it doesn't end the string right there.
print(myString)

doc = nlp(myString)
for token in doc:
    print(token.text)



# It is smart enough to detect when a full stop is going to be a stand-alone token
doc2 = nlp(u"We're here to help! send snail-mail, support!@g.com")
for token in doc2:
    print(token)

doc3 = nlp(u"A 5km NYC cab ride costs $10.30")
for token in doc3:
    print(token)


doc4 = nlp(u"Let's visit St. Louis in the U.S. next year")
for token in doc4:
    #to seperate each token 
    print(token, end=' | ')

# doc4.vocab "The number of tokenz in the language we imported in the beggining"

doc5 = nlp(u"Apple to build a Hong Kong factory for $6 million")
for entity in doc5.ents:
    print(entity)
    print(entity.label_)
    print(str(spacy.explain(entity.label_)))
    print('\n')

doc6 = nlp(u'Autonomous cars shift insurance liability towards manufactures.')

for chunk in doc6.noun_chunks:
    print(chunk)


# Visualization
doc = nlp(u'Apple is going to build a UK. factory for $6 million.')
displacy.serve(doc, style='dep',options={'distance':110})