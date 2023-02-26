import spacy

#importing English core language model, web_sm = smaller version
nlp = spacy.load('en_core_web_sm')

doc = nlp(u'Tesla is looking at buying U.S. startup for $6 million')

#now, using the nlp, it is gonna parse doc into tokens

for token in doc:
    print("Text "+token.text+" POS "+ str(token.pos_))

# print(nlp.pipeline)
# print(nlp.pipe_names)


doc2 = nlp(u"Tesla isn't looking into startups anymore.")
for token in doc2:
    print("Text "+token.text+" POS "+ str(token.pos_))


doc3 = nlp(u"This is the first sentence. This is another sentence. This is the lst ssentence")

for sentence in doc3.sents:
    print(sentence)