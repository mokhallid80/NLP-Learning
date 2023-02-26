import spacy

nlp = spacy.load('en_core_web_sm')


doc = nlp(u'The quick brown fox jumped over the lazy dog\'s back')

print(doc.text)
for token in doc:
    print(f"{token.text:{10}} {token.pos_:{10}} {token.tag_:{10}} {spacy.explain(token.tag_)}")


doc = nlp(u"I read books on NLP")

#checking the word 'read'
print(doc.text)
for token in doc:
    print(f"{token.text:{10}} {token.pos_:{10}} {token.tag_:{10}} {spacy.explain(token.tag_)}")
#notice how read here is in the present


doc = nlp(u"I read a book on NLP")

#checking the word 'read'
print(doc.text)
for token in doc:
    print(f"{token.text:{10}} {token.pos_:{10}} {token.tag_:{10}} {spacy.explain(token.tag_)}")
#notice how read here is in the past


doc = nlp(u'The quick brown fox jumped over the lazy dog\'s back')
pos_counts = doc.count_by(spacy.attrs.POS)
#returns the count of each POS code

#we can get the text o a code using doc.vocab[X].text
print(pos_counts)


for k,v in sorted(pos_counts.items()):
    print(f"{k}. {doc.vocab[k].text:{5}} {v}")

#you can do the same with fine-grained tags ( the more detailed tags)