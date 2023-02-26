import spacy
from scipy import spatial

nlp = spacy.load('en_core_web_md')


# doc to vector
x = nlp(u'lion').vector

tokens = nlp(u'dog cat pet')

# similarity values between 0 and 1
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))



print('\n')

tokens = nlp(u'like love hate')

# similarity values between 0 and 1
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

print('\n')


tokens = nlp(u'dog cat nargle')

# ovv -> out of vocab
for token in tokens:
    print(token.text, token.has_vector, token.vector_norm)



# you can calculate new vectors
cosine_similarity = lambda vec1,vec2: 1 - spatial.distance.cosine(vec1, vec2)

king = nlp.vocab['king'].vector
man = nlp.vocab['man'].vector
woman = nlp.vocab['woman'].vector
# king - man + woman
new_vector = king - man +woman

computed_similarities = []
# for all words in my vocab, we will
# compare the cosine_similarity with 
# the word's vecotr
for word in nlp.vocab:
    if word.has_vector:
        if word.is_lower:
            if word.is_alpha:
                similarity = cosine_similarity(new_vector, word.vector)
                computed_similarities.append((word,similarity))

# desc order for item in index 1
computed_similarities = sorted(computed_similarities, key=lambda item: -item[1])

for t in computed_similarities[0:10]:
    print(t[0].text, t[1])
