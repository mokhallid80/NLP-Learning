# RUN THIS CELL to perform standard imports:
import spacy
from spacy.matcher import Matcher


nlp = spacy.load('en_core_web_sm')

with open('../UPDATED_NLP_COURSE/TextFiles/owlcreek.txt', encoding = 'latin1') as f:
    doc = nlp(f.read())

print("#Tokens ",len(doc))

sentences = []
for sen in doc.sents:
    sentences.append(sen)

print("#Sentences ",len(sentences))

secondSentence = sentences[1]
print("2nd Sentence ", secondSentence)
    
for token in secondSentence:
    print(token.text, token.pos_, token.dep_, token.lemma_)



matcher = Matcher(nlp.vocab)


pattern = [{"LOWER":"swimming"}, {"IS_SPACE":True}, {"LOWER":"vigorously"}]

matcher.add('Swimming', None, pattern)

found_matches = matcher(doc)

print(found_matches)


def find_sentence(start, end):
    # we have the start and the end of the token
    # let the start = 5 and end = 7. So, matching is only
    # word number 5 and 6 out of all the whole doc
    # each sentence got from n to k tokens
    # we need to make sure that 5 >= n and 6 <= k

    seen_tokens = 0
    for sen in sentences:
        s_start = seen_tokens
        s_end = s_start + len(sen)

        if(start >= s_start and end <= s_end):
            return sen

        seen_tokens += len(sen)
    
    return None




for match_id, start, end in found_matches:
    string_id = nlp.vocab.strings[match_id]  # get string representation
    span = doc[start-5:end+5]     
    print("Span ", span)

    # Now we have the start word and the end word
    # we can go over each sentence and if coontains the index, print it out
    print("The Sentence Is ", find_sentence(start, end))
    print("\n")