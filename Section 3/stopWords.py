import spacy

nlp = spacy.load('en_core_web_sm')

# print(nlp.Defaults.stop_words)


# We can check if a word is a stop word


print(nlp.vocab['is'].is_stop) #returns True
print(nlp.vocab['goal'].is_stop) #returns False


print(nlp.vocab['btw'].is_stop) #returns False




# We can add more stopwords
nlp.Defaults.stop_words.add('btw')
nlp.vocab['btw'].is_stop = True

print(nlp.vocab['btw'].is_stop) #returns False



# We can remove words as well
nlp.Defaults.stop_words.add('beyond')
nlp.vocab['beyond'].is_stop = False
print(nlp.vocab['beyond'].is_stop) #returns False
