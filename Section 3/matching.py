import spacy
from spacy.matcher import Matcher
from spacy.matcher import PhraseMatcher

nlp = spacy.load('en_core_web_sm')

matcher = Matcher(nlp.vocab)

# we wanna detect it in the following cases
# SolarPower
# Solar-power
# Solar power

pattern1 = [{'LOWER':'solarpower'}]
# if we transformed the word to lower case 

pattern2 = [{'LOWER':'solar'},{'IS_PUNCT':True},{'LOWER':'power'}]
# will find anything in the following format solarXpower where X is any punct

pattern3 = [{'LOWER':'solar'},{'LOWER':'power'}]
# solar power


# Now, we gonna add the patterns to our matcher
# none is a call back function
matcher.add('SolarPower', None, pattern1, pattern2, pattern3)



doc = nlp(u"The solar power industry is growing. Solar-power is the future.")

found_matches = matcher(doc)
print(found_matches)


# we can delete patterns too.
matcher.remove('SolarPower')

pattern1 = [{'LOWER':'solarpower'}]
pattern2 = [{'LOWER':'solar'},{'IS_PUNCT':True,'OP':'*'},{'LOWER':'power'}]
# pattern2 will allow us to match sth like solar-----power (any amount of PUNCTs inbetweem)
matcher.add('SolarPower', None, pattern1, pattern2)

doc2 = nlp(u"Solar--power is solar---power yay!")
found_matches = matcher(doc2)
print(found_matches)




# phrase matcher

p_matcher = PhraseMatcher(nlp.vocab)

doc3 = ""
with open('../UPDATED_NLP_COURSE/TextFiles/reaganomics.txt', encoding = 'latin1') as f:
    doc3 = nlp(f.read())


phrase_list =['voodoo economics', 'supply-side economics', 'trickle-down economics', 'free-market economics']

# list of docs
phrase_patterns = [nlp(text) for text in phrase_list]


p_matcher.add('EconMatcher',None, *phrase_patterns)

found_matches = p_matcher(doc3)

# print(found_matches)
for match_id, start, end in found_matches:
    string_id = nlp.vocab.strings[match_id]  # get string representation
    span = doc3[start:end]                    # get the matched span

    print(match_id, string_id, start, end, span.text)
# all of the returned indexes are for tokens indexes. Not chars