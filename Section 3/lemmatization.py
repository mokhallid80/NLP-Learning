import spacy

nlp = spacy.load('en_core_web_sm')

def show_lemmas(text):
    print(f'{"Text":{12}} {"POS":{6}} {"Lemma#":<{22}} {"Lemma"}')

    for token in text:
        print(f'{token.text:{12}} {token.pos_:{6}} {token.lemma:<{22}} {token.lemma_}')



doc1 = nlp(u'I am a runner running in a race bevause I love to run since I ran today.')
show_lemmas(doc1)

