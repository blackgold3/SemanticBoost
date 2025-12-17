import os
import numpy as np
import spacy
nlp = spacy.load('en_core_web_sm')

def process_text(sentence):
    sentence = sentence.replace('-', '')
    doc = nlp(sentence)
    tokens = []
    for token in doc:
        word = token.text
        if not word.isalpha():
            continue
        if (token.pos_ == 'NOUN' or token.pos_ == 'VERB') and (word != 'left'):
            word = token.lemma_
        
        pos = token.pos_
        tokens.append('%s/%s'%(word, pos))
    
    tokens = " ".join(tokens)
    return tokens

def action2text(action):
    if action == "":
        return '%s#%s#%s#%s\n'%('', '', '0.0', '0.0')

    tokens = process_text(action)
    begin = '0.0'
    end = '0.0'
    texts = '%s#%s#%s#%s'%(action, tokens, begin, end)
    return texts

def text2full(text, train=False):
    if len(text.split("#")) > 1:
        cap, tokens, begin, end = text.split("#")
        text = '%s#%s#%s#%s'%(cap, tokens, "0.0", "0.0")
        return text
    elif train:
        return '%s#%s#%s#%s'%(text, "", "0.0", "0.0")
    else:
        tokens = process_text(text)
        begin = '0.0'
        end = '0.0'
        texts = '%s#%s#%s#%s'%(text, tokens, begin, end)
        return texts      