import pandas as pd
import numpy as np
import enchant
import regex, string
from nltk import bigrams
from collections import Counter

def bigram(article):

    cnt = Counter()
    sentences = regex.split("(?V1)(?<=[a-z])(?=[A-Z])|(?<=[.!?]) +(?=[A-Z])", article)
    for sentence in sentences:
        words = sentence.translate(None, string.punctuation).split()
        string_bigrams = bigrams(words)
        for j in string_bigrams:
            cnt[j] += 1
			
    return cnt
	
def misspelling_count(article):
    d = enchant.Dict('en-US')
    misspelling_count = 0
    words = article.translate(None, string.punctuation).split()
    for w in words:
        try:
            if not d.check(w):
                misspelling_count += 1
        except:
            pass
    return misspelling_count
	
def main():
    train = pd.read_csv('training_set_rel3.csv', index_col = 'essay_id')
	
    bigram_list = []
    misspelling_count_list = []
	
    for i in train['essay']:
        bigram_list.append(bigram(i))
		
        misspelling_count_list.append(misspelling_count(i))
	
    train['bigram'] = bigram_list
    train['misspelling_count'] = misspelling_count_list

    train.to_csv('train_ngram_spellchecker.csv')
    	
if __name__ == '__main__':
    main()
