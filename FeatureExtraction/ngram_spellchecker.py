import pandas as pd
import numpy as np
import sys
import unicodedata
import enchant
import regex as re
from nltk import bigrams
from collections import Counter
import pickle


def remove_punctuation(text):
    return re.sub(ur'\p{P}+', ' ', text)
	
def bigram(article):

    cnt = Counter()
    article = article.decode('utf-8', 'ignore')
    sentences = regex.split("(?V1)(?<=[a-z])(?=[A-Z])|(?<=[.!?]) +(?=[A-Z])", article)
    for sentence in sentences:
        words = remove_punctuation(sentence).split()
        string_bigrams = bigrams(words)
        for j in string_bigrams:
            cnt[j] += 1
			
    return cnt
	
def misspelling_count(article):

    d = enchant.Dict('en_US')
    misspelling_count = 0
    #article = article.decode('utf-8', 'ignore')
    words = remove_punctuation(article).split()
    for w in words:
        if not d.check(w):
            misspelling_count += 1
    return misspelling_count
	
def main():
    train = pd.DataFrame.from_csv('training_set_rel3.tsv', sep='\t')
	
    bigram_list = []
    misspelling_count_list = []
	
    for i in train['essay']:
        i = i.decode('utf-8', 'ignore')
        #bigram_list.append(bigram(i))
		
        d = enchant.Dict('en_US')
        misspelling_count_list.append(misspelling_count(i))
	
    #train['bigram'] = bigram_list
    #train['misspelling_count'] = misspelling_count_list
    try:
        misspelling_count_pkl = open('misspelling_count.pkl', 'wb')
        pickle.dump(misspelling_count_list, misspelling_count_pkl)
        misspelling_count_pkl.close()
    except Exception:
        print "Cannot write word_list into file due to the exception:", sys.exc_info()[0]
        raise
    #train.to_csv('train_ngram_spellchecker.csv')
	
if __name__ == '__main__':
    main()
