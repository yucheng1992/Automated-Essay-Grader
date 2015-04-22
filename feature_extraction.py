# coding=utf-8
import os
import numpy as np
import sys
import pandas as pd
import string
from collections import Counter
from nltk.corpus import stopwords
import pickle
from math import log
import util as util
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import csv

class EssayInstance:
    """
    Build the review class
    """
    
    def __init__(self):
        self.word_list = []
        self.word_dict = Counter()
        self.word_tfidf = Counter()
        self.pos_tags = Counter()

    def construct_word_dict(self,essay,stop_word = None):
        """
        count the words in word_list, transform to a dict of (word: word_count)
        :param stop_word: a list of stop words, The words you hope to filter, not included in dict, default set to None
        :param count_words: a list, the words you hope to keep in the count_dict, if set to None, will keep all the words
        :return:
        """
        symbols = string.punctuation
        words = map(lambda Element: Element.translate(None, symbols).strip(), essay.strip().split(' '))
        words = filter(None, words)
        new_word_list = []
        for word in words:
            try:
                word = decode(word)
                word = str(word)
                new_word_list.append(word)
            except:
                pass
        self.word_list = new_word_list
        
        c = Counter()
        for item in words:
        	c[item] += 1

        if stop_word:
            for i in stop_word:
                del c[i]

        self.word_dict = c
        if len(self.word_list) > 0:
            #print self.word_list
            pos_tag_list = pos_tag(self.word_list)
            pos_tags = [item[1] for item in pos_tag_list]
            symbols = string.punctuation
            pos_tags = map(lambda Element: Element.translate(None, symbols).strip(), pos_tags)
            pos_tags = filter(None, pos_tags)
            cnt = Counter()
            for i in pos_tags:
                cnt[i] += 1
            self.pos_tags = cnt

   
    def transform_to_tfidf(self, idf_dict):
        """
        Take a idf dict as input, constrcut the {word: tfidf} vector 
        tfidf = tf*(idf+1)
        """
        for word in self.word_dict:
            self.word_tfidf[word] = self.word_dict.get(word) * idf_dict.get(word, 1)

class corpus_df:
    """
    Build the review document-frequency class
    df is a dictionay of {word: frequency}
    """
    def __init__(self):
        self.df = Counter()
        self.idf = Counter()
        self.N = 0
        
    def fit(self, essay_list):
        """
        Construct the df Counter based on review_list
        """
        for essay in essay_list:
            current_df = Counter(essay.word_dict.keys())
            util.increment(self.df, 1, current_df)
        
        #inverse_df: log(N/n[word])
        self.N = len(essay_list)
        self.idf = self.df.copy()
        for word in self.idf:
            self.idf[word] = log(float(self.N)/self.idf[word])

def decode(words):
    return str(words.decode("utf8","ignore"))
def main():
    train = pd.read_csv('training_set_rel3.tsv',sep='\t')
    bagofwords_list = []
    essay_list = []
    pos_list = []
    i = 0
    for essay in train['essay']:
        print i
        e = EssayInstance()
        e.construct_word_dict(essay)
        essay_list.append(e)
        bagofwords_list.append(e.word_dict)
        pos_list.append(e.pos_tags)
        i = i+1
    corpus = corpus_df()
    corpus.fit(essay_list)	
    for i in essay_list:
        i.transform_to_tfidf(corpus.idf)
    
    with open('features_words.csv','wb') as f:
        writer = csv.writer(f)
        writer.writerow(bagofwords_list)
    with open('features_pos.csv','wb') as f:
        writer = csv.writer(f)
        writer.writerow(pos_list)
        
main()
