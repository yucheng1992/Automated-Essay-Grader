import sys
sys.path.append("..")
import nltk
import string
import os
import operator
import pandas as pd
import numpy as np
import unicodedata
import sys, pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

tokenDict = {}
stemmer = PorterStemmer()

def stemTokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stemTokens(tokens, stemmer)
    return stems

def dotProduct(d1, d2):
   
    """
    @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
    @param dict d2: same as d1
    @return float: the dot product between d1 and d2
    """
    
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        return np.sum(d1.get(key, 0) * value for key, value in d2.items())


def cosSimilarity(tfidf, featureNames, tfidfDictPrompt, subset, label, i):
    """
    @param tfidf: a model trained from given text sets.
    @param featureNames: attribute from tfidf( featureNames = tfidf.get_feature_names())
    @param tfidfDictPrompt: dict, the tfidf score for essay prompt trained from tfidf.
    @param subset: pd.Series, treat the essay set separately according to their essay_set from 1-8.
    @param label: str, indicate whether it is from training set or validate set.
    @param i: number of essay_set, from 1-8.	
    @return: compute the cosine similarity of tfidf score for each essay with corresponding prompt. Save it in pickle.
    """
    simScore = []
    for essay in subset:
        responseEssay = tfidf.transform([essay.decode('utf-8', 'ignore')])
    
        tfidfDictEssay = {}
        for col in responseEssay.nonzero()[1]:
            tfidfDictEssay[featureNames[col]] = responseEssay[0, col]
    
        simScore.append(dotProduct(tfidfDictPrompt, tfidfDictEssay))
    
    try:
        simScore_pkl = open('FeatureData/simScore{}{}.pkl'.format(label, i), 'wb')
        pickle.dump(simScore, simScore_pkl)
        simScore_pkl.close()
    except Exception:
        print "Cannot write word_list into file due to the exception:", sys.exc_info()[0]
        raise 



def main():
    
    train = pd.read_csv('Data/training_set_rel3.tsv', sep = '\t', index_col = 0)
    validate = pd.read_csv('Data/valid_set.tsv', sep = '\t', index_col = 0)
    
    for i in range(3,7):
        
        ''' 
        Since only essay sets 3-6 are the type of Source Dependent Responses, which require students writing an essay
        according to the information provided in the prompt. We will compute the cosine similarity between prompt and essay using TF-IDF.
        Select all essay in set i in training and validate data and compute its cosine similarity with corresponding set prompt.
		'''
 
        trainSubset = train[train['essay_set'] == i]['essay']
        validateSubset = validate[validate['essay_set'] == i]['essay']
		
        # Read the prompt for set i.
        f = open('Data/essay_set_{}_prompt.txt'.format(i),'rb')
        prompt = f.read()
        prompt = prompt.translate(None, string.punctuation).decode('utf-8', 'ignore')
        prompt = unicodedata.normalize('NFKD', prompt).encode('ascii','ignore')
    
        # Add prompt to the end of the Series.
        trainSubset.append(pd.Series(prompt))
    
        # Compute the tfidf for given document sets.
        tokenList = []
        for essay in trainSubset:
            lowers = essay.lower()
            noPunctuation = lowers.translate(None, string.punctuation)
            tokenList.append(noPunctuation.decode('utf-8', 'ignore'))
        
        tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
        tfs = tfidf.fit_transform(tokenList)
    
        featureNames = tfidf.get_feature_names()
        responsePrompt = tfidf.transform([prompt])
    
        tfidfDictPrompt = {}
        for col in responsePrompt.nonzero()[1]:
            tfidfDictPrompt[featureNames[col]] = responsePrompt[0, col]
        
        # Compute the cosine similarity score using tfidf for each essay set in both training and validate dataset.
        cosSimilarity(tfidf, featureNames, tfidfDictPrompt, trainSubset, 'Train', i) 
        cosSimilarity(tfidf, featureNames, tfidfDictPrompt, validateSubset, 'Validate', i)

if __name__ == '__main__':
    main()
