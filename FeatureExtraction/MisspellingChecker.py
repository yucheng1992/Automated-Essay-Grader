import pandas as pd
import numpy as np
import sys
sys.path.append("..")
import unicodedata
import enchant
import regex as re
from nltk import bigrams
from collections import Counter
import pickle


def removePunctuation(text):
    return re.sub(ur'\p{P}+', ' ', text)


def misspellingCount(essay):
    """
    Count the misspelling words in each essay, using the dictionary in Enchant as the standard dictionary.
    :param essay: string
    :return: int, number of misspellings.
    """
    d = enchant.Dict('en_US')
    misspellingCount = 0
    words = removePunctuation(essay).split()
    for w in words:
        if not d.check(w):
            misspellingCount += 1
    return misspellingCount


def misspellingCountSet(essaySet):
    """
    Count the misspelling words for each essay in the set.
    :param essaySet: a list of string
    :return: a list of integers, representing the number of misspellings in each essay.
    """
    misspellingCountList = []
    for essay in essaySet:
        essay = essay.decode('utf-8', 'ignore')
        d = enchant.Dict('en_US')
        misspellingCountList.append(misspellingCount(essay))
	
    return misspellingCountList	


def main():
    train = pd.DataFrame.from_csv('Data/training_set_rel3.tsv', sep='\t')
    validate = pd.DataFrame.from_csv('Data/valid_set.tsv', sep='\t')
	
    misspellingCountTrainList = misspellingCountSet(train['essay'])
    misspellingCountValidateList = misspellingCountSet(validate['essay'])

    try:
        misspellingCountTrain_pkl = open('FeatureData/misspellingCountTrain.pkl', 'wb')
        pickle.dump(misspellingCountTrainList, misspellingCountTrain_pkl)
        misspellingCountTrain_pkl.close()
    except Exception:
        print "Cannot write word_list into file due to the exception:", sys.exc_info()[0]
        raise
    
    try:	
        misspellingCountValidate_pkl = open('FeatureData/misspellingCountValidate.pkl', 'wb')
        pickle.dump(misspellingCountValidateList, misspellingCountValidate_pkl)
        misspellingCountValidate_pkl.close()
    except Exception:
        print "Cannot write word_list into file due to the exception:", sys.exc_info()[0]
        raise
	
if __name__ == '__main__':
    main()
