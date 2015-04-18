import csv
import pandas as pd
import re
import sys


def parseTsvFile(fileName):
    '''
    Read the essays from the tsv file ans return a dataframe containing all the essays.
    '''
    df = pd.DataFrame

    try:
        df = pd.DataFrame.from_csv(fileName, sep="\t")
    except Exception:
        print "Cannot open the file due to exception", sys.exc_info()[0]
    
    return df


def countWord(article):
    '''
    Take an article as input and return the numbers of its words
    '''
    words = article.split(" ")
    return len(words)

def countSentence(article):
    '''
    Calculate the number of sentences in an article
    '''
    sentences = re.split("\.|!", article)
    return len(sentences)


if __name__ == "__main__":
    fileName = "training_set_rel3.tsv"
    article = parseTsvFile(fileName)
