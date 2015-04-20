import csv
import pandas as pd
import re
import sys


def parseTsvFile(fileName):
    '''Read the essays from the tsv file ans return a dataframe containing all the essays.'''

    df = pd.DataFrame

    try:
        df = pd.DataFrame.from_csv(fileName, sep="\t")
    except Exception:
        print "Cannot open the file due to exception", sys.exc_info()[0]
    
    return df


def readStopWords(fileName):
    """Read a list of stop words from file"""
    
    stopWords = []

    try:
        f = open(fileName)
        for line in f:
            stopWords.append(line.rstrip())
        return stopWords
    except Exception:
        print "Cannot open the stop words file due to exception", sys.exc_info()[0]


def deleteStopWords(article, stopWords):
    """delete the stop words of an article"""
    
    wordList = []
    words = article.split(" ")
    for word in words:
        if word not in stopWords:
            wordList.append(word)
    return wordList



def countWord(wordList):
    '''Take an article as input and return the numbers of its words'''

    return len(wordList)


def countSentence(article):
    '''Calculate the number of sentences in an article'''

    sentences = re.split("\.|!", article)
    return len(sentences)


def countAverageWordLength(wordList):
    """
    calculate the average word length of an article.
    """
    totalLength = 0
    for word in wordList:
        totalLength += len(word)
    
    return float(totalLength) / countWord(wordList)


def countClauseWord(article, clauseWordsList):
    """Calculate the number of clause words in an article"""
    num = 0
    wordList = article.split(" ")
    for word in wordList:
        if word in clauseWordsList:
            num += 1
    return num


def main():
    """Main function of this module"""
    dataFileName = "training_set_rel3.tsv"
    article = parseTsvFile(dataFileName)

    stopWordsFileName = "stopWords.txt"
    stopWords = readStopWords(stopWordsFileName)
    
    clauseWordsList = ["which", "where", "what", "why", "who"]

    wordNumber = []
    sentenceNumber = []
    averageWordLength = []
    clauseWordNumber = []

    for essay in article["essay"]:
        wordNumber.append(countWord(deleteStopWords(essay, stopWords)))
        sentenceNumber.append(countSentence(essay))
        averageWordLength.append(countAverageWordLength(deleteStopWords(essay, stopWords)))
        clauseWordNumber.append(countClauseWord(essay, clauseWordsList))

if __name__ == "__main__":
   main() 
