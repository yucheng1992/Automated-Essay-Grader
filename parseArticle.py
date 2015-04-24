import csv
import pandas as pd
import re
import sys

class parseArticle():
    '''This is a class built for generating features from all the articles.''' 
    def __init__(self):
        """initiate the class"""
        self.dataFileName = "training_set_rel3.tsv"
        self.stopWordsFileName = "stopWords.txt"
        self.clauseWordsList = ["which", "where", "what", "why", "who"]
        self.article = self.parseTsvFile()["essay"]
        self.stopWords = self.readStopWords()
        self.wordNumber = []
        self.sentenceNumber = []
        self.averageWordLength = []
        self.clauseWordNumber = []

    def parseTsvFile(self):
        '''Read the essays from the tsv file ans return a dataframe containing all the essays.'''
    
        df = pd.DataFrame
    
        try:
            df = pd.DataFrame.from_csv(self.dataFileName, sep="\t")
        except Exception:
            print "Cannot open the file due to exception", sys.exc_info()[0]
        
        return df
    
    
    def readStopWords(self):
        """Read a list of stop words from file"""
        
        stopWords = []
    
        try:
            f = open(self.stopWordsFileName)
            for line in f:
                stopWords.append(line.rstrip())
            return stopWords
        except Exception:
            print "Cannot open the stop words file due to exception", sys.exc_info()[0]
    
    
    def deleteStopWords(self, article):
        """delete the stop words of an article"""
        
        wordList = []
        words = article.split(" ")
        for word in words:
            if word not in self.stopWords:
                wordList.append(word)
        return wordList
    
    
    def countWord(self, wordList):
        '''Take an article as input and return the numbers of its words'''
    
        return len(wordList)
    
    
    def countSentence(self, article):
        '''Calculate the number of sentences in an article'''
    
        sentences = re.split("\.|!", article)
        return len(sentences)
    
    
    def countAverageWordLength(self, wordList):
        """calculate the average word length of an article."""
        totalLength = 0
        for word in wordList:
            totalLength += len(word)
        
        return float(totalLength) / self.countWord(wordList)
    
    
    def countClauseWord(self, article):
        """Calculate the number of clause words in an article"""
        num = 0
        wordList = article.split(" ")
        for word in wordList:
            if word in self.clauseWordsList:
                num += 1
        return num

    def generateFeatures(self):
        """generate necessary features for all the articles"""
        for essay in self.article:
            self.wordNumber.append(self.countWord(self.deleteStopWords(essay)))
            self.sentenceNumber.append(self.countSentence(essay))
            self.averageWordLength.append(self.countAverageWordLength(self.deleteStopWords(essay)))
            self.clauseWordNumber.append(self.countClauseWord(essay))
        return self.wordNumber, self.sentenceNumber, self.averageWordLength, self.clauseWordNumber


if __name__ == "__main__":
    articleFeatures = parseArticle()
    print "==================Generating features=================="
    wordNumber, sentenceNumber, averageWordLength, clauseWordNumber = articleFeatures.generateFeatures()
    print "==================Done!=================="
