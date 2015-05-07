import csv
import pandas as pd
import re
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


class GenerateArticleFeatures():
    '''This is a class built for generating features from all the articles.''' 
    
    def __init__(self):
        """Initiate the class"""
        self.trainDataFileName = "../Data/training_set_rel3.tsv"
        self.testDataFileName = "../Data/valid_set.tsv"
        self.stopWordsFileName = "../Data/StopWords.txt"
        self.clauseWordsList = ["which", "where", "what", "why", "who"]
        self.trainArticle = self.parseTsvFile(self.trainDataFileName)["essay"]
        self.trainFile = self.parseTsvFile(self.trainDataFileName)
        self.testArticle = self.parseTsvFile(self.testDataFileName)["essay"]
        self.testFile = self.parseTsvFile(self.testDataFileName)
        self.stopWords = self.readStopWords()


    def parseTsvFile(self, fileName):
        '''
        Read the essays from the tsv file ans return a dataframe containing all the essays.
        @para   fileName: The name for the tsv file.
        @return df: The DataFrame that is changed from the tsv file.
        '''
    
        df = pd.DataFrame
    
        try:
            df = pd.DataFrame.from_csv(fileName, sep="\t")
        except Exception:
            print "Cannot open the file due to exception", sys.exc_info()[0]
        
        return df
    
    
    def readStopWords(self):
        """
        Read a list of stop words from the stop words file.
        @return stopWords: A list that contains all the stop words read from the stop words file.
        """
        
        stopWords = []
    
        try:
            f = open(self.stopWordsFileName)
            for line in f:
                stopWords.append(line.rstrip())
        except Exception:
            print "Cannot open the stop words file due to exception", sys.exc_info()[0]
        return stopWords
    

    def deleteStopWords(self, article):
        """
        Delete the stop words of an article
        @para   article:  The specific essay.
        @return wordList: A list contains all the words from the essay excluding the words in the stop words list.
        """
        
        wordList = []
        words = article.split(" ")
        for word in words:
            if word not in self.stopWords:
                wordList.append(word)
        return wordList
    
    
    def countWord(self, wordList):
        '''
        Take an article as input and return the numbers of its words
        @para   wordList: A list which contains all the words in the essay excluding the stop words.
        @return length: The number of the words in the essay.
        '''
        length = len(wordList)
        return length
    
    
    def countSentence(self, article):
        '''
        Calculate the number of sentences in an article.
        @para   article: The specific essay.
        @return length: The number of the sentences in the essay.
        '''
    
        sentences = re.split("\.|!", article)
        length = len(sentences)
        return length
    
    
    def countAverageWordLength(self, wordList):
        """
        calculate the average word length of an article.
        @para    wordList: A list which contains all the words in the essay excluding the stop words.
        @return  averageWordLength: The average word length of all the words in the essay excluding the stop words.
        """
        totalLength = 0
        for word in wordList:
            totalLength += len(word)
        averageWordLength = float(totalLength) / self.countWord(wordList)
        return averageWordLength
    
    
    def countClauseWord(self, article):
        """
        Calculate the number of clause words in an article
        @para   article: The specific essay
        @return num: The number of the clause words in the article.
        """
        num = 0
        wordList = article.split(" ")
        for word in wordList:
            if word in self.clauseWordsList:
                num += 1
        return num


    def generateFeatures(self):
        """
        Generate all the necessary features for all the articles in the eight essay sets.
        """
        
        # Initialize training set's feature list.
        trainTotalWordNumber = []
        trainTotalSentenceNumber = []
        trainTotalAverageWordLength = []
        trainTotalClauseWordNumber = []
        trainTotalFeature = []
        trainTotalTfidf = []

        # Initialize test set's feature list.
        testTotalWordNumber = []
        testTotalSentenceNumber = []
        testTotalAverageWordLength = []
        testTotalClauseWordNumber = []
        testTotalFeature = []
        testTotalTfidf = []
        
        for i in range(1, 9):
            # Select the essays from the DataFrame.
            trainMask = self.trainFile["essay_set"] == i
            trainEssaySet = self.trainFile[trainMask]["essay"]
            
            testMask = self.testFile["essay_set"] == i
            testEssaySet = self.testFile[testMask]["essay"]
            
            vectorizer = CountVectorizer(decode_error="replace", strip_accents="unicode", stop_words=self.stopWords)
            
            print "============================Transforming EssaySet%d's training articles to word vectors==========================" %i
            trainX = vectorizer.fit_transform(trainEssaySet.tolist())
            print "=======================EssaySet%d's training articles have been transformed to word vectors======================" %i 
            
            print "============================Transforming EssaySet%d's testing articles to word vectors===========================" %i
            testX = vectorizer.transform(testEssaySet.tolist())
            print "========================EssaySet%d's testing articles have been transformed to word vectors======================" %i
            
            transformer = TfidfTransformer()
            
            print "=====================Transforming EssaySet%d's training articles' bag of words to tf-idf vector==================" %i
            transformer.fit(trainX.toarray())
            trainTfidf = transformer.transform(trainX.toarray()).toarray()
            transformer.fit(testX.toarray())
            print "==================EssaySet%d's training articles' bag of words have been tranformed to tf-idf vector=============" %i
            
            print "=====================Transforming EssaySet%d's testing articles' bag of words to tf-idf vector===================" %i
            testTfidf = transformer.transform(testX.toarray()).toarray()
            print "==================EssaySet%d's training articles' bag of words have been tranformed to tf-idf vector=============" %i   
            print

            trainTotalFeature.append(vectorizer.get_feature_names())
            trainTotalTfidf.append(trainTfidf)
            testTotalTfidf.append(testTfidf)
           
            # Append the training set's basic features.
            wordNumber = []
            sentenceNumber = []
            averageWordLength = []
            clauseWordNumber = []
            for essay in trainEssaySet:
                wordNumber.append(self.countWord(self.deleteStopWords(essay)))
                sentenceNumber.append(self.countSentence(essay))
                averageWordLength.append(self.countAverageWordLength(self.deleteStopWords(essay)))
                clauseWordNumber.append(self.countClauseWord(essay))
            trainTotalWordNumber.append(wordNumber)
            trainTotalSentenceNumber.append(sentenceNumber)
            trainTotalAverageWordLength.append(averageWordLength)
            trainTotalClauseWordNumber.append(clauseWordNumber)
            
            # Append the test set's basic features.
            wordNumber = []
            sentenceNumber = []
            averageWordLength = []
            clauseWordNumber = []
            for essay in testEssaySet:
                wordNumber.append(self.countWord(self.deleteStopWords(essay)))
                sentenceNumber.append(self.countSentence(essay))
                averageWordLength.append(self.countAverageWordLength(self.deleteStopWords(essay)))
                clauseWordNumber.append(self.countClauseWord(essay))
            testTotalWordNumber.append(wordNumber)
            testTotalSentenceNumber.append(sentenceNumber)
            testTotalAverageWordLength.append(averageWordLength)
            testTotalClauseWordNumber.append(clauseWordNumber)

        return trainTotalWordNumber, trainTotalSentenceNumber, trainTotalAverageWordLength, trainTotalClauseWordNumber, trainTotalFeature, trainTotalTfidf, testTotalWordNumber, testTotalSentenceNumber, testTotalAverageWordLength, testTotalClauseWordNumber, testTotalTfidf


if __name__ == '__main__':
    features = GenerateArticleFeatures()
    features.generateFeatures()
