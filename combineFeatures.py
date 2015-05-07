import numpy as np
import pandas as pd
import pickle
from GenerateArticleFeatures import *
from collections import Counter
from util import *
from operator import itemgetter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


class CombineFeatures():
    """Combine all the necessary features for every article."""
    
    def __init__(self):
        '''
        Initialize the CombineFeatures class.
        '''
        self.posTrainFileName = "featuresPosEssaySet"
        self.wordsTrainFileName = "featureWordsEssaySet"
        self.missSpellTrainFileName = "missSpellingCount.pkl"
        self.dataTrainFileName = "training_set_rel3.tsv"

        self.posTestFileName = "testFeaturesPosEssaySet"
        self.wordsTestFileName = "testFeatureWordsEssaySet"
        self.missSpellTestFileName = "misspelling_count.pkl"
        self.dataTestFileName = "valid_set.tsv"
        
        self.trainWordNumber, self.trainSentenceNumber, self.trainAverageWordLength, self.trainClauseWordNumber, self.trainFeature, self.trainTfidf, self.testWordNumber, self.testSentenceNumber, self.testAverageWordLength, self.testClauseWordNumber, self.testTfidf = self.readFeaturesFromGenerateArticleFeatures()

    
    def returnDomainOneScore(self, essaySetNumber):
        """
        Select all the domain1 scores from the dataset.
        @para   essaySetNumber: The number of the essay set.
        @return domainOneScore: The domain1_score of the training essay set.
        """
        
        df = pd.DataFrame

        try:
            df = pd.DataFrame.from_csv(self.dataTrainFileName, sep="\t")
        except Exception, e:
            print "Cannot open the data file due to the exception:", sys.exc_info()[0]
            raise e

        mask = df["essay_set"] == essaySetNumber
        domainOneScore = df[mask]["domain1_score"].tolist()
        return domainOneScore

    
    def returnDomainTwoScore(self):
        """
        Select the domain2_score from the essay set2.
        @return domainTwoScore: The domain2_score of the training essay set2.
        """

        df = pd.DataFrame

        try:
            df = pd.DataFrame.from_csv(self.dataTrainFileName, sep="\t")
        except Exception, e:
            print "Cannot open the train data file due to the exception:", sys.exc_info()[0]
            raise e
        
        mask = df["essay_set"] == 2
        domainTwoScore = df[mask]["domain2_score"].tolist()
        return domainTwoScore


    def readTrainFile(self, essaySetNumber):
        """
        Read training data's part of speech, misspellings, bag of words from pkl files.
        @para    essaySetNumber: The number of the training essay set.
        @return  posData:   The training set's part of speech data.
                 missData:  The training set's misspellings data.
                 wordsData: The training set's bag of words data.
        """
        try:
            print "==============Loading training Essay set%d's features from file==============" %(essaySetNumber)    
            fileName = self.posTrainFileName + str(essaySetNumber) + ".pkl"
            wordsFileName = self.wordsTrainFileName + str(essaySetNumber) + ".pkl"
            fPos = open(fileName, "rb")
            posData = pickle.load(fPos)

            fMiss = open(self.missSpellTrainFileName, "rb")
            missData = pickle.load(fMiss)
            fWords = open(wordsFileName, "rb")
            wordsData = pickle.load(fWords)
            print "===============Training Essay set%d's features have been loaded!=============" %(essaySetNumber)
            
        except Exception, e:
            print "Cannot open file due to the exception: "
            print e
            raise
        return posData, missData, wordsData


    def readTestFile(self, essaySetNumber):
        """
        Read testing data's part of speech, misspellings, bag of words from pkl files.
        @para   essaySetNumber: The number of the testing essay set.
        @return posData:  The testing set's part of speech data.
                missData: The testing set's misspellings data.
                wordsData The testing set's bag of words data.
        """
        try:
            print "===============Loading testing Essay set%d's features from file==============" %(essaySetNumber)
            fileName = self.posTestFileName + str(essaySetNumber) + ".pkl"
            wordsFileName = self.wordsTestFileName + str(essaySetNumber) + ".pkl"
            fPos = open(fileName, "rb")
            posData = pickle.load(fPos)

            fMiss = open(self.missSpellTestFileName, "rb")
            missData = pickle.load(fMiss)
            fWords = open(wordsFileName, "rb")
            wordsData = pickle.load(fWords)
            print "===============Testing Essay set%d's features have been loaded!==============" %(essaySetNumber)
            
        except Exception, e:
            print "Cannot open file due to the exception: "
            print e
            raise
        return posData, missData, wordsData

    
    def readFeaturesFromGenerateArticleFeatures(self):
        """
        Read all the features from parseArticle.
        """
         
        articleFeatures = GenerateArticleFeatures()
        return articleFeatures.generateFeatures()


    def combineAllFeatures(self, essaySetNumber, misspellingDataStartIndex, isTrain=1):
        """
        Combine all the articless features together.
        @para   essaySetNumber: The number of the essay set.
                misspellingDataStartIndex: The start index of the misspelling data of this essay set.
                isTrain: If isTrain is 1, this is a training set, and if isTrain is not 1, this is a test set.
        @return combineData: All the combined features of this essay set.
                j: The start index of the next essay set's misspelling data.
        """

        if isTrain == 1:
            domainOneScores = self.returnDomainOneScore(essaySetNumber)
            if essaySetNumber == 2:
                domainTwoScores = self.returnDomainTwoScore()
            posData, misspellingData, wordsData = self.readTrainFile(essaySetNumber) 
        else:
            posData, misspellingData, wordsData = self.readTestFile(essaySetNumber)
        combineData = []
        j = misspellingDataStartIndex
        print j
        for i in range(len(posData)):
            if isTrain == 1:
                dictionary = dict(zip(self.trainFeature[essaySetNumber-1], self.trainTfidf[essaySetNumber-1][i]))
                if essaySetNumber == 2:
                    print i
                dictionary = Counter(dictionary)
                featureCounter = posData[i] + dictionary
                featureCounter["wordNumber"] = self.trainWordNumber[essaySetNumber-1][i]
                featureCounter["sentenceNumber"] = self.trainSentenceNumber[essaySetNumber-1][i]
                featureCounter["averageWordLength"] = self.trainAverageWordLength[essaySetNumber-1][i]
                featureCounter["clauseWordNumber"] = self.trainClauseWordNumber[essaySetNumber-1][i]
                featureCounter["missSpelling"] = misspellingData[j]
                featureCounter["domain1_score"] = domainOneScores[i]
                if essaySetNumber == 2:
                    featureCounter["domain2_score"] = domainTwoScores[i]
            else:
                dictionary = dict(zip(self.trainFeature[essaySetNumber-1], self.testTfidf[essaySetNumber-1][i]))
                dictionary = Counter(dictionary)
                featureCounter = posData[i] + dictionary
                featureCounter["wordNumber"] = self.testWordNumber[essaySetNumber-1][i]
                featureCounter["sentenceNumber"] = self.testSentenceNumber[essaySetNumber-1][i]
                featureCounter["averageWordLength"] = self.testAverageWordLength[essaySetNumber-1][i]
                featureCounter["clauseWordNumber"] = self.testClauseWordNumber[essaySetNumber-1][i]
                featureCounter["missSpelling"] = misspellingData[j]
            
            combineData.append(featureCounter)
            j = j + 1
        transformer = TfidfTransformer()
        
        return combineData, j


    def writeToCsv(self):
        """
        write the combined features to a csv file
        """
        missDataStartTrainIdx = 0
        missDataStartTestIdx = 0
        for i in range(1, 9):
            print "======================Combining essay set%d's features=======================" %(i)
            trainFeatures, j = self.combineAllFeatures(i, missDataStartTrainIdx)
            testFeatures, k = self.combineAllFeatures(i, missDataStartTestIdx, 0)

            print "=================Essay set%d's features have been combined!==================" %(i)
            
            missDataStartTrainIdx = j
            missDataStartTestIdx = k

            totalCounter = Counter()

            for feature in trainFeatures:
                totalCounter = increment(totalCounter, 0, feature)
            for feature in testFeatures:
                totalCounter = increment(totalCounter , 0, feature)
            
            for feature in trainFeatures:
                feature = increment(feature, 1, totalCounter)
            for feature in testFeatures:
                feature = increment(feature, 1, totalCounter)
            
            dfTrain = pd.DataFrame(trainFeatures)
            dfTest = pd.DataFrame(testFeatures)
            print "=================Writing essay set%d's features to csv file==================" %(i)
            dfTrain.to_csv("trainingTfidfEssaySet{}.csv".format(i))
            dfTest.to_csv("testingTfidfEssaySet{}.csv".format(i))
            print "============Essay set%d's features have been written to csv file!============" %(i)
            print 
        return
        
if __name__ == '__main__':
    cf = combineFeatures()
    cf.writeToCsv()
