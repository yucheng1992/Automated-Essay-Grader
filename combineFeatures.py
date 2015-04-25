import numpy as np
import pandas as pd
import pickle
from parseArticle import *
from collections import Counter
from util import *


class combineFeatures():
    """Combine all the necessary features for every article."""
    
    def __init__(self):
        self.spellcheckerFileName = "bigram.pkl"
        self.posFileName = "featuresPos.pkl"
        self.wordsFileName = "featureWords.pkl"
        self.wordNumber, self.sentenceNumber, self.averageWordLength, self.clauseWordNumber = self.readFeaturesFromParseArticle()
        self.spellData, self.posData, self.wordsData = self.readFile()

    def readFile(self):
        """Read data from specific files"""
        try:
            fSpell = open(self.spellcheckerFileName, "rb")
            spellData = pickle.load(fSpell)
            
            fPos = open(self.posFileName, "rb")
            posData = pickle.load(fPos)

            fWords = open(self.wordsFileName, "rb")
            wordsData = pickle.load(fWords)
            
        except Exception, e:
            print "Cannot open file due to the exception: "
            print e
            raise
        return spellData, posData, wordsData

    
    def readFeaturesFromParseArticle(self):
        """read all the features from parseArticle"""
        
        articleFeatures = parseArticle()
        return articleFeatures.generateFeatures()


    def combineAllFeatures(self):
        """combine all the articless features together"""
        combineData = []
        for i in range(len(self.wordNumber)):
            featureCounter = self.posData[i] + self.wordsData[i]
            featureCounter["wordNumber"] = self.wordNumber[i]
            featureCounter["sentenceNumber"] = self.sentenceNumber[i]
            featureCounter["averageWordLength"] = self.averageWordLength[i]
            featureCounter["clauseWordNumber"] = self.clauseWordNumber[i]
            combineData.append(featureCounter)
        return combineData


    def writeToCsv(self):
        """write the combined features to a csv file"""
        features = self.combineAllFeatures()
        
        totalCounter = Counter()

        for feature in features:
            totalCounter = increment(totalCounter, 0,feature)

        for feature in features:
            feature += totalCounter
        return features

if __name__ == '__main__':
    cf = combineFeatures()
    cf.writeToCsv()
