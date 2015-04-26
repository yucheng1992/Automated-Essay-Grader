import numpy as np
import pandas as pd
import pickle
from parseArticle import *
from collections import Counter
from util import *
from operator import itemgetter


class combineFeatures():
    """Combine all the necessary features for every article."""
    
    def __init__(self):
        self.spellcheckerFileName = "bigram.pkl"
        self.posFileName = "featuresPosEssaySet"
        self.wordsFileName = "featureWordsEssaySet"
        self.missSpellFileName = "missSpellingCount.pkl"
        self.dataFileName = "training_set_rel3.tsv"
        self.wordNumber, self.sentenceNumber, self.averageWordLength, self.clauseWordNumber = self.readFeaturesFromParseArticle()
        # self.posData, self.missSpellData = self.readFile()

    
    def returnDomain1Score(self, essaySetNumber):
        """select all the domain1 scores from the dataset"""
        
        df = pd.DataFrame

        try:
            df = pd.DataFrame.from_csv(self.dataFileName, sep="\t")
        except Exception:
            print "Cannot open the data file due to the exception:", sys.exc_info()[0]
        
        mask = df["essay_set"] == essaySetNumber

        return df[mask]["domain1_score"].tolist()


    def readFile(self, essaySetNumber):
        """Read data from specific files"""
        try:
            # fSpell = open(self.spellcheckerFileName, "rb")
            # spellData = pickle.load(fSpell)
            print "============Loading essay set%d's features from file============" %(essaySetNumber)    
            fileName = self.posFileName + str(essaySetNumber) + ".pkl"

            fPos = open(fileName, "rb")
            posData = pickle.load(fPos)

            fMiss = open(self.missSpellFileName, "rb")
            missData = pickle.load(fMiss)
            # fWords = open(self.wordsFileName, "rb")
            # wordsData = pickle.load(fWords)
            print "============Essay set%d's features have been loaded!===========" %(essaySetNumber)
            
        except Exception, e:
            print "Cannot open file due to the exception: "
            print e
            raise
        return posData, missData

    
    def readFeaturesFromParseArticle(self):
        """read all the features from parseArticle"""
        
        articleFeatures = parseArticle()
        return articleFeatures.generateFeatures()


    def combineAllFeatures(self, essaySetNumber):
        """combine all the articless features together"""
         
        scores = self.returnDomain1Score(essaySetNumber)
        posData, missData = self.readFile(essaySetNumber) 
        combineData = []
        for i in range(len(posData)):
            # featureCounter = self.posData[i] + self.wordsData[i]
            featureCounter = posData[i]
            featureCounter["wordNumber"] = self.wordNumber[essaySetNumber-1][i]
            featureCounter["sentenceNumber"] = self.sentenceNumber[essaySetNumber-1][i]
            featureCounter["averageWordLength"] = self.averageWordLength[essaySetNumber-1][i]
            featureCounter["clauseWordNumber"] = self.clauseWordNumber[essaySetNumber-1][i]
            featureCounter["missSpelling"] = missData[i]
            featureCounter["score"] = scores[i]
            combineData.append(featureCounter)
        return combineData


    def writeToCsv(self):
        """write the combined features to a csv file"""
        for i in range(1, 9):
            print "============Combining essay set%d's features============" %(i)
            features = self.combineAllFeatures(i)
            print "============Essay set%d's features have been combined!============" %(i)
            print 

            totalCounter = Counter()

            for feature in features:
                totalCounter = increment2(totalCounter, 0, feature)
            
            for feature in features:
                feature = increment2(feature, 1, totalCounter)
            
            df = pd.DataFrame(features)
            
            print "============Writing essay set%d's features to csv file============" %(i)
            df.to_csv("essaySet{}.csv".format(i))
            print "============Essay set%d's features have been written to csv file!============" %(i)
            print 
       

if __name__ == '__main__':
    cf = combineFeatures()
    cf.writeToCsv()
