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
        self.posFileName = "featuresPosEssaySet1.pkl"
        self.wordsFileName = "featureWordsEssaySet1.pkl"
        self.missSpellFileName = "missSpellingCount.pkl"
        self.dataFileName = "training_set_rel3.tsv"
        self.wordNumber, self.sentenceNumber, self.averageWordLength, self.clauseWordNumber = self.readFeaturesFromParseArticle()
        self.posData, self.missSpellData = self.readFile()

    
    def returnDomain1Score(self):
        """select all the domain1 scores from the dataset"""
        
        df = pd.DataFrame

        try:
            df = pd.DataFrame.from_csv(self.dataFileName, sep="\t")
        except Exception:
            print "Cannot open the data file due to the exception:", sys.exc_info()[0]

        return df["domain1_score"].tolist()

    def readFile(self):
        """Read data from specific files"""
        try:
            # fSpell = open(self.spellcheckerFileName, "rb")
            # spellData = pickle.load(fSpell)
            
            fPos = open(self.posFileName, "rb")
            posData = pickle.load(fPos)

            fMiss = open(self.missSpellFileName, "rb")
            missData = pickle.load(fMiss)
            # fWords = open(self.wordsFileName, "rb")
            # wordsData = pickle.load(fWords)
            
        except Exception, e:
            print "Cannot open file due to the exception: "
            print e
            raise
        return posData, missData

    
    def readFeaturesFromParseArticle(self):
        """read all the features from parseArticle"""
        
        articleFeatures = parseArticle()
        return articleFeatures.generateFeatures()


    def combineAllFeatures(self):
        """combine all the articless features together"""
        
        scores = self.returnDomain1Score()
    
        combineData = []
        for i in range(len(self.posData)):
            # featureCounter = self.posData[i] + self.wordsData[i]
            featureCounter = self.posData[i]
            featureCounter["wordNumber"] = self.wordNumber[0][i]
            featureCounter["sentenceNumber"] = self.sentenceNumber[0][i]
            featureCounter["averageWordLength"] = self.averageWordLength[0][i]
            featureCounter["clauseWordNumber"] = self.clauseWordNumber[0][i]
            featureCounter["missSpelling"] = self.missSpellData[i]
            featureCounter["score"] = scores[i]
            combineData.append(featureCounter)
        return combineData


    def writeToCsv(self):
        """write the combined features to a csv file"""
        features = self.combineAllFeatures()
        
        totalCounter = Counter()
    

        for feature in features:
            totalCounter = increment2(totalCounter, 0,feature)
        

        for feature in features:
            feature = increment2(feature, 1, totalCounter)
            print feature    
        
        df = pd.DataFrame(features)
        
        df.to_csv("essaySet1.csv")
       

if __name__ == '__main__':
    cf = combineFeatures()
    cf.writeToCsv()
