import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from score import quadratic_weighted_kappa


class randomForestRegression(object):
    """Use random forest regression model to train the data and make predictions on the validation data"""
    
    def __init__(self, trainingFileName, validationFileName, validationScoreFileName):
       self.trainingFileName = trainingFileName
       self.validationFileName = validationFileName
       self.validationScoreFileName = validationScoreFileName


    def readTrainingValidationData(self):
        """Read training data and validation data from files"""
        try:
            trainingData = pd.read_csv(self.trainingFileName)
        except Exception, e:
            print "Cannot open the training data due to the exception:", e
            raise e

        try:
            validationData = pd.read_csv(self.validationFileName)
        except Exception, e:
            print "Cannot open the validation data due to the exception:", e
            raise e

        return trainingData, validationData


    def readVaidationScores(self, essaySetNumber):
        """Read validation scores from file"""
        
        try:
            score = open(self.validationScoreFileName, "rb")
            scoreData = pickle.load(score)
        except Exception, e:
            print "Cannot open the validation scores data due tp the exception:", e
            raise e
        return scoreData[essaySetNumber-1]


    def randomForestRegressionModel(self, essaySetNumber):
        """Train the random forest model using the training data"""
        
        trainingData, validationData = self.readTrainingValidationData()
        if essaySetNumber != 2:
            validationDomainOneScore = self.readVaidationScores(essaySetNumber)
        else:
            validationDomainOneScore = self.readVaidationScores(essaySetNumber)[0::2]
            validationDomainTwoScore = self.readVaidationScores(essaySetNumber)[1::2]
        
        maxScore = 0
        maxDepth = 0
        minLeaf = 0
        minSplit = 0
        
        trainingData = trainingData.drop(["Unnamed: 0", "domain1_score"], 1)
        validationData = validationData.drop(["Unnamed: 0", "domain1_score"], 1)
        trainingDomainOneScore = trainingData["domain1_score"]
        if essaySetNumber == 2:
            trainingDomainTwoScore = trainingData["domain2_score"]
            validationData = validationData.drop("domain2_score", 1)

        # train model and select best parameters
        for depth in range(6, 50, 10):
            for split in range(1, 20, 5):
                for leaf in range(20, 80, 10):
                    clf = RandomForestRegressor(max_depth=depth, min_samples_split=split, min_samples_leaf=leaf)
                    clf = clf.fit(trainingData, trainingDomainOneScore)
                        
                    # make predictions
                    predictDomainOneScores = clf.predict(validationData)
                    predictDomainOneScores = map(round, predictDomainOneScores)
                    predictDomainOneScores = map(int, predictDomainOneScores) 
                    score = quadratic_weighted_kappa(validationDomainOneScore, predictDomainOneScores)
                    
                    if essaySetNumber == 2:
                        clf = RandomForestRegressor(max_depth=depth, min_samples_split=split, min_samples_leaf=leaf)
                        clf = clf.fit(trainingData, trainingDomainTwoScore)
                        
                        predictDomainTwoScores = clf.predict(validationData)
                        predictDomainTwoScores = map(round, predictDomainTwoScores)
                        predictDomainTwoScores = map(int, predictDomainTwoScores)
                        score = (score + quadratic_weighted_kappa(validationDomainTwoScore, predictDomainTwoScores)) / 2.0
                    
                    if score > maxScore:
                        maxScore = score
                        maxDepth = depth
                        minLeaf = leaf
                        minSplit = split

        return  maxScore, maxDepth, minSplit, minLeaf


    def evaluatPredictions(self, essaySetNumber):
        """Make evaluations on predictions"""
        max = self.randomForestRegressionModel(essaySetNumber) 
        return max
        # return quadratic_weighted_kappa(trueScore, prediction)

if __name__ == '__main__':
    f = open("maximalKappaScores.txt", "wb")
    for i in range(1, 9):
        train = "trainingTfidfEssaySet" + str(i) + ".csv"
        test = "testingTfidfEssaySet" + str(i) + ".csv"
        model = randomForestRegression(train, test, "validationScores.pkl")
        max, depth, split, leaf = model.randomForestRegressionModel(i)
        f.write("Kappa score = %f, max_depth = %d, min_samples_split = %d, min_samples_leaf = %d \n" %(max, depth, split, leaf))
    f.close()
