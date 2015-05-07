import sys
sys.path.append("..")
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from Util.KappaScores import quadratic_weighted_kappa


class gradientBoostingRegressor(object):
    """Use Gradient Boosting Regressor to train the data and make predictions on the validation data"""
    
    def __init__(self, trainingFileName, validationFileName, validationScoreFileName):
        '''
        Initialize the model.
        @para trainingFileName:        The filename of the training file.
              validationFileName:      The filename of the validation data.
              validationScoreFileName: The filename of the validation data domain scores.
        '''
        self.trainingFileName = trainingFileName
        self.validationFileName = validationFileName
        self.validationScoreFileName = validationScoreFileName
	

    def evaluatPredictions(self, trueScores, predictScores):
        """
        Make evaluations on predictions
        @para trueScore:     The actual scores of the validation data.
              predictScores: The predicted scores that the model made on the validation data.
        @return score: The corresponding Quadratic Weighted Kappa score.
        """
        score = quadratic_weighted_kappa(trueScores, predictScores)
        return score


    def readTrainingValidationData(self):
        """
        Read training data and validation data from files.
        @return training data and validation data.
        """
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
        """
        Read validation scores from file.
        @para   essaySetNumber: The number of the essay set.
        @return score:          A list containing the validation data domain score of the specific essay set number.
        """
        
        try:
            score = open(self.validationScoreFileName, "rb")
            scoreData = pickle.load(score)
            score = scoreData[essaySetNumber-1]
        except Exception, e:
            print "Cannot open the validation scores data due tp the exception:", e
            raise e
        return score


    def gradientBoostingRegressor(self, essaySetNumber):
        """
        Train the random forest model using the training data.
        @para   essaySetNumber: The number of the essay set.
        @return score: The maximal Kappa Score.
        """
        
        trainingData, validationData = self.readTrainingValidationData()
        if essaySetNumber != 2:
            validationDomainOneScore = self.readVaidationScores(essaySetNumber)
        else:
            validationDomainOneScore = self.readVaidationScores(essaySetNumber)[0::2]
            validationDomainTwoScore = self.readVaidationScores(essaySetNumber)[1::2]
    
        trainingDomainOneScore = trainingData["domain1_score"]
        trainingData = trainingData.drop(["Unnamed: 0", "domain1_score"], 1)
        validationData = validationData.drop(["Unnamed: 0", "domain1_score"], 1)
        
        if essaySetNumber == 2:
            trainingDomainTwoScore = trainingData["domain2_score"]
            trainingData = trainingData.drop('domain2_score', 1)
            validationData = validationData.drop("domain2_score", 1)

        # Train the model, after trying several parameters, we find that tuning parameters in GradientBoostingRegressor
	# in this case does not help much.
        clf = GradientBoostingRegressor()
        clf = clf.fit(trainingData, trainingDomainOneScore)
                        
        # make predictions
        predictDomainOneScores = clf.predict(validationData)
        predictDomainOneScores = map(round, predictDomainOneScores)
        predictDomainOneScores = map(int, predictDomainOneScores) 
        score = self.evaluatPredictions(validationDomainOneScore, predictDomainOneScores)
                    
        if essaySetNumber == 2:
            clf = GradientBoostingRegressor()
            clf = clf.fit(trainingData, trainingDomainTwoScore)
                        
            predictDomainTwoScores = clf.predict(validationData)
            predictDomainTwoScores = map(round, predictDomainTwoScores)
            predictDomainTwoScores = map(int, predictDomainTwoScores)
            score = (score + self.evaluatPredictions(validationDomainTwoScore, predictDomainTwoScores)) / 2.0

        return  score


if __name__ == '__main__':
    # Write the result into a file called "GradientBoostingMaximalKappaScores.txt"
    f = open("../Result/GradientBoostingMaximalKappaScores.txt", "wb")
    for i in range(1, 9):
        train = "../TrainingData/trainingTfidfEssaySet" + str(i) + ".csv"
        test = "../TestData/testingTfidfEssaySet" + str(i) + ".csv"
        model = gradientBoostingRegressor(train, test, "validationScores.pkl")
        score = model.gradientBoostingRegressor(i)
        f.write("Kappa score = %f \n" %(score))
        print 'Set {} done'.format(i)
    f.close()
