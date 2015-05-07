import pandas as pd
import numpy as np
import sys
import pickle


class SelectValidationScores():
    """Select Validation Scores from the validation score file."""
    def __init__(self):
        '''
        Initialize the SelectValidationScores class
        '''
        self.validationScoreFileName = "valid_sample_submission_1_column.csv" 
        self.validationSetFileName = "valid_set.tsv"
        self.scores = self.readValidationScore()


    def readValidationScore(self):
        '''Read valiation score from the file'''
        try:
            df = pd.read_csv(self.validationScoreFileName)
        except:
            print "Cannot open the validation score file due to the exception:", sys.exc_info()[0]
            raise
        score = []

        for item in df["predicted_score"]:
            score.append(item)
        return score


    def readValidationSet(self):
        """read validation set from file"""
        
        df = pd.DataFrame
        
        try:
            df = pd.DataFrame.from_csv(self.validationSetFileName, sep="\t")    
        except Exception, e:
            raise e
        
        return df


    def splitValidationScores(self):
        '''Split the validation scores into 8 essay set.'''
        score = self.readValidationScore()
        df = self.readValidationSet()
        
        scoreLengthList = []

        for i in range(1, 9):
            mask = df["essay_set"] == i
            scoreLengthList.append(df[mask].shape[0])
        
        essaySetScores = []

        idxStart = 0

        for i in range(len(scoreLengthList)):
            if i == 1:
                essaySetScores.append(score[idxStart: scoreLengthList[i] * 2 + idxStart])
                idxStart += scoreLengthList[i] * 2
            else:
                essaySetScores.append(score[idxStart: scoreLengthList[i] + idxStart])
                idxStart += scoreLengthList[i]
        return essaySetScores


    def writeEssaySetScoresToFile(self):
        """write different validation essay set's scores to files"""
        
        score = self.splitValidationScores()

        try:
            scoreFile = open("validationScores.pkl", "wb")
            pickle.dump(score, scoreFile)
            scoreFile.close()
        except Exception, e:
            print "Cannot write validation scores to file due to the exception:", e
            raise e


if __name__ == '__main__':
    score = SelectValidationScores()
    score.writeEssaySetScoresToFile()
