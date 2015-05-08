from FeatureExtraction import ExtractPos
from FeatureExtraction import GenerateArticleFeatures
from FeatureExtraction import CombineFeatures
from FeatureExtraction import MisspellingChecker
from FeatureExtraction import SimilarityScore
from Model import GradientBoostingRegressionTreeModel
from Model import RandomForestRegressionModel
from Util import util
from Util import KappaScores
from Util import SelectValidationScores


if __name__ == '__main__':
    MisspellingChecker.main()
    SimilarityScore.main()
    SelectValidationScores.main()
    ExtractPos.main()
    CombineFeatures.main()
    GradientBoostingRegressionTreeModel.main()
    RandomForestRegressionModel.main()
