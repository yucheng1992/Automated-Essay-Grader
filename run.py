from FeatureExtraction import FeatureExtraction
from FeatureExtraction import GenerateArticleFeatures
from FeatureExtraction import CombineFeatures
from Model import GradientBoostingRegressionTreeModel
from Model import RandomForestRegressionModel
from Util import util
from Util import KappaScores
from Util import SelectValidationScores


if __name__ == '__main__':
    FeatureExtraction.main()
    CombineFeatures.main()
    GradientBoostingRegressionTreeModel.main()
    RandomForestRegressionModel.main()
