# Automated-Essay-Grader
This repository contains all the code written for the final project of CSCI-GA.2590-001 Natural Language Processing 15Spring

Team members:
- Yucheng Lu
- Fangyun Sun
- Wenying Liu

Our project is based on a kaggle competition here:
https://www.kaggle.com/c/asap-aes

Run:
- First, you need to install all the packages in the requirements.txt.
- Then, use python run.py to run the whole program. And the program will first generate all the necessary features and store them into FeatureData directory and then add these features into training data and test data. And the program will store the training data into TrainingData directory and store test data into TestData directory. The preocess will take about 4-5 hours. 
- Finally, the program will train linear regression model, gradient boosting tree model and random forest regression model and store the results in the Result directory.

Reminder:
- For the linear regression does not perform well in a high-dimensional data set, so we just use the basic features of the articles and the part of speech features. So we store the generated data in advance in the LinearRegressionData directory.
