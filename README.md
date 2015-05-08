# Automated-Essay-Grader
This repository contains all the code written for the final project of CSCI-GA.2590-001 Natural Language Processing 15Spring

Team members:
- Yucheng Lu
- Fangyun Sun
- Wenying Liu

Our project is based on a kaggle competition here:
https://www.kaggle.com/c/asap-aes

Run:
First, you need to install all the packages in the requirements.txt, and then, use python run.py to run the whole program. And the program will first generate all the necessary features and store them into FeatureData directory and then add these features into training data and test data. And the program will store the training data into TrainingData directory and store test data into TestData directory. The preocess will take about 4-5 hours. Then the program will train linear regression model, gradient boosting tree model and random forest regression model and store the results in the Result directory.
