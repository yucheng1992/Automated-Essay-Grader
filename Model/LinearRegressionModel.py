import pickle 
import numpy 
import pandas as pd
from sklearn import linear_model
import csv
from Util.KappaScores import quadratic_weighted_kappa,mean_quadratic_weighted_kappa

def generate_train(filename):
    """
    Read the train dataset and generate the X_train and y_train.
    @return X(features) training data and y(score) training data.
    """
    train = pd.read_csv(filename)
    X_train = train.drop(["Unnamed: 0","score"],1)
    y_train = train['score']
    return X_train,y_train

def generate_test(filename_X, filename_y,essayNum):
    """
    Read the test dataset and generate the X_test and y_test.
    @return X(features) testing data and y(score) testing data.
    """
    test = pd.read_csv(filename_X)
    if essayNum != 2:
        X_test = test.drop("Unnamed: 0",1)
    else:
        X_test = test.drop(["Unnamed: 0","NONE"],1)
    pfile = open(filename_y,'rb')
    test_score = pickle.load(pfile)
    pfile.close()
    y_test = test_score[i-1]
    return X_test,y_test

def linear_predict(X_train,y_train,X_test,y_test,essayNum):
    """
    Train the linear regresion model using the training data and predict 
    the score for test dataset.
    @para    essayNum: The number of the essay set.
    @return  kappaScore: The kappa score between the true value and prediction.
    """

    clf = linear_model.LinearRegression()
    clf.fit(X_train,y_train)
    predictDomainOneScores = clf.predict(X_test)
    predictDomainOneScores = map(lambda x: int(round(x)),predictDomainOneScores)

    #Notice: Only for essay set 2, there are two domain scores and we have to train the model twice.
    if essayNum != 2:
        kappaScore = quadratic_weighted_kappa(y_test,predictDomainOneScores)
    else:
        kappaDomainOne = quadratic_weighted_kappa(y_test[0::2],predictDomainOneScores)        
        data = pd.DataFrame.from_csv("training_set_rel3.tsv",sep="\t")
        domainTwoScore = data[data["essay_set"] == 2]["domain2_score"]

        clf = linear_model.LinearRegression()
        clf.fit(X_train,domainTwoScore)
        predictDomainTwoScores = clf.predict(X_test)
        predictDomainTwoScores = map(lambda x: int(round(x)),predictDomainTwoScores)

        kappaDomainTwo = quadratic_weighted_kappa(y_test[1::2],predictDomainTwoScores)
        kappaScore = (kappaDomainOne + kappaDomainTwo)/2.0
        
    return kappaScore

if __name__ == '__main__':
    
    res_list = []
    for i in range(1,9):
        X_train,y_train = generate_train('essaySet' + str(i) + '.csv')
        X_test,y_test = generate_test('validessaySet'+str(i)+'.csv','validationScores.pkl',i)
        kappa_score = linear_predict(X_train,y_train,X_test,y_test,i)
        res_list.append(kappa_score)
    
    mean_kappa = mean_quadratic_weighted_kappa(res_list)
    res_list.append(mean_kappa)

    #Write the result into a file called "LinearRegressionKappaScores.txt"
    writer = csv.writer(open('../Result/LinearRegressionKappaScores.txt','wb'))
    for item in res_list:
        writer.writerow([item])

    
