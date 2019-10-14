import pandas as pd
import os
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

# Trying to set the seed
np.random.seed(0)
import random
random.seed(0)

# Reading file into data frame
cwd = os.getcwd()
filename = cwd + '/default of credit card clients.xls'
nanDict = {}
df = pd.read_excel(filename, header=1, skiprows=0, index_col=0, na_values=nanDict)

df.rename(index=str, columns={"default payment next month": "defaultPaymentNextMonth"}, inplace=True)

# Features and targets
X = df.loc[:, df.columns != 'defaultPaymentNextMonth'].values
y = df.loc[:, df.columns == 'defaultPaymentNextMonth'].values

# Categorical variables to one-hot's
onehotencoder = OneHotEncoder(categories="auto")

X = ColumnTransformer(
    [("", onehotencoder, [3]),],
    remainder="passthrough"
).fit_transform(X)

y.shape

# Train-test split
trainingShare = 0.5
seed  = 1
XTrain, XTest, yTrain, yTest=train_test_split(X, y, train_size=trainingShare, \
                                              test_size = 1-trainingShare,
                                             random_state=seed)

# Input Scaling
sc = StandardScaler()
XTrain = sc.fit_transform(XTrain)
XTest = sc.transform(XTest)

# One-hot's of the target vector
Y_train_onehot, Y_test_onehot = onehotencoder.fit_transform(yTrain), onehotencoder.fit_transform(yTest)

# Remove instances with zeros only for past bill statements or paid amounts
'''
df = df.drop(df[(df.BILL_AMT1 == 0) &
                (df.BILL_AMT2 == 0) &
                (df.BILL_AMT3 == 0) &
                (df.BILL_AMT4 == 0) &
                (df.BILL_AMT5 == 0) &
                (df.BILL_AMT6 == 0) &
                (df.PAY_AMT1 == 0) &
                (df.PAY_AMT2 == 0) &
                (df.PAY_AMT3 == 0) &
                (df.PAY_AMT4 == 0) &
                (df.PAY_AMT5 == 0) &
                (df.PAY_AMT6 == 0)].index)
'''
df = df.drop(df[(df.BILL_AMT1 == 0) &
                (df.BILL_AMT2 == 0) &
                (df.BILL_AMT3 == 0) &
                (df.BILL_AMT4 == 0) &
                (df.BILL_AMT5 == 0) &
                (df.BILL_AMT6 == 0)].index)

df = df.drop(df[(df.PAY_AMT1 == 0) &
                (df.PAY_AMT2 == 0) &
                (df.PAY_AMT3 == 0) &
                (df.PAY_AMT4 == 0) &
                (df.PAY_AMT5 == 0) &
                (df.PAY_AMT6 == 0)].index)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

lambdas=np.logspace(-5,7,13)
parameters = [{'C': 1./lambdas, "solver":["lbfgs"]}]#*len(parameters)}]
scoring = ['accuracy', 'roc_auc']
logReg = LogisticRegression()
gridSearch = GridSearchCV(logReg, parameters, cv=5, scoring=scoring, refit='roc_auc')

# "refit" gives the metric used deciding best model.
# See more http://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html
gridSearch.fit(XTrain, yTrain.ravel())

def gridSearchSummary(method, scoring):
    """Prints best parameters from Grid search
    and AUC with standard deviation for all
    parameter combos """

    method = eval(method)
    if scoring == 'accuracy':
        mean = 'mean_test_score'
        sd = 'std_test_score'
    elif scoring == 'auc':
        mean = 'mean_test_roc_auc'
        sd = 'std_test_roc_auc'
    print("Best: %f using %s" % (method.best_score_, method.best_params_))
    means = method.cv_results_[mean]
    stds = method.cv_results_[sd]
    params = method.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

def createConfusionMatrix(method, printOut=True):
    """
    Computes and prints confusion matrices, accuracy scores,
    and AUC for test and training sets
    """
    confusionArray = np.zeros(6, dtype=object)
    method = eval(method)

    # Train
    yPredTrain = method.predict(XTrain)
    yPredTrain = (yPredTrain > 0.5)
    cm = confusion_matrix(
        yTrain, yPredTrain)
    cm = np.around(cm/cm.sum(axis=1)[:,None], 2)
    confusionArray[0] = cm

    accScore = accuracy_score(yTrain, yPredTrain)
    confusionArray[1] = accScore

    AUC = roc_auc_score(yTrain, yPredTrain)
    confusionArray[2] = AUC

    if printOut:
        print('\n###################  Training  ###############')
        print('\nTraining Confusion matrix: \n', cm)
        print('\nTraining Accuracy score: \n', accScore)
        print('\nTrain AUC: \n', AUC)

    # Test
    yPred = method.predict(XTest)
    yPred = (yPred > 0.5)
    cm = confusion_matrix(
        yTest, yPred)
    cm = np.around(cm/cm.sum(axis=1)[:,None], 2)
    confusionArray[3] = cm

    accScore = accuracy_score(yTest, yPred)
    confusionArray[4] = accScore

    AUC = roc_auc_score(yTest, yPred)
    confusionArray[5] = AUC

    if printOut:
        print('\n###################  Testing  ###############')
        print('\nTest Confusion matrix: \n', cm)
        print('\nTest Accuracy score: \n', accScore)
        print('\nTestAUC: \n', AUC)

    return confusionArray


import matplotlib.pyplot as plt
import seaborn
# import scikitplot as skplt

seaborn.set(style="white", context="notebook", font_scale=1.5,
            rc={"axes.grid": True, "legend.frameon": False,
"lines.markeredgewidth": 1.4, "lines.markersize": 10})
seaborn.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 4.5})

yPred = gridSearch.predict_proba(XTest)
print(yTest.ravel().shape, yPred.shape)

#skplt.metrics.plot_cumulative_gain(yTest.ravel(), yPred_onehot)
# skplt.metrics.plot_cumulative_gain(yTest.ravel(), yPred)

defaults = sum(yTest == 1)
total = len(yTest)
defaultRate = defaults/total
def bestCurve(defaults, total, defaultRate):
    x = np.linspace(0, 1, total)

    y1 = np.linspace(0, 1, defaults)
    y2 = np.ones(total-defaults)
    y3 = np.concatenate([y1,y2])
    return x, y3

x, best = bestCurve(defaults=defaults, total=total, defaultRate=defaultRate)
plt.plot(x, best)


plt.show()
