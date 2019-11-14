from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def read_data(path):
    df = pd.read_csv(path)
    arr = np.array(df)
    X_labels = df.columns.values[:-1]
    Y_label = df.columns.values[-1]
    X = arr[:,:-1]
    Y = arr[:,-1]
    del df
    return X, Y, X_labels

def get_model(X, Y, n_neighbors, options):
    """SPLITTING THE DATASET"""
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, **options)

    """PREPROCESSING"""
    # NB: No need for one-hot encoding – categorical columns are already binary!
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train, Y_train)
    X_test = scaler.transform(X_test)

    """PERFORMING LINEAR REGRESSION FIT"""
    model = KNN(n_neighbors = n_neighbors).fit(X_train, Y_train)

    return model, X_train, X_test, Y_train, Y_test

def get_stats(model, X_test, Y_test):

    """PREDICTING OUTPUTS USING LINEAR REGRESSION MODEL"""
    Y_predict = model.predict(X_test)
    print(Y_predict)

    """CALCULATING MSE"""
    MSE = np.mean((Y_predict - Y_test)**2)

    """CALCULATING R2-SCORE"""
    SS_tot = np.sum((Y_test-np.mean(Y_test))**2)
    SS_res = np.sum((Y_predict-Y_test)**2)
    R2 = 1 - SS_res/SS_tot

    return MSE, R2

def sort_data(model, X_train, Y_train, X_test, Y_test):

    """GETTING STATISTICAL DATA"""
    MSE, R2 = get_stats(model, X_test, Y_test)

    """COMPILING INFORMATION FOR PRINTING TO TERMINAL"""

    info = (f"\nDATA DIMENSIONS:\n\n\t"
            f"X_train: ({X_train.shape[0]}, {X_train.shape[1]})\t"
            f"X_test : ({X_test.shape[0]}, {X_test.shape[1]})\n\t"
            f"Y_train: ({Y_train.shape[0]}, 1)\t"
            f"Y_test : ({Y_test.shape[0]}, 1)\n\t"
            f"Train Percentage: {100*(1-test_percent):0.0f}%\t"
            f"Test Percentage: {100*test_percent:0.0f}%\n\n"
            f"STATISTICS:\n\n\tMSE:\t{MSE:.4E}\n\tR²:\t{R2:.4E}")

    return info, MSE, R2

"""PARAMETERS"""
test_percent = 0.5  # Float in range (0,1)
rand_seed = None#11235813
n_neighbors = 12

"""DESCRIBING LABELS"""
label_descr = {"pregnant":"Number of Pregnancies",
"glucose":"Plasma Glucose Concentration",
"pressure":"Diastolic Blood Pressure", "triceps":"Triceps Skin Fold Thickness",
"insulin":"2-H Serum Insulin", "mass":"Body Mass Index",
"pedigree":"Diabetes Pedigree Function", "age":"Age"}

"""TEST TRAIN SPLIT OPTIONS"""
options = {"test_size":test_percent, "random_state":rand_seed}

"""IMPLEMENTING SOME PARAMETERS"""
np.random.seed(rand_seed)

"""READING DATA FROM FILE"""
data_path = "pima-indians-diabetes.csv"
X, Y, labels = read_data(data_path)
p = X.shape[1]

"""CREATING MODEL"""
model, X_train, X_test, Y_train, Y_test = get_model(X, Y, n_neighbors, options)
# DISPLAYING INITIAL RESULTS
info, MSE, R2 = sort_data(model, X_train, Y_train, X_test, Y_test)
print(info)
