from sklearn.model_selection import train_test_split, cross_validate, LeaveOneOut
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier as KNN
from pygam import s as GAM_spline
from multiprocessing import Pool
import matplotlib.pyplot as plt
from pygam import LinearGAM
import pandas as pd
import numpy as np

def read_data(path):
    df = pd.read_csv(path)
    arr = np.array(df)
    X = arr[:,:-1]
    Y = arr[:,-1]
    del df
    labels = ["pregnant", "glucose", "pressure", "triceps", "insulin", "mass",
              "pedigree", "age"]
    return X, Y, np.array(labels)

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

def matrix_info(X_train, Y_train, X_test, Y_test):

    """COMPILING INFORMATION FOR PRINTING TO TERMINAL"""

    info = (f"\nDATA DIMENSIONS:\n\n\t"
            f"X_train: ({X_train.shape[0]}, {X_train.shape[1]})\t"
            f"X_test : ({X_test.shape[0]}, {X_test.shape[1]})\n\t"
            f"Y_train: ({Y_train.shape[0]}, 1)\t"
            f"Y_test : ({Y_test.shape[0]}, 1)\n\t"
            f"Train Percentage: {100*(1-test_percent):0.0f}%\t"
            f"Test Percentage: {100*test_percent:0.0f}%")

    return info

def k_fold(data):
    X, Y, k, n_neighbors = data

    """PERFORMING LASSO FIT"""
    model = KNN(n_neighbors = n_neighbors)
    cv_results = cross_validate(model, X, Y, cv = k)
    scores = cv_results["test_score"]
    return np.mean(scores)

def loo_cv(data):
    X, Y, n_neighbors = data

    """PERFORMING LASSO FIT"""
    model = KNN(n_neighbors = n_neighbors)
    cv_gen = LeaveOneOut().split(X, Y)
    cv_results = cross_validate(model, X, Y, cv = cv_gen)
    scores = cv_results["test_score"]
    return np.mean(scores)

def CV_compare(X_train, Y_train, X_test, Y_test):
    k_neighbors = np.arange(1, 25, 1)
    k_fold_scores = []
    loo_scores = []

    pool = Pool()
    k_fold_args = [(X, Y, 5, n) for n in k_neighbors]
    loo_args = [(X, Y, n) for n in k_neighbors]

    k_fold_idx = []
    loo_idx = []

    print(f"{0:4d}%", end = "")

    count = 0
    for n,i in enumerate(pool.imap(k_fold, k_fold_args)):
        count += 1
        k_fold_scores.append(i)
        k_fold_idx.append(n)
        print(f"\r{int(50*n/len(k_neighbors)):4d}%", end = "")

    count = 0
    for n,i in enumerate(pool.imap(loo_cv, loo_args)):
        count += 1
        loo_scores.append(i)
        loo_idx.append(n)
        print(f"\r{50+int(50*count/len(k_neighbors)):4d}%", end = "")
    print(f"\r", end = "")

    loo_scores = np.array(loo_scores)[loo_idx]
    k_fold_scores = np.array(k_fold_scores)[k_fold_idx]

    plt.semilogx(k_neighbors, k_fold_scores, label = "Cross-Validation")
    plt.semilogx(k_neighbors, loo_scores, label = "LOO")
    plt.xlabel("Number of Neighbors $k$")
    plt.ylabel("Accuracy-Score")
    plt.xlim([np.min(k_neighbors), np.max(k_neighbors)])
    plt.legend()
    plt.show()

def GAM(X, Y):

    """SPLITTING THE DATASET"""
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, **options)

    """PREPROCESSING"""
    # NB: No need for one-hot encoding – categorical columns are already binary!
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    """CREATING A DESIGN MATRIX"""
    poly = PolynomialFeatures(1)
    X_test = poly.fit_transform(X_test)
    X_train = poly.fit_transform(X_train)

    gam_input = None
    for n in range(X_train.shape[1]):
        if gam_input is not None:
            gam_input += GAM_spline(n)
        else:
            gam_input = GAM_spline(n)

    gam = LinearGAM(gam_input).fit(X_train, Y_train)
    Y_predict = gam.predict(X_test)
    Y_predict[Y_predict >= 0.5] = 1
    Y_predict[Y_predict < 0.5] = 0
    accuracy = (Y_predict.squeeze() == Y_test.squeeze()).astype(int)
    accuracy = np.sum(accuracy)/accuracy.shape[0]
    return accuracy

def forward_substitution(X, Y, labels, options):
    X = X.copy()
    MSE_arr, R2_arr = np.zeros(p), np.zeros(p)
    col_init = [i for i in range(p)]
    cols = col_init.copy()
    col_rank = []
    accuracies = []
    for i in range(p):
        accuracy_step = []
        X_step = X[:,col_init]
        for j in range(p-i):
            X_temp = np.hstack([X[:,col_rank], X_step[:,j,None]])
            accuracy = GAM(X_temp, Y)
            accuracy_step.append(accuracy)
        argmax = np.argmax(accuracy_step)
        accuracies.append(accuracy_step[argmax])
        col_rank.append(col_init[argmax])
        del col_init[argmax]
    col_rank = np.array(col_rank)
    return accuracies, labels[col_rank], col_rank

"""PARAMETERS"""
test_percent = 0.5  # Float in range (0,1)
rand_seed = 12345
k_neighbors = 100

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

"""CREATING BASIC MODEL"""
model, X_train, X_test, Y_train, Y_test = get_model(X, Y, k_neighbors, options)

"""DISPLAYING INITIAL RESULTS"""
info = matrix_info(X_train, Y_train, X_test, Y_test)
print(info)

"""K-FOLD CROSS VALIDATION AND BOOTSTRAP"""
# CV_compare(X_train, Y_train, X_test, Y_test)

"""COMPARING GENERALIZED ADDITIVE MODELS WITH FORWARD SUBSTITUTION"""
accuracies, labels_forward, idx = forward_substitution(X, Y, labels, options)
print("\nFEATURE IMPORTANCE (GENERALIZED ADDITIVE MODEL):\n")
msg1 = f"{'Rank':^7s} {'Forward Substitution':<28s} {'Cumulative Accuracy':^28s}"
msg2 = "–"*len(msg1)
print("\t" + msg1 + "\n\t" + msg2 + "\n")
for n,(i,j) in enumerate(zip(labels_forward, accuracies)):
    print(f"\t{n+1:^7d} {label_descr[i]:<28s} {j:^28.4E}")
