from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import resample
from pygam import s as GAM_spline
from pygam import f as GAM_factor
from multiprocessing import Pool
import matplotlib.pyplot as plt
from pygam import l as GAM_line
from pygam import LinearGAM
from sklearn import tree
from scipy import stats
import xgboost as xgb
import pandas as pd
import numpy as np

def read_data(path):
    with open(path, "r") as infile:
        data = infile.readlines()
        labels = np.array(data[0].split())[:-1]
        data = data[1:]
    for i in range(len(labels)):
        labels[i] = labels[i].strip()[1:-1]
    input_arr = []
    output_arr = []
    for line in data:
        line = line.split()
        input_arr.append(line[:-1])
        output_arr.append(line[-1].strip())
    input_arr = np.array(input_arr).astype(np.float64)
    output_arr = np.array(output_arr).astype(np.float64)
    if output_arr.ndim == 1:
        output_arr = output_arr[:,np.newaxis]
    return input_arr, output_arr, labels

def get_model(X, Y, options):
    """SPLITTING THE DATASET"""
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, **options)

    """PREPROCESSING"""
    # NB: No need for one-hot encoding – categorical columns are already binary!
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # scaler = StandardScaler()
    # Y_train = scaler.fit_transform(Y_train)
    # Y_test = scaler.transform(Y_test)

    """CREATING A DESIGN MATRIX"""
    poly = PolynomialFeatures(1)
    X_test = poly.fit_transform(X_test)
    X_train = poly.fit_transform(X_train)

    """PERFORMING LINEAR REGRESSION FIT"""
    model = LinearRegression().fit(X_train, Y_train)

    return model, X_train, X_test, Y_train, Y_test

def get_stats(model, X_test, Y_test, labels):

    """PREDICTING OUTPUTS USING LINEAR REGRESSION MODEL"""
    Y_predict = model.predict(X_test)

    """EXTRACTING COEFFICIENTS"""
    betas = model.coef_.squeeze()[1:]

    """CALCULATING STANDARD ERROR"""
    var_betas = np.diagonal(np.linalg.inv(X_test.T @ X_test))
    std_err = np.sqrt(var_betas)[1:]

    """CALCULATING P-VALUES"""
    t_values = betas/std_err
    p_values = 2*(1-stats.t.cdf(np.abs(t_values), X_test.shape[0]-X_test.shape[1]))

    """CALCULATING MSE"""
    MSE = np.mean((Y_predict - Y_test)**2)

    """CALCULATING R2-SCORE"""
    SS_tot = np.sum((Y_test-np.mean(Y_test))**2)
    SS_res = np.sum((Y_predict-Y_test)**2)
    R2 = 1 - SS_res/SS_tot

    """SORTING FEATURES BASED ON P-VALUES (low --> high)"""
    idx = np.argsort(p_values)

    if len(idx) > 1:
        labels = labels[idx]
        betas = betas[idx]
        std_err = std_err[idx]
        p_values = p_values[idx]

    return labels, betas, std_err, p_values, MSE, R2, idx

def sort_data(model, X_train, Y_train, X_test, Y_test, labels):

    """GETTING STATISTICAL DATA"""
    labels, betas, std_err, p_values, MSE, R2, idx = \
    get_stats(model, X_test, Y_test, labels)

    """COMPILING INFORMATION FOR PRINTING TO TERMINAL"""

    stat_vals = f"\t{'Label':^10s} {'Coeff':^10s} {'Std Err':^10s} {'P-Val':^10s} "
    longest_str = np.max(list(map(len, globals()["label_descr"].values())))+1
    stat_vals += f"{'Description':^{longest_str}s}\n"
    stat_vals += "\t" + "–"*len(stat_vals) + "\n"
    for i,j,k,l in zip(labels, betas, std_err, p_values):
        stat_vals += f"\t{i:>10s} {j:>10.2E} {k:>10.2E} {l:>10.2E}"
        stat_vals += f" {label_descr[i]:>{longest_str}s}\n"

    info = (f"\nDATA DIMENSIONS:\n\n\t"
            f"X_train: ({X_train.shape[0]}, {X_train.shape[1]})\t"
            f"X_test : ({X_test.shape[0]}, {X_test.shape[1]})\n\t"
            f"Y_train: ({Y_train.shape[0]}, {Y_train.shape[1]})\t"
            f"Y_test : ({Y_test.shape[0]}, {Y_test.shape[1]})\n\t"
            f"Train Percentage: {100*(1-test_percent):0.0f}%\t"
            f"Test Percentage: {100*test_percent:0.0f}%\n\n"
            f"PREDICTION INFO SORTED BY IMPORTANCE:\n\n{stat_vals}\n\n"
            f"STATISTICS:\n\n\tMSE:\t{MSE:.4E}\n\tR²:\t{R2:.4E}")

    return info, labels, betas, std_err, p_values, MSE, R2, idx

def backward_elimination(X, Y, labels, options):
    X_step = X.copy()
    labels_short = labels.copy()
    MSE_arr, R2_arr = np.zeros(p), np.zeros(p)
    labels_importance = []
    for i in range(p):
        model, X_train, X_test, Y_train, Y_test = get_model(X_step, Y, options)
        temp = get_stats(model, X_test, Y_test, labels)
        p_values, MSE, R2, idx = temp[-4:]
        MSE_arr[i] = MSE
        R2_arr[i] = R2
        X_step = X_step[:,idx[:-1]]
        labels_importance.insert(0, labels_short[idx[-1]])
        labels_short = labels_short[idx[:-1]]
    return MSE_arr, R2_arr, labels_importance

def forward_substitution(X, Y, labels, options):
    X = X.copy()
    MSE_arr, R2_arr = np.zeros(p), np.zeros(p)
    col_init = [i for i in range(p)]
    cols = col_init.copy()
    col_rank = []
    for i in range(p):
        p_values_step = []
        X_step = X[:,col_init]
        for j in range(p-i):
            X_temp = np.hstack([X[:,col_rank], X_step[:,j,None]])
            temp = get_model(X_temp, Y, options)
            temp = get_stats(temp[0], temp[2], temp[4], labels)
            p_values, MSE, R2, idx = temp[-4:]
            p_values_step.append(p_values[idx][-1])
        argmin = np.argmin(p_values_step)
        col_rank.append(col_init[argmin])
        del col_init[argmin]
    col_rank = np.array(col_rank)
    return MSE_arr, R2_arr, labels[col_rank]

def k_fold(data):
    X, Y, k, alpha = data
    """CREATING A DESIGN MATRIX"""
    poly = PolynomialFeatures(1)
    X_design = poly.fit_transform(X)

    """PERFORMING LASSO FIT"""
    model = Lasso(alpha = alpha, max_iter = 1E5)
    cv_results = cross_validate(model, X, Y, cv = k)
    scores = cv_results["test_score"]
    return np.mean(scores)

def bootstrap(data):
    X, Y, k, alpha, options = data

    """SPLITTING THE DATASET"""
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, **options)

    """PREPROCESSING"""
    # NB: No need for one-hot encoding – categorical columns are already binary!
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    scaler = StandardScaler()
    Y_train = scaler.fit_transform(Y_train)
    Y_test = scaler.transform(Y_test)

    """CREATING A DESIGN MATRIX"""
    poly = PolynomialFeatures(1)
    X_test = poly.fit_transform(X_test)
    X_train = poly.fit_transform(X_train)

    scores = []

    """PERFORMING LASSO FIT"""
    model = Lasso(alpha = alpha, max_iter = 1E5)
    for i in range(k):
        X_step, Y_step = resample(X_train, Y_train, random_state = rand_seed,
                                  n_samples = X_train.shape[0], replace = True)
        model.fit(X_step, Y_step)
        scores.append(model.score(X_test, Y_test))
    return np.mean(scores)

    print(scores)

def CV_compare(alphas):
    k_fold_scores = []
    bootstrap_scores = []

    pool = Pool()
    k_fold_args = [(X, Y, k, a) for a in alphas]
    bootstrap_args = [(X, Y, k, a, options) for a in alphas]

    k_fold_idx = []
    bootstrap_idx = []

    print(f"{0:4d}%", end = "")

    count = 0
    for n,i in enumerate(pool.imap(k_fold, k_fold_args)):
        count += 1
        k_fold_scores.append(i)
        k_fold_idx.append(n)
        print(f"\r{int(50*n/len(alphas)):4d}%", end = "")

    count = 0
    for n,i in enumerate(pool.imap(bootstrap, bootstrap_args)):
        count += 1
        bootstrap_scores.append(i)
        bootstrap_idx.append(n)
        print(f"\r{50+int(50*count/len(alphas)):4d}%", end = "")
    print(f"\r", end = "")

    bootstrap_scores = np.array(bootstrap_scores)[bootstrap_idx]
    k_fold_scores = np.array(k_fold_scores)[k_fold_idx]

    boot_idx = np.argmax(bootstrap_scores)
    k_fold_idx = np.argmax(k_fold_scores)

    plt.semilogx(alphas, k_fold_scores, label = "Cross-Validation")
    plt.semilogx(alphas, bootstrap_scores, label = "Bootstrap")
    plt.semilogx([alphas[k_fold_idx]], [k_fold_scores[k_fold_idx]], "bo",
                  label = f"KNN best $\\alpha$ = {alphas[k_fold_idx]:.4E}")
    plt.semilogx([alphas[boot_idx]], [bootstrap_scores[boot_idx]], "ro",
                  label = f"Bootstrap best $\\alpha$ = {alphas[boot_idx]:.4E}")
    plt.xlabel("Complexity Parameter $\\alpha$")
    plt.ylabel("R²-Score")
    plt.xlim([np.min(alphas), np.max(alphas)])
    plt.legend()
    plt.savefig("plot_1.pdf", dpi = 250)
    plt.close()

def GAM(X, Y, factor = False):

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

    linear = ['n', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'y', 'n', 'y',
    'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n']

    # for feature in X_train.T:
    #     unique = np.unique(feature)
    #     if len(unique) < 6:
    #         linear.append("n")
    #     else:
    #         idx = np.argsort(feature)
    #         plt.plot(feature[idx], Y.squeeze()[idx])
    #         plt.show()
    #         linear.append(input("Linear?\t"))

    linear = np.array(linear)
    linear[linear == "n"] = 0
    linear[linear == "y"] = 1
    linear = linear.astype(bool)

    gam_input = None
    for n,is_linear in enumerate(linear):
        if gam_input is not None:
            if is_linear:
                gam_input += GAM_line(n)
                if factor:
                    gam_input += GAM_factor(n)
            else:
                gam_input += GAM_spline(n)
        else:
            if is_linear:
                gam_input = GAM_line(n)
                if factor:
                    gam_input += GAM_factor(n)
            else:
                gam_input = GAM_spline(n)

    gam = LinearGAM(gam_input, fit_intercept = False, max_iter = int(1E5))
    gam.fit(X_train, Y_train)
    Y_predict = gam.predict(X_test)
    MSE = np.mean((Y_predict - Y_test)**2)
    return MSE

"""PARAMETERS"""
test_percent = 0.5      # Float in range (0,1)
rand_seed = 11235813
k = 5                   # "k" in k-fold cross validation

"""DESCRIBING LABELS"""
label_descr = { "ALTER": "Age", "ADHEU": "Allergic Coryza", "SEX": "Gender",
"HOCHOZON": "High Ozone Village", "AMATOP": "Maternal Atopy",
"AVATOP": "Paternal Atopy", "ADEKZ": "Neurodermatitis", "ARAUCH": "Smoker",
"AGEBGEW": "Birth Weight", "FSNIGHT": "Night/Morning Cough",
"FLGROSS": "Height", "FMILB": "Dust Sensitivity", "FNOH24": "Max. NO2",
"FTIER": "Fur Sensitivity", "FPOLL": "Pollen Sensitivity",
"FLTOTMED": "No. of Medis/Lufu", "FO3H24": "24h Max Ozone Value",
"FSPT": "Allergic Reaction", "FTEH24": "24h Max Temperature",
"FSATEM": "Shortness of Breath", "FSAUGE": "Itchy Eyes", "FLGEW": "Weight",
"FSPFEI": "Wheezy Breath", "FSHLAUF": "Cough"}

"""TEST TRAIN SPLIT OPTIONS"""
options = {"test_size":test_percent, "random_state":rand_seed}

"""IMPLEMENTING SOME PARAMETERS"""
np.random.seed(rand_seed)

"""READING DATA FROM FILE"""
data_path = "data.dat"
X, Y, labels = read_data(data_path)
p, q = X.shape[1], Y.shape[1]

"""DETERMINING FEATURE IMPORTANCE"""
model, X_train, X_test, Y_train, Y_test = get_model(X, Y, options)

"""DISPLAYING INITIAL RESULTS"""
info, labels_sort, betas_sort, std_err_sort, p_values_sort, MSE, R2, idx = \
sort_data(model, X_train, Y_train, X_test, Y_test, labels)
print(info)

"""BACKWARD ELIMINATION"""
MSE_back, R2_back, labels_back = backward_elimination(X, Y, labels, options)

"""FORWARD SUBSTITUTION"""
MSE_back, R2_back, labels_forward = forward_substitution(X, Y, labels, options)

"""COMPARING LAST TWO METHODS"""
print("\nFEATURE IMPORTANCE COMPARISON:\n")
msg1 = f"{'Rank':7s} {'Backward Elimination':30s} {'Forward Substitution':30s}"
msg2 = "–"*len(msg1)
print("\t" + msg1 + "\n\t" + msg2 + "\n")
for n,(i,j) in enumerate(zip(labels_back, labels_forward)):
    print(f"\t{n+1:<7d} {label_descr[i]:30s} {label_descr[j]:30s}")

"""K-FOLD CROSS VALIDATION AND BOOTSTRAP"""
alphas = np.logspace(-5, -0.5, 5000)
CV_compare(alphas)

"""IMPLEMENTING GENERALIZED ADDITIVE MODEL"""
MSE_GAM_lin = GAM(X, Y, factor = False)
MSE_GAM_fac = GAM(X, Y, factor = True)
msg = (f"        \nGENERALIZED ADDITIVE MODEL:\n\n\tLinear MSE:\t"
       f"{MSE_GAM_lin:.4E}\n\tPolynomial MSE:\t{MSE_GAM_fac:.4E}")
print(msg)

"""BOOSTING"""
LR = AdaBoostRegressor(base_estimator = LinearRegression())
LR.fit(X_train, Y_train.squeeze())
score = LR.score(X_test, Y_test.squeeze())
coefficients = LR.estimators_[-1].coef_
coef = ""
for c in coefficients:
    coef += f"{c:5.2f} "
coef = coef.strip()
print(f"\nLINEAR REGRESSION BOOSTING COEFFICIENTS:\n\n{coef}")

msg = (f"\nBOOSTING SCORES:\n\n\tLinear Regression:\t{score:.4E}\n\t")

DTR = AdaBoostRegressor(base_estimator = DecisionTreeRegressor())
DTR.fit(X_train, Y_train.squeeze())
score = DTR.score(X_test, Y_test.squeeze())
msg += (f"Decision Tree:\t\t{score:.4E}")
print(msg)

"""
$ python3 problem1.py

DATA DIMENSIONS:

	X_train: (248, 25)	X_test : (248, 25)
	Y_train: (248, 1)	Y_test : (248, 1)
	Train Percentage: 50%	Test Percentage: 50%

PREDICTION INFO SORTED BY IMPORTANCE:

	  Label      Coeff     Std Err     P-Val        Description
	––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
	   FLGROSS   1.83E-01   1.06E-01   8.53E-02               Height
	       SEX  -1.04E-01   6.63E-02   1.18E-01               Gender
	     FLGEW   7.47E-02   9.99E-02   4.55E-01               Weight
	    FO3H24   5.85E-02   1.33E-01   6.62E-01  24h Max Ozone Value
	    FTEH24  -5.23E-02   1.24E-01   6.73E-01  24h Max Temperature
	     FPOLL  -4.53E-02   1.33E-01   7.34E-01   Pollen Sensitivity
	  FLTOTMED  -1.53E-02   5.31E-02   7.74E-01    No. of Medis/Lufu
	   FSNIGHT   1.89E-02   6.73E-02   7.79E-01  Night/Morning Cough
	     FTIER  -1.97E-02   7.53E-02   7.94E-01      Fur Sensitivity
	    FNOH24  -2.26E-02   9.55E-02   8.13E-01             Max. NO2
	    FSPFEI   2.32E-02   9.88E-02   8.14E-01        Wheezy Breath
	   FSHLAUF  -1.32E-02   6.13E-02   8.30E-01                Cough
	   AGEBGEW   1.46E-02   6.84E-02   8.31E-01         Birth Weight
	    ARAUCH   1.50E-02   7.08E-02   8.33E-01               Smoker
	      FSPT   3.30E-02   1.58E-01   8.35E-01    Allergic Reaction
	     ADEKZ   1.31E-02   6.52E-02   8.41E-01      Neurodermatitis
	  HOCHOZON  -1.85E-02   9.46E-02   8.45E-01   High Ozone Village
	     FMILB  -1.54E-02   8.92E-02   8.63E-01     Dust Sensitivity
	    FSAUGE  -1.14E-02   7.33E-02   8.77E-01           Itchy Eyes
	     ALTER   9.84E-03   8.08E-02   9.03E-01                  Age
	    AMATOP   5.69E-03   7.20E-02   9.37E-01       Maternal Atopy
	     ADHEU  -5.44E-03   6.96E-02   9.38E-01      Allergic Coryza
	    FSATEM   3.70E-03   9.36E-02   9.69E-01  Shortness of Breath
	    AVATOP  -3.57E-04   7.04E-02   9.96E-01       Paternal Atopy


STATISTICS:

	MSE:	4.5847E-02
	R²:	6.1327E-01

FEATURE IMPORTANCE COMPARISON:

	Rank    Backward Elimination           Forward Substitution
	–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

	1       Height                         Height
	2       Gender                         Gender
	3       Weight                         Weight
	4       Fur Sensitivity                Fur Sensitivity
	5       24h Max Temperature            Pollen Sensitivity
	6       24h Max Ozone Value            Wheezy Breath
	7       Pollen Sensitivity             Smoker
	8       Night/Morning Cough            Birth Weight
	9       Smoker                         Max. NO2
	10      Birth Weight                   No. of Medis/Lufu
	11      Max. NO2                       Age
	12      High Ozone Village             Allergic Coryza
	13      No. of Medis/Lufu              Neurodermatitis
	14      Wheezy Breath                  Night/Morning Cough
	15      Cough                          Cough
	16      Neurodermatitis                Allergic Reaction
	17      Itchy Eyes                     24h Max Temperature
	18      Allergic Reaction              Dust Sensitivity
	19      Dust Sensitivity               Itchy Eyes
	20      Age                            24h Max Ozone Value
	21      Allergic Coryza                High Ozone Village
	22      Maternal Atopy                 Maternal Atopy
	23      Shortness of Breath            Shortness of Breath
	24      Paternal Atopy                 Paternal Atopy

GENERALIZED ADDITIVE MODEL:

	Linear MSE:	1.9198E-01
	Polynomial MSE:	1.9103E-01

LINEAR REGRESSION BOOSTING COEFFICIENTS:

0.00  0.01 -0.01 -0.13 -0.03  0.01 -0.06 -0.03  0.01  0.04  0.04  0.19 -0.09 -0.05 -0.10 -0.12  0.01 -0.03  0.16  0.00 -0.00  0.03  0.07  0.06 -0.04

BOOSTING SCORES:

	Linear Regression:	6.2508E-01
	Decision Tree:		5.5541E-01
"""
