from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.utils import resample
from pygam import s as GAM_spline
from pygam import f as GAM_factor
from multiprocessing import Pool
import matplotlib.pyplot as plt
from pygam import l as GAM_line
from pygam import LinearGAM
from scipy import stats
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

    scaler = MinMaxScaler()
    Y_train = scaler.fit_transform(Y_train)
    Y_test = scaler.transform(Y_test)

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

def print_data(mode, stop, MSE, labels, betas, std_err, p_values):
    stat_vals = f"\t{'Label':^10s} {'Coeff':^10s} {'Std Err':^10s} {'P-Val':^10s} "
    longest_str = np.max(list(map(len, globals()["label_descr"].values())))+1
    stat_vals += f"{'Description':^{longest_str}s}\n"
    stat_vals += "\t" + "–"*len(stat_vals) + "\n"
    for i,j,k,l in zip(labels, betas, std_err, p_values):
        stat_vals += f"\t{i:>10s} {j:>10.2E} {k:>10.2E} {l:>10.2E}"
        stat_vals += f" {label_descr[i]:>{longest_str}s}\n"

    string = (f"\nPREDICTION INFO SORTED BY IMPORTANCE:\n\n{stat_vals}\n\n"
              f"STATISTICS:\n\n\tMSE:\t\t\t{MSE:.4E}\n\tStopping Criterion:\t"
              f"p > {stop}")
    print(string)

def print_data_tex(mode, stop, MSE, labels, betas, std_err, p_values):
    stat_vals = f"\t{'Label':^10s} {'Coeff':^10s} {'Std Err':^10s} {'P-Val':^10s} "
    longest_str = np.max(list(map(len, globals()["label_descr"].values())))+1
    stat_vals += f"{'Description':^{longest_str}s}\n"
    stat_vals += "\t" + "–"*len(stat_vals) + "\n"
    for i,j,k,l in zip(labels, betas, std_err, p_values):
        stat_vals += f"\t{i:>10s} {j:>10.2E} {k:>10.2E} {l:>10.2E}"
        stat_vals += f" {label_descr[i]:>{longest_str}s}\n"

    string = (f"\nPREDICTION INFO SORTED BY IMPORTANCE:\n\n{stat_vals}\n\n"
              f"STATISTICS:\n\n\tMSE:\t\t\t{MSE:.4E}\n\tStopping Criterion:\t"
              f"p > {stop}")

    string = ("\\begin{table}[H]\n\\center\n\t\\begin{tabular}{l l l l}\n\t"
              "\\textbf{Feature} & \\textbf{Coefficient}& \\textbf{Standard "
              "Error} & \\textbf{P-Value} \\\\ \n\t"
              "\\hline\n")
    for n,(i,j,k,l) in enumerate(zip(labels,betas,std_err,p_values)):
        string += f"\t{i} & {j:.2f} & {k:.2f} & {l:.2f}"
        if n < len(labels) - 1:
            string += "\\\\"
        string += "\n"
    string += ("\t\\end{tabular}\n\\caption{Remaining features after "
               f"{mode} with stopping criterion $p \\geq {stop}$. "
               f"$MSE = {MSE:.4E}$"
               "\\label{table_N}}\n\\end{table}")

    print(string)

def backward_elimination(X, Y, labels, options, stop = 0.5):
    X_step = X.copy()
    labels_short = labels.copy()
    labels_importance = []
    count = 0
    for i in range(p):
        model, X_train, X_test, Y_train, Y_test = get_model(X_step, Y, options)
        temp = get_stats(model, X_test, Y_test, labels)
        if temp[-4][i] < stop:
            betas, std_err, p_values, MSE, R2, idx = temp[1:]
            X_step = X_step[:,idx[:-1]]
            labels_importance.insert(0, labels_short[idx[-1]])
            labels_short = labels_short[idx[:-1]]
        else:
            betas_arr = betas[:i]
            std_err_arr = std_err[:i]
            p_arr = np.array(p_values[:i])
            labels_importance = labels_short[:i]
            break
        if i == p-1:
            p_arr = np.array(p_values)
            betas_arr = np.array(betas)
            std_err_arr = np.array(std_err)

    return MSE, betas_arr, std_err_arr, p_arr, labels_importance

def forward_substitution(X, Y, labels, options, stop = 0.5):
    X = X.copy()
    col_init = [i for i in range(p)]
    cols = col_init.copy()
    p_arr, betas_arr, std_err_arr, col_rank = [],[],[],[]
    for i in range(p):
        p_values_step = []
        betas_step = []
        std_err_step = []
        X_step = X[:,col_init]
        for j in range(p-i):
            X_temp = np.hstack([X[:,col_rank], X_step[:,j,None]])
            temp = get_model(X_temp, Y, options)
            temp = get_stats(temp[0], temp[2], temp[4], labels)
            betas, std_err, p_values, MSE, R2, idx = temp[1:]
            p_values_step.append(p_values[idx][-1])
            betas_step.append(betas[idx][-1])
            std_err_step.append(std_err[idx][-1])
        argmin = np.argmin(p_values_step)
        if p_values_step[argmin] < stop:
            p_arr.append(p_values_step[argmin])
            betas_arr.append(betas_step[argmin])
            std_err_arr.append(std_err_step[argmin])
            col_rank.append(col_init[argmin])
            del col_init[argmin]
        else:
            break
    col_rank = np.array(col_rank)
    p_arr = np.array(p_values)
    betas_arr = np.array(betas)
    std_err_arr = np.array(std_err)
    return MSE, betas_arr, std_err_arr, p_arr, labels[col_rank]

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
    label = f"KNN best $\\alpha$ = {alphas[k_fold_idx]:.4E}, MSE = {k_fold_scores[k_fold_idx]:.4E}")
    plt.semilogx([alphas[boot_idx]], [bootstrap_scores[boot_idx]], "ro",
    label = f"Bootstrap best $\\alpha$ = {alphas[boot_idx]:.4E}, MSE = {bootstrap_scores[boot_idx]:.4E}")
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
stop_b = 0.7
mode_b = "backward elimination"
MSE_b, betas_b, std_err_b, p_b, labels_back = \
backward_elimination(X.copy(), Y.copy(), labels, options, stop_b)

print("\nBACKWARD ELIMINATION")
print_data(mode_b, stop_b, MSE_b, labels_back, betas_b, std_err_b, p_b)

"""FORWARD SUBSTITUTION"""
stop_f = 0.7
mode_f = "forward substitution"
MSE_f, betas_f, std_err_f, p_f, labels_forward = \
forward_substitution(X.copy(), Y.copy(), labels, options, stop_f)

print("\nFORWARD SUBSTITUTION")
print_data(mode_f, stop_f, MSE_f, labels_forward, betas_f, std_err_f, p_f)

"""BACKWARD ELIMINATION"""
stop_b = 0.8
mode_b = "backward elimination"
MSE_b, betas_b, std_err_b, p_b, labels_back = \
backward_elimination(X.copy(), Y.copy(), labels, options, stop_b)

print("\nBACKWARD ELIMINATION")
print_data(mode_b, stop_b, MSE_b, labels_back, betas_b, std_err_b, p_b)

"""FORWARD SUBSTITUTION"""
stop_f = 0.8
mode_f = "forward substitution"
MSE_f, betas_f, std_err_f, p_f, labels_forward = \
forward_substitution(X.copy(), Y.copy(), labels, options, stop_f)

print("\nFORWARD SUBSTITUTION")
print_data(mode_f, stop_f, MSE_f, labels_forward, betas_f, std_err_f, p_f)

"""COMPARING LAST TWO METHODS"""
print("\nFEATURE IMPORTANCE COMPARISON:\n")
msg1 = f"{'Rank':7s} {'Backward Elimination':30s} {'Forward Substitution':30s}"
msg2 = "–"*len(msg1)
print("\t" + msg1 + "\n\t" + msg2 + "\n")
for n,(i,j) in enumerate(zip(labels_back, labels_forward)):
    print(f"\t{n+1:<7d} {label_descr[i]:30s} {label_descr[j]:30s}")

"""K-FOLD CROSS VALIDATION AND BOOTSTRAP"""
alphas = np.logspace(-5, -0.5, 1000)
CV_compare(alphas)

"""IMPLEMENTING GENERALIZED ADDITIVE MODEL"""
MSE_GAM_lin = GAM(X.copy(), Y.copy(), factor = False)
MSE_GAM_fac = GAM(X.copy(), Y.copy(), factor = True)
msg = (f"        \nGENERALIZED ADDITIVE MODEL:\n\n\tLinear MSE:\t"
       f"{MSE_GAM_lin:.4E}\n\tPolynomial MSE:\t{MSE_GAM_fac:.4E}")
print(msg)

"""BOOSTING"""
LR = AdaBoostRegressor(base_estimator = LinearRegression())
LR.fit(X_train, Y_train.squeeze())
Y_predict = LR.predict(X_test)
score = np.mean((Y_predict-Y_test.squeeze())**2)
coefficients = LR.estimators_[-1].coef_
coef = ""
for c,l in zip(coefficients, labels):
    coef += f"{c:16.4E}"
coef = coef.strip()
print(f"\nLINEAR REGRESSION BOOSTING COEFFICIENTS:\n\n{coef}")

msg = (f"\nBOOSTING SCORES:\n\n\tLinear Regression:\t{score:.4E}\n\t")

DTR = AdaBoostRegressor(base_estimator = DecisionTreeRegressor(), n_estimators=500)
DTR.fit(X_train, Y_train.squeeze())
Y_predict = DTR.predict(X_test)
score = np.mean((Y_predict-Y_test.squeeze())**2)
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
	   FLGROSS   8.47E-02   1.06E-01   4.25E-01               Height
	       SEX  -4.81E-02   6.63E-02   4.69E-01               Gender
	     FLGEW   3.46E-02   9.99E-02   7.29E-01               Weight
	    FO3H24   2.71E-02   1.33E-01   8.39E-01  24h Max Ozone Value
	    FTEH24  -2.42E-02   1.24E-01   8.45E-01  24h Max Temperature
	     FPOLL  -2.10E-02   1.33E-01   8.75E-01   Pollen Sensitivity
	  FLTOTMED  -7.07E-03   5.31E-02   8.94E-01    No. of Medis/Lufu
	   FSNIGHT   8.76E-03   6.73E-02   8.97E-01  Night/Morning Cough
	     FTIER  -9.12E-03   7.53E-02   9.04E-01      Fur Sensitivity
	    FNOH24  -1.05E-02   9.55E-02   9.13E-01             Max. NO2
	    FSPFEI   1.08E-02   9.88E-02   9.13E-01        Wheezy Breath
	   FSHLAUF  -6.11E-03   6.13E-02   9.21E-01                Cough
	   AGEBGEW   6.75E-03   6.84E-02   9.22E-01         Birth Weight
	    ARAUCH   6.94E-03   7.08E-02   9.22E-01               Smoker
	      FSPT   1.53E-02   1.58E-01   9.23E-01    Allergic Reaction
	     ADEKZ   6.07E-03   6.52E-02   9.26E-01      Neurodermatitis
	  HOCHOZON  -8.56E-03   9.46E-02   9.28E-01   High Ozone Village
	     FMILB  -7.13E-03   8.92E-02   9.36E-01     Dust Sensitivity
	    FSAUGE  -5.27E-03   7.33E-02   9.43E-01           Itchy Eyes
	     ALTER   4.56E-03   8.08E-02   9.55E-01                  Age
	    AMATOP   2.64E-03   7.20E-02   9.71E-01       Maternal Atopy
	     ADHEU  -2.52E-03   6.96E-02   9.71E-01      Allergic Coryza
	    FSATEM   1.71E-03   9.36E-02   9.85E-01  Shortness of Breath
	    AVATOP  -1.65E-04   7.04E-02   9.98E-01       Paternal Atopy


STATISTICS:

	MSE:	9.8266E-03
	R²:	6.1327E-01

BACKWARD ELIMINATION

PREDICTION INFO SORTED BY IMPORTANCE:

	  Label      Coeff     Std Err     P-Val        Description
	––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
	   FLGROSS   8.47E-02   1.06E-01   4.23E-01               Height
	       SEX  -4.82E-02   6.61E-02   4.67E-01               Gender


STATISTICS:

	MSE:			9.8286E-03
	Stopping Criterion:	p > 0.7

FORWARD SUBSTITUTION

PREDICTION INFO SORTED BY IMPORTANCE:

	  Label      Coeff     Std Err     P-Val        Description
	––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
	   FLGROSS   8.49E-02   9.23E-02   3.58E-01               Height
	       SEX  -4.65E-02   6.35E-02   4.65E-01               Gender
	     FLGEW   4.16E-02   9.13E-02   6.49E-01               Weight


STATISTICS:

	MSE:			9.7452E-03
	Stopping Criterion:	p > 0.7

BACKWARD ELIMINATION

PREDICTION INFO SORTED BY IMPORTANCE:

	  Label      Coeff     Std Err     P-Val        Description
	––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
	   FLGROSS   8.48E-02   1.06E-01   4.23E-01               Height
	       SEX  -4.81E-02   6.61E-02   4.68E-01               Gender
	     FLGEW   3.46E-02   9.97E-02   7.29E-01               Weight


STATISTICS:

	MSE:			9.8754E-03
	Stopping Criterion:	p > 0.8

FORWARD SUBSTITUTION

PREDICTION INFO SORTED BY IMPORTANCE:

	  Label      Coeff     Std Err     P-Val        Description
	––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
	   FLGROSS   8.50E-02   9.24E-02   3.59E-01               Height
	       SEX  -4.84E-02   6.37E-02   4.48E-01               Gender
	     FLGEW   4.15E-02   9.13E-02   6.50E-01               Weight
	     FTIER  -1.61E-02   5.84E-02   7.83E-01      Fur Sensitivity


STATISTICS:

	MSE:			1.0055E-02
	Stopping Criterion:	p > 0.8

FEATURE IMPORTANCE COMPARISON:

	Rank    Backward Elimination           Forward Substitution
	–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

	1       Height                         Height
	2       Gender                         Gender
	3       Weight                         Weight

GENERALIZED ADDITIVE MODEL:

	Linear MSE:	1.9198E-01
	Polynomial MSE:	1.9103E-01

LINEAR REGRESSION BOOSTING COEFFICIENTS:

0.0000E+00      5.9147E-03     -4.9335E-03     -5.8675E-02     -1.2691E-02      6.2121E-03     -2.7185E-02     -1.3062E-02      5.7216E-03      1.9285E-02      1.9060E-02      8.8347E-02     -4.0337E-02     -2.3491E-02     -4.5723E-02     -5.3652E-02      6.2884E-03     -1.5265E-02      7.3114E-02      1.4786E-03     -6.4116E-04      1.6088E-02      3.0611E-02      2.8255E-02

BOOSTING SCORES:

	Linear Regression:	9.5266E-03
	Decision Tree:		1.0904E-02

"""
