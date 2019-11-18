from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
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

def get_stats(model, X_train, Y_train, X_test, Y_test, labels):

    """PREDICTING OUTPUTS USING LINEAR REGRESSION MODEL"""
    Y_predict_train = model.predict(X_train)
    Y_predict_test = model.predict(X_test)

    """EXTRACTING COEFFICIENTS"""
    betas = model.coef_.squeeze()[1:]

    """CALCULATING STANDARD ERROR"""
    var_betas = np.diagonal(np.linalg.inv(X_test.T @ X_test))
    std_err = np.sqrt(var_betas)[1:]

    """CALCULATING P-VALUES"""
    t_values = betas/std_err
    p_values = 2*(1-stats.t.cdf(np.abs(t_values), X_test.shape[0]-X_test.shape[1]))

    """CALCULATING MSE"""
    MSE_train = np.mean((Y_predict_train - Y_train)**2)
    MSE_test = np.mean((Y_predict_test - Y_test)**2)

    """CALCULATING R2-SCORE"""
    SS_tot = np.sum((Y_test-np.mean(Y_test))**2)
    SS_res = np.sum((Y_predict_test-Y_test)**2)
    R2 = 1 - SS_res/SS_tot

    """SORTING FEATURES BASED ON P-VALUES (low --> high)"""
    idx = np.argsort(p_values)

    if len(idx) > 1:
        labels = labels[idx]
        betas = betas[idx]
        std_err = std_err[idx]
        p_values = p_values[idx]

    return labels, betas, std_err, p_values, MSE_train, MSE_test, R2, idx

def sort_data(model, X_train, Y_train, X_test, Y_test, labels):

    """GETTING STATISTICAL DATA"""
    labels, betas, std_err, p_values, MSE_train, MSE_test, R2, idx = \
    get_stats(model, X_train, Y_train, X_test, Y_test, labels)

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
            f"STATISTICS:\n\n\tMSE train:\t{MSE_train:.4E}\n\tMSE test:\t"
            f"{MSE_test:.4E}\n\tR²:\t{R2:.4E}")

    return info, labels, betas, std_err, p_values, MSE_train, MSE_test, R2, idx

def print_data(mode, stop, MSE_train, MSE_test, labels, betas, std_err, p_values):
    stat_vals = f"\t{'Label':^10s} {'Coeff':^10s} {'Std Err':^10s} {'P-Val':^10s} "
    longest_str = np.max(list(map(len, globals()["label_descr"].values())))+1
    stat_vals += f"{'Description':^{longest_str}s}\n"
    stat_vals += "\t" + "–"*len(stat_vals) + "\n"
    for i,j,k,l in zip(labels, betas, std_err, p_values):
        stat_vals += f"\t{i:>10s} {j:>10.2E} {k:>10.2E} {l:>10.2E}"
        stat_vals += f" {label_descr[i]:>{longest_str}s}\n"

    string = (f"\nPREDICTION INFO SORTED BY IMPORTANCE:\n\n{stat_vals}\n\n"
              f"STATISTICS:\n\n\tMSE train:\t{MSE_train:.4E}\n\tMSE test:\t"
              f"{MSE_test:.4E}\n\tR²:\t{R2:.4E}")
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
        temp = get_stats(model, X_train, Y_train, X_test, Y_test, labels)
        if temp[3][i] < stop:
            betas, std_err, p_values, MSE_train, MSE_test, R2, idx = temp[1:]
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

    return MSE_train, MSE_test, betas_arr, std_err_arr, p_arr, labels_importance

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
            model, X_train, X_test, Y_train, Y_test = get_model(X_temp, Y, options)
            temp = get_stats(model, X_train, Y_train, X_test, Y_test, labels)
            betas, std_err, p_values, MSE_train, MSE_test, R2, idx = temp[1:]
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
    return MSE_train, MSE_test, betas_arr, std_err_arr, p_arr, labels[col_rank]

def k_fold(data):
    X, Y, k, alpha, options = data

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

    """PERFORMING LASSO FIT"""
    model = Lasso(alpha = alpha, max_iter = 1E5)
    cv_results = cross_validate(model, X_train, Y_train, cv = k,
                                scoring = "neg_mean_squared_error",
                                return_train_score = True)
    scores_train = -cv_results["train_score"]
    scores_test = -cv_results["test_score"]
    return np.mean(scores_train), np.mean(scores_test)

def bootstrap(data):
    X, Y, k, alpha, options = data

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

    scores_train = []
    scores_test = []

    """PERFORMING LASSO FIT"""
    model = Lasso(alpha = alpha, max_iter = 1E5)
    for i in range(k):
        X_step, Y_step = resample(X_train, Y_train, random_state = rand_seed,
                                  n_samples = X_train.shape[0], replace = True)

        model.fit(X_step, Y_step)
        Y_predict_train = model.predict(X_train)
        Y_predict_test = model.predict(X_test)
        scores_train.append(np.mean((Y_predict_train - Y_train)**2))
        scores_test.append(np.mean((Y_predict_test - Y_test)**2))

    return np.mean(scores_train), np.mean(scores_test)

def CV_compare(alphas):
    k_fold_train_scores = []
    k_fold_test_scores = []

    bootstrap_train_scores = []
    bootstrap_test_scores = []

    pool = Pool()
    all_args = [(X, Y, k, a, options) for a in alphas]

    k_fold_idx = []
    bootstrap_idx = []

    print(f"{0:4d}%", end = "")

    count = 0
    for n,i in enumerate(pool.imap(k_fold, all_args)):
        count += 1
        k_fold_train_scores.append(i[0])
        k_fold_test_scores.append(i[1])
        k_fold_idx.append(n)
        print(f"\r{int(50*n/len(alphas)):4d}%", end = "")

    count = 0
    for n,i in enumerate(pool.imap(bootstrap, all_args)):
        count += 1
        bootstrap_train_scores.append(i[0])
        bootstrap_test_scores.append(i[1])
        bootstrap_idx.append(n)
        print(f"\r{50+int(50*count/len(alphas)):4d}%", end = "")
    print(f"\r", end = "")

    bootstrap_train_scores = np.array(bootstrap_train_scores)[bootstrap_idx]
    bootstrap_test_scores = np.array(bootstrap_test_scores)[bootstrap_idx]
    k_fold_train_scores = np.array(k_fold_train_scores)[k_fold_idx]
    k_fold_test_scores = np.array(k_fold_test_scores)[k_fold_idx]

    boot_idx = np.argmin(bootstrap_test_scores)
    k_fold_idx = np.argmin(k_fold_test_scores)

    plt.semilogx(alphas, k_fold_train_scores, label = "Cross-Validation (Train)")
    plt.semilogx(alphas, bootstrap_train_scores, label = "Bootstrap (Train)")
    plt.semilogx(alphas, k_fold_test_scores, label = "Cross-Validation (Test)")
    plt.semilogx(alphas, bootstrap_test_scores, label = "Bootstrap (Test)")

    plt.semilogx([alphas[k_fold_idx]], [k_fold_train_scores[k_fold_idx]], "bo",
    label = f"KNN best $\\alpha$ = {alphas[k_fold_idx]:.4E}, MSE train = {k_fold_train_scores[k_fold_idx]:.4E}")
    plt.semilogx([alphas[boot_idx]], [bootstrap_train_scores[boot_idx]], "ro",
    label = f"Bootstrap best $\\alpha$ = {alphas[boot_idx]:.4E}, MSE train = {bootstrap_train_scores[boot_idx]:.4E}")

    plt.semilogx([alphas[k_fold_idx]], [k_fold_test_scores[k_fold_idx]], "go",
    label = f"MSE test = {k_fold_test_scores[k_fold_idx]:.4E}")
    plt.semilogx([alphas[boot_idx]], [bootstrap_test_scores[boot_idx]], "co",
    label = f"MSE test = {bootstrap_test_scores[boot_idx]:.4E}")

    plt.xlabel("Complexity Parameter $\\alpha$")
    plt.ylabel("MSE")
    plt.xlim([np.min(alphas), np.max(alphas)])
    plt.legend()
    plt.savefig("plot_1.pdf", dpi = 250)
    plt.close()

    # print(f"\nLASSO:\n\tMSE train:\t{}")

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
    Y_predict_train = gam.predict(X_train)
    Y_predict_test = gam.predict(X_test)
    MSE_train = np.mean((Y_predict_train - Y_train)**2)
    MSE_test = np.mean((Y_predict_test - Y_test)**2)
    return MSE_train, MSE_test

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
info, labels_sort, betas_sort, std_err_sort, p_values_sort, MSE_train, MSE_test, R2, idx = \
sort_data(model, X_train, Y_train, X_test, Y_test, labels)
print(info)

"""BACKWARD ELIMINATION"""
stop_b = 0.7
mode_b = "backward elimination"
MSE_train_b, MSE_test_b, betas_b, std_err_b, p_b, labels_back = \
backward_elimination(X.copy(), Y.copy(), labels, options, stop_b)

print("\nBACKWARD ELIMINATION")
print_data(mode_b, stop_b, MSE_train_b, MSE_test_b, labels_back, betas_b, std_err_b, p_b)

"""FORWARD SUBSTITUTION"""
stop_f = 0.7
mode_f = "forward substitution"
MSE_train_f, MSE_test_f, betas_f, std_err_f, p_f, labels_forward = \
forward_substitution(X.copy(), Y.copy(), labels, options, stop_f)

print("\nFORWARD SUBSTITUTION")
print_data(mode_f, stop_f, MSE_train_f, MSE_test_f, labels_forward, betas_f, std_err_f, p_f)

"""BACKWARD ELIMINATION"""
stop_b = 0.8
mode_b = "backward elimination"
MSE_train_b, MSE_test_b, betas_b, std_err_b, p_b, labels_back = \
backward_elimination(X.copy(), Y.copy(), labels, options, stop_b)

print("\nBACKWARD ELIMINATION")
print_data(mode_b, stop_b, MSE_train_b, MSE_test_b, labels_back, betas_b, std_err_b, p_b)

"""FORWARD SUBSTITUTION"""
stop_f = 0.8
mode_f = "forward substitution"
MSE_train_f, MSE_test_f, betas_f, std_err_f, p_f, labels_forward = \
forward_substitution(X.copy(), Y.copy(), labels, options, stop_f)

print("\nFORWARD SUBSTITUTION")
print_data(mode_f, stop_f, MSE_train_f, MSE_test_f, labels_forward, betas_f, std_err_f, p_f)

"""COMPARING LAST TWO METHODS"""
print("\nFEATURE IMPORTANCE COMPARISON:\n")
msg1 = f"{'Rank':7s} {'Backward Elimination':30s} {'Forward Substitution':30s}"
msg2 = "–"*len(msg1)
print("\t" + msg1 + "\n\t" + msg2 + "\n")
for n,(i,j) in enumerate(zip(labels_back, labels_forward)):
    print(f"\t{n+1:<7d} {label_descr[i]:30s} {label_descr[j]:30s}")

"""K-FOLD CROSS VALIDATION AND BOOTSTRAP"""
alphas = np.logspace(-8, 1, 1000)
CV_compare(alphas)

"""IMPLEMENTING GENERALIZED ADDITIVE MODEL"""
MSE_GAM_lin = GAM(X.copy(), Y.copy(), factor = False)
MSE_GAM_fac = GAM(X.copy(), Y.copy(), factor = True)
msg = (f"        \nGENERALIZED ADDITIVE MODEL:\n\n\tLinear MSE:\t"
       f"{MSE_GAM_lin[0]:.4E}\t{MSE_GAM_lin[1]:.4E}\n\tPolynomial MSE:\t"
       f"{MSE_GAM_fac[0]:.4E}\t{MSE_GAM_fac[1]:.4E}")
print(msg)

"""BOOSTING"""
LR = AdaBoostRegressor(base_estimator = LinearRegression())
LR.fit(X_train, Y_train.squeeze())

Y_predict_train = LR.predict(X_train)
Y_predict_test = LR.predict(X_test)
score_train = np.mean((Y_predict_train-Y_train.squeeze())**2)
score_test = np.mean((Y_predict_test-Y_test.squeeze())**2)

coefficients = LR.estimators_[-1].coef_
coef = ""
for c,l in zip(coefficients, labels):
    coef += f"{c:16.4E}"
coef = coef.strip()
print(f"\nLINEAR REGRESSION BOOSTING COEFFICIENTS:\n\n{coef}")

msg = (f"\nBOOSTING SCORES:\n\n\tLinear Regression:\t{score_train:.4E}\t{score_test:.4E}\n\t")

DTR = AdaBoostRegressor(base_estimator = DecisionTreeRegressor(), n_estimators=500)
DTR.fit(X_train, Y_train.squeeze())
Y_predict_train = DTR.predict(X_train)
Y_predict_test = DTR.predict(X_test)
score_train = np.mean((Y_predict_train-Y_train.squeeze())**2)
score_test = np.mean((Y_predict_test-Y_test.squeeze())**2)
msg += (f"Decision Tree:\t\t{score_train:.4E}\t{score_test:.4E}")
print(msg)
