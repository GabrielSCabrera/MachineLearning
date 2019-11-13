from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from scipy import stats
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
    X_train = scaler.fit_transform(X_train, Y_train)
    X_test = scaler.transform(X_test)

    """PERFORMING LINEAR REGRESSION FIT"""
    model = LinearRegression().fit(X_train, Y_train)

    return model, X_train, X_test, Y_train, Y_test

def get_stats(model, X_test, Y_test, labels):

    """PREDICTING OUTPUTS USING LINEAR REGRESSION MODEL"""
    Y_predict = model.predict(X_test)

    """EXTRACTING COEFFICIENTS"""
    betas = model.coef_.squeeze()

    """CALCULATING STANDARD ERROR"""
    residuals = Y_test - Y_predict
    RSS = residuals.T @ residuals
    var_hat = RSS[0,0]/(X_test.shape[0] - X_test.shape[1])
    var_betas = np.linalg.inv(X_test.T @ X_test)*var_hat
    std_err = np.diagonal(var_betas)**0.5

    """CALCULATING P-VALUES"""
    t_values = betas/std_err
    p_values = 2*(1 - stats.t.cdf(np.abs(t_values), Y_test.shape[0]-X_test.shape[1]))

    """CALCULATING MSE"""
    MSE = np.mean((Y_predict - Y_test)**2)

    """CALCULATING R2-SCORE"""
    SS_tot = np.sum((Y_test-np.mean(Y_test))**2)
    SS_res = np.sum((Y_predict-Y_test)**2)
    R2 = 1 - SS_res/SS_tot

    """SORTING FEATURES BASED ON STANDARD ERROR (high --> low)"""
    idx = np.argsort(p_values)
    return labels[idx], betas[idx], std_err[idx], p_values[idx], MSE, R2, idx

def sort_data(model, X_train, Y_train, X_test, Y_test, labels):
    """DESCRIBING LABELS"""
    label_descr = {
    "ALTER": "Age",
    "ADHEU": "Allergic Coryza",
    "SEX": "Gender",
    "HOCHOZON": "High Ozone Village",
    "AMATOP": "Maternal Atopy",
    "AVATOP": "Paternal Atopy",
    "ADEKZ": "Neurodermatitis",
    "ARAUCH": "Smoker",
    "AGEBGEW": "Birth Weight",
    "FSNIGHT": "Night/Morning Cough",
    "FLGROSS": "Height",
    "FMILB": "Dust Sensitivity",
    "FNOH24": "Max. NO2",
    "FTIER": "Fur Sensitivity",
    "FPOLL": "Pollen Sensitivity",
    "FLTOTMED": "No. of Medis/Lufu",
    "FO3H24": "24h Max Ozone Value",
    "FSPT": "Allergic Reaction",
    "FTEH24": "24h Max Temperature",
    "FSATEM": "Shortness of Breath",
    "FSAUGE": "Itchy Eyes",
    "FLGEW": "Weight",
    "FSPFEI": "Wheezy Breath",
    "FSHLAUF": "Cough"}

    """GETTING STATISTICAL DATA"""
    labels, betas, std_err, p_values, MSE, R2, idx = \
    get_stats(model, X_test, Y_test, labels)

    """COMPILING INFORMATION FOR PRINTING TO TERMINAL"""

    stat_vals = f"\t{'Label':^10s} {'Coeff':^10s} {'Std Err':^10s} {'P-Val':^10s} "
    longest_str = np.max(list(map(len, label_descr.values())))+1
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

def backward(X, Y, labels, options):
    model, X_train, X_test, Y_train, Y_test = get_model(X, Y, options)

    info, labels_sort, betas_sort, std_err_sort, p_values_sort, MSE, R2, idx = \
    sort_data(model, X_train, Y_train, X_test, Y_test, labels)

    X_short = X[:,idx]

    MSE_arr, R2_arr = np.zeros(p), np.zeros(p)
    MSE_arr[-1] = MSE
    R2_arr[-1] = R2
    for i in range(p-1):
        X_step = X_short[:,:p-i]
        labels_step = labels[:p-i]
        model, X_train, X_test, Y_train, Y_test = get_model(X_step, Y, options)
        labels, betas, std_err, p_values, MSE, R2, idx = \
        get_stats(model, X_test, Y_test, labels)
        MSE_arr[p-i-1] = MSE
        R2_arr[p-i-1] = R2
        X_short = X[:,idx]
    return MSE_arr, R2_arr

"""PARAMETERS"""
test_percent = 0.5  # Float in range (0,1)
rand_seed = 11235813

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

# DISPLAYING INITIAL RESULTS
info, labels_sort, betas_sort, std_err_sort, p_values_sort, MSE, R2, idx = \
sort_data(model, X_train, Y_train, X_test, Y_test, labels)
print(info)

"""BACKWARD ELIMINATION"""
MSE_back, R2_back = backward(X, Y, labels, options)
