from sklearn.model_selection import train_test_split, cross_validate, LeaveOneOut
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier as KNN
from multiprocessing import Pool
import matplotlib.pyplot as plt

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

"""PARAMETERS"""
test_percent = 0.5  # Float in range (0,1)
rand_seed = None#11235813
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
# DISPLAYING INITIAL RESULTS
info = matrix_info(X_train, Y_train, X_test, Y_test)
print(info)

"""K-FOLD CROSS VALIDATION AND BOOTSTRAP"""
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
