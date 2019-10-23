import numpy as np
from poly import Poly
np.random.seed(0)

def generate_Franke_data(x_min, x_max, N):

    # Generating NxN meshgrid of x,y values in range [x_min, x_max]
    x = np.random.random((N,N))*(x_max-x_min) + x_min
    y = np.random.random((N,N))*(x_max-x_min) + x_min

    # Calculating the values of the Franke function at each (x,y) coordinate
    Z = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    Z += 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    Z += 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    Z += -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    init_error = np.random.normal(0, 1, Z.shape)

    f_xy = Z.copy().flatten()
    Z = Z + init_error

    # Making compatible input arrays for Regression object
    X = np.zeros((x.shape[0]*x.shape[1], 2))
    X[:,0] = x.flatten()
    X[:,1] = y.flatten()
    Y = Z.flatten()

    return X, Y, f_xy

"""TEST SETTINGS"""

N1 = 1E2
X1_0 = 0
X1_N1 = 2*np.pi

X1 = np.linspace(X1_0, X1_N1, int(N1))
Y1 = np.cos(X1) + np.random.normal(0, 0.25, X1.shape)

X1_2D = X1[:,np.newaxis]
Y1_2D = Y1[:,np.newaxis]

N2 = 100
X2_0 = 0
X2_N2 = 1
X2, Y2, f_xy = generate_Franke_data(X2_0, X2_N2, N2)

Y2_2D = Y2[:,np.newaxis]

count = 1
success = 0
test_descr_list = []
failed = []
number_len = 4
status_pos = 50

"""HELPER FUNCTIONS"""

def test_msg(test_descr):
    lmsg = f"\r{globals()['count']:<4d}"
    lmsg += test_descr
    status = "EXECUTING"
    msg = lmsg + " "*(status_pos-len(status)-len(lmsg))
    msg += f"\033[93m{status}\033[m"
    print(msg, end = "")
    globals()["test_descr_list"].append(test_descr)

def fail_msg(e = None):
    lmsg = f"\r{globals()['count']:<{number_len}d}"
    lmsg += globals()["test_descr_list"][-1]
    status = "FAIL"
    msg = lmsg + " "*(status_pos-len(status)-len(lmsg))
    msg += f"\033[91m{status}\033[m"
    print(msg)
    if e is not None and len(e.args) != 0:
        status = "ERROR"
        err_len = 77 - number_len - len(status) - 2
        msg = " "*number_len + f"\033[m{status}:\033[m "
        print(msg, end = "")
        e_msg = e.args[0].strip()
        if len(e_msg) > err_len:
            msg = e_msg[:err_len] + "..."
        else:
            msg = e_msg
        print(msg)

    failed_msg = f'{count:<{number_len}d}{globals()["test_descr_list"][-1]}'
    globals()["failed"].append(failed_msg)
    globals()["count"] += 1

def pass_msg():
    lmsg = f"\r{globals()['count']:<4d}"
    lmsg += globals()["test_descr_list"][-1]
    status = "PASS"
    msg = lmsg + " "*(status_pos-len(status)-len(lmsg))
    msg += f"\033[92m{status}\033[m"
    print(msg)
    globals()["count"] += 1
    globals()["success"] += 1

def run_summary():
    title_pass = f"PASSED TESTS: {success}"
    spaces = 80-len(title_pass)
    title_pass = f"\n\033[1;44m{title_pass}" + " "*spaces

    failed_tests = count-success-1
    if failed_tests != 0:
        title_fail = f"FAILED TESTS: {failed_tests}"
        spaces = 80-len(title_fail)
        title_fail = f"\033[1;41m{title_fail}" + " "*spaces + "\033[m"

        print(title_pass + "\033[m")
        print(title_fail)
        for f in failed:
            print(f)
    else:
        msg2 = "100% SUCCESS"
        l = len(msg2)
        print(title_pass[:-spaces+5] + msg2 + " "*(spaces-5-l) + "\033[m")

"""TEST EXECUTION"""

test_msg("Init Poly")
try:
    M1 = Poly(X1, Y1)
    M2 = Poly(X2, Y2)
except Exception as e:
    fail_msg(e)
else:
    pass_msg()

tests = ["M1.p == 1", "M2.p == 2", "M1.q == 1", "M2.q == 1", "M1.N == N1",
         "M2.N == N2**2"]

for t in tests:
    test_msg(t)
    try:
        assert eval(t)
    except Exception as e:
        fail_msg(e)
    else:
        pass_msg()

test_msg("M1.X == X1_2D")
try:
    assert(np.all(np.array_equal(M1.X, X1_2D)))
except Exception as e:
    fail_msg(e)
else:
    pass_msg()

test_msg("M2.X == X2")
try:
    assert(np.all(np.array_equal(M2.X, X2)))
except Exception as e:
    fail_msg(e)
else:
    pass_msg()

test_msg("M1.Y == Y1_2D")
try:
    assert(np.all(np.array_equal(M1.Y, Y1_2D)))
except Exception as e:
    fail_msg(e)
else:
    pass_msg()

test_msg("M2.Y == Y2_2D")
try:
    assert(np.all(np.array_equal(M2.Y, Y2_2D)))
except Exception as e:
    fail_msg(e)
else:
    pass_msg()

tests = ["M1.split()", "M2.split()", "M1.X_train", "M2.X_train", "M1.X_test",
         "M2.X_test", "M1.Y_train", "M2.Y_train", "M1.Y_test", "M2.Y_test"]

for t in tests:
    test_msg(t)
    try:
        exec(t)
    except Exception as e:
        fail_msg(e)
    else:
        pass_msg()

M1.OLS(1)

"""SUMMARY"""
run_summary()
