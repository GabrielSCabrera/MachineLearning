from solving_PDEs import PDE_solver
import numpy as np

count = 1
success = 0
test_descr_list = []
failed = []
number_len = 4
status_pos = 60

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

test_msg("Creating PDE_solver() instances")
try:
    A = PDE_solver()
except Exception as e:
    fail_msg(e)
else:
    pass_msg()

test_msg("Setting initial conditions")
try:
    u0 = np.array([1,2,3,4,5], dtype = float)
    A.set_initial_conditions(u0)
except Exception as e:
    fail_msg(e)
else:
    pass_msg()

test_msg("Running the solver")
try:
    T, dt = 10, 0.1
    L, dx = 1, 0.01
    # Array of time values from 0 to T
    t = np.arange(0, T + dt, dt)
    # Array of position values from 0 to L
    x = np.arange(0, L + dx, dx)
    A.solve(t, x)
except Exception as e:
    fail_msg(e)
else:
    pass_msg()
