import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import lib.datasearch as ds
import argparse

def main():
    """Parse the arguments"""
    argp = argparser()
    p = vars(argp)

    """Declare the data for analysis"""
    ds.EMNIST_data_balanced(p)
    return 1

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cm", "--create_model",
        help="Used to create a new model.",
        action="store_true"
    )
    parser.add_argument(
        "-gs", "--grid_search",
        help="Used to calculate a grid search",
        action="store_true"
    )
    parser.add_argument(
        "-em", "--eval_model",
        help="Used to evaluate the model",
        action="store_true"
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    import time
    start = time.time()
    main()
    print(f"Completed in {(time.time()-start):.0f} seconds.")
    """
    Test the functions using:

    $ python3 -m pytest

    """
