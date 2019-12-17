from lib.datasearch import DataScaler
import numpy as np
import pandas as pd

def test_DataScaler():
    """Testing dataset:"""
    df = pd.DataFrame(np.array([[0, 2, 1, 2, 2, 1],
                                [1, 0, 0, 0, 3, 2],
                                [2, 1, 2, 0, 0, 1],
                                [3, 0, 1, 2, 2, 0],
                                [1, 2, 1, 1, 3, 1],
                                [0, 0, 2, 2, 2, 1],
                                [1, 1, 0, 0, 3, 2],
                                [2, 0, 2, 1, 0, 0],
                                [1, 1, 2, 1, 3, 2],
                                [0, 2, 0, 2, 0, 2],
                                [3, 0, 2, 1, 1, 0],
                                [0, 2, 0, 0, 0, 1]]),\
        columns=["Man", "Woman", "Child", "Old", "Teen", "Mean"])

    # Test the one hot encoding:
    for i in range(100):
        lists = [[], [], [], []]
        for j in range(6):
            """Randomly declare which of the features is where."""
            dice = np.random.randint(4)
            lists[dice].append(j)
        train, test = DataScaler(df, frac=0.2,\
            hcol=lists[0], ncol=lists[1], bcol=lists[2], scol=lists[3])

        if train.shape[1]==test.shape[1]:
            if i%10==0:
                print(f"\r{i/10+1}%", end="")
        elif train.columns!=test.columns:
            raise ValueError("Train and test columns do not equal.")
        else:
            raise ValueError("A feature was removed.")
    print("")
    print("Test successful.")
    """Works pretty properly. The only disfunctionality is that categories
    which are labeled with 1, 2, 3 etc. are required to start with 0."""

if __name__ == '__main__':
    test_DataScaler()
    pass
