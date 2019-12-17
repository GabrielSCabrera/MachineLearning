import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn.model_selection as sklms
import sklearn.preprocessing as sklpp
from lib.arabic_balanced import ArabicBalanced


def EMNIST_data_balanced(p):
    """Main function to analyse the balanced data set."""
    try:
        train = pd.read_csv("EMNIST_data/emnist_csv/emnist-balanced-train.csv")
        test = pd.read_csv("EMNIST_data/emnist_csv/emnist-balanced-test.csv")
        print("Training and testing data successfully read...")
    except:
        raise FileNotFoundError("Files not found. Begin by downloading the "+\
            "balanced training and testing data into the following directory:"+\
                " \n\t\t\tEMNIST_data/emnist_csv/emnist-balanced-train.csv"+\
                " \n\t\t\tEMNIST_data/emnist_csv/emnist-balanced-test.csv")

    """
    Values range from 0 to 255, and the first
    column is the label of the image. Data shapes are:
    (112799, 785)
    (18799, 785)
    """

    # Create fitting column names
    labels = ["label"]
    for i in range(1, 29):
        for j in range(1, 29):
            labels.append(str(j)+"x"+str(i))

    train.columns = labels
    test.columns = labels

    X_train = train.loc[:, train.columns!="label"]
    y_train = \
        pd.Series(train.loc[:, train.columns=="label"].values.reshape(-1,))
    X_test = test.loc[:, test.columns!="label"]
    y_test = \
        pd.Series(test.loc[:, test.columns=="label"].values.reshape(-1,))

    # transform the formatting of the balanced EMNIST labels
    import string
    symbol_list =   list(string.digits) +\
                    list(string.ascii_uppercase) + list(string.ascii_lowercase)
    non_lowercase = [
        "c", "i", "j", "k", "l", "m", "o",
        "p", "s", "u", "v", "w", "x", "y", "z"
    ]       # Digits which are regarded as 'their own capitals'
    for s in non_lowercase:
        symbol_list.remove(s)

    y_train = y_train.apply(lambda x: str(symbol_list[x]))
    y_test = y_test.apply(lambda x: str(symbol_list[x]))

    if not p["grid_search"]:
        a = ArabicBalanced(
            modelname = "BEMNIST_3",
            learning_rate=0.1
        )
        a.set_data(X_train, y_train, X_test, y_test)

        if p["create_model"]:
            print("Generating single model...")
            a.fit_data()
            a.save()
        else:
            print("Loading single model...")
            a.load()

        if p["eval_model"]:
            print("Performing single model evaluation...")
            a.predict_data()
            a.assert_performance()
            a.illustrate_performance()
            a.incorrect_labels()
            a.illustrate_label_grid()
        else:
            print("Loading a model without evaluating it is redundant bro...")

    else:
        import os.path
        "Perform a grid search over learning rates and max depths."
        modn = "BEMNIST_GS3"
        lr1 = 0.4; lr2 = 0.6; lr_res = 5
        md1 = 1; md2 = 10; md_res = 10
        lr_ar = np.linspace(lr1, lr2, lr_res)
        md_ar = np.linspace(md1, md2, md_res, dtype=int)
        if p["create_model"]:
            print(f"Creating several grid search models labeled {modn} using:")
            print(f"\tLearning rate array {lr_ar}")
            print(f"\tMax depth array {md_ar}")
            for i, lr in enumerate(lr_ar):
                for j, md in enumerate(md_ar):
                    modelname = modn + "_" + str(i) + str(j)
                    if os.path.isfile("BEMNISTmodels/" + modelname):
                        print(modelname + " already exists. Moving on.")
                    else:
                        a = ArabicBalanced(
                            modelname = modelname,
                            learning_rate = lr,
                            max_depth = md,
                            n_jobs = -1,
                            n_estimators=300
                        )
                        a.set_data(
                            X_train,
                            y_train,
                            X_test,
                            y_test
                        )
                        a.fit_data()
                        a.save()
                        print(f"learning rate {lr:.2f} and max depth "+\
                                f"{md:.0f} complete.")
        else:
            print(f"Loading grid search models labeled {modn}.")
            """Save several models within a grid of models array"""
            grid_models = np.zeros((lr_res, md_res), dtype=object)
            for i in range(lr_res):
                for j in range(md_res):
                    modelname = modn + "_" + str(i) + str(j)
                    a = ArabicBalanced(
                        modelname = modelname,
                        learning_rate = lr_ar[i],
                        max_depth = md_ar[j],
                        n_jobs = -1,
                        n_estimators=150
                    )
                    a.load()
                    grid_models[i,j] = a

        if p["eval_model"]:
            if lr_res != 1 and md_res != 1:
                """Produce a heatmap of the resolutions"""
                print(f"Performing grid search analysis...")
                accuracies = np.zeros((lr_res, md_res))
                Jaccard_indices = np.zeros((lr_res, md_res))
                zero_one_losses = np.zeros((lr_res, md_res))
                for i in range(lr_res):
                    for j in range(md_res):
                        a = grid_models[i,j]
                        a.set_data(X_train, y_train, X_test, y_test)
                        a.predict_data()
                        a.assert_performance(verbose=False)
                        accuracies[i,j] = a.accuracy_score * 100
                        Jaccard_indices[i,j] = np.mean(a.Jaccard_index) * 100
                        zero_one_losses[i,j] = a.zero_one_loss * 100

                ax = plt.subplot(111)
                sb.heatmap(data = accuracies, fmt=".3f", ax = ax, annot=True)
                bottom, top = ax.get_ylim()
                ax.set_ylim(bottom + 0.5, top-0.5)
                plt.title("Accuracies of the grid in %.")
                ax.set_xticklabels([f"{i:.0f}" for i in md_ar])
                ax.set_yticklabels([f"{i:.2f}" for i in lr_ar])
                plt.ylabel("Learning rate")
                plt.xlabel("Maximum tree depth")
                plt.show()

                ax = plt.subplot(111)
                sb.heatmap(data = Jaccard_indices, fmt=".3f", ax = ax, annot=True)
                bottom, top = ax.get_ylim()
                ax.set_ylim(bottom + 0.5, top-0.5)
                plt.title("Mean of the Jaccard indices of the grid in %.")
                ax.set_xticklabels([f"{i:.0f}" for i in md_ar])
                ax.set_yticklabels([f"{i:.2f}" for i in lr_ar])
                plt.ylabel("Learning rate")
                plt.xlabel("Maximum tree depth")
                plt.show()

                ax = plt.subplot(111)
                sb.heatmap(data = zero_one_losses, fmt=".3f", ax = ax, annot=True)
                bottom, top = ax.get_ylim()
                ax.set_ylim(bottom + 0.5, top-0.5)
                plt.title("Zero-One losses of the grid in %.")
                ax.set_xticklabels([f"{i:.0f}" for i in md_ar])
                ax.set_yticklabels([f"{i:.2f}" for i in lr_ar])
                plt.ylabel("Learning rate")
                plt.xlabel("Maximum tree depth")
                plt.show()

            else:
                """If one of the parameters has
                resolution 1, produce a line plot"""
                if lr_res == 1:
                    """Line plot of the max depth"""
                    accuracies = np.zeros((lr_res, md_res))
                    Jaccard_indices = np.zeros((lr_res, md_res))
                    zero_one_losses = np.zeros((lr_res, md_res))
                    for i in range(lr_res):
                        for j in range(md_res):
                            a = grid_models[i,j]
                            a.set_data(X_train, y_train, X_test, y_test)
                            a.predict_data()
                            a.assert_performance(verbose=False)
                            accuracies[i,j] = a.accuracy_score
                            Jaccard_indices[i,j] = np.mean(a.Jaccard_index)
                            zero_one_losses[i,j] = a.zero_one_loss

                    plt.plot(lr_ar, accuracies[0, :], "r-", label="Accuracies")
                    plt.plot(lr_ar, Jaccard_indices[0, :], "g-", \
                        label="Mean Jaccard indices")
                    plt.plot(lr_ar, zero_one_losses[0, :], "b-", \
                        label="Zero-One losses")
                    plt.grid()
                    plt.xlabel("Learning rate")
                    plt.title("Model performance as a function of learning rate.")
                    plt.legend(loc="best")
                    plt.show()
                    pass

                elif md_res == 1:
                    """Line plot of the learning rate"""
                    accuracies = np.zeros((lr_res, md_res))
                    Jaccard_indices = np.zeros((lr_res, md_res))
                    zero_one_losses = np.zeros((lr_res, md_res))
                    for i in range(lr_res):
                        for j in range(md_res):
                            a = grid_models[i,j]
                            a.set_data(X_train, y_train, X_test, y_test)
                            a.predict_data()
                            a.assert_performance(verbose=False)
                            accuracies[i,j] = a.accuracy_score
                            Jaccard_indices[i,j] = np.mean(a.Jaccard_index)
                            zero_one_losses[i,j] = a.zero_one_loss

                    plt.plot(lr_ar, accuracies[:,0], "r-", label="Accuracies")
                    plt.plot(lr_ar, Jaccard_indices[:,0], "g-", \
                        label="Mean Jaccard indices")
                    plt.plot(lr_ar, zero_one_losses[:,0], "b-", \
                        label="Zero-One losses")
                    plt.grid()
                    plt.xlabel("Learning rate")
                    plt.title("Model performance as a function of learning rate.")
                    plt.legend(loc="best")
                    plt.show()


def DataScaler(df, **kwargs):
    """
    Takes in a dataframe and indices which should be
    either scaled or treated categorically. The function
    then scales the training data by itself and the testing
    data by the training data. The function does not
    keep the data sorted. It returns an array where the
    one-hot encoded/categorical columns come first and then
    the scaled ones. The categorical values do not need to
    be scaled, but do not need to be encoded either if they're
    binary. This is what the one-hot index list is for.

    Parameters:
    -----------
    df : DataFrame object
        All the data in one DataFrame
    scol : list
        Scale column indices.
    ncol : list
        Normalize column indices.
    bcol : list
        Binary column indices.
    hcol : list
        One-hot encoded column indices.
    frac : decimal, Default 0.33
        Fraction of testing data. Should be in [0,1]

    Returns:
    --------
    traindf : DataFrame object
        Train data. X and y concatenated
    testdf : DataFrame object
        Test data. X and y concatenated
    """

    scl = kwargs.get("scol", [])
    ncl = kwargs.get("ncol", [])
    hcl = kwargs.get("hcol", [])
    bcl = kwargs.get("bcol", [])
    frc = kwargs.get("frac", 0.50)

    """Raise syntax error messages"""
    msg = "Sum of columns does not match with DataFrame dimensions."
    assert len(hcl+scl+bcl+ncl) == df.shape[1], msg

    msg = "Some values are not accounted for. "\
        + "Should range in total from 0,1,...,p-1"
    assert list(range(df.shape[1])) == sorted(hcl+scl+bcl+ncl), msg

    """Save the column names"""
    col_scale = [df.columns[c] for c in scl]
    col_norm = [df.columns[n] for n in ncl]
    col_bin = [df.columns[b] for b in bcl]
    col_hot = [df.columns[h] for h in hcl]

    """Perform the data splitting before the scaling"""
    train_df, test_df = sklms.train_test_split(df, test_size=frc,\
        shuffle=True)

    total_train = pd.DataFrame()
    total_test  = pd.DataFrame()

    if col_scale:
        """Scale the non-categorical variables:"""
        scaler = sklpp.StandardScaler().fit(train_df[col_scale])
        scaled_train = pd.DataFrame(scaler.transform(train_df[col_scale]),\
            columns=col_scale)
        scaled_test = pd.DataFrame(scaler.transform(test_df[col_scale]), \
            columns=col_scale)

        """Add to the total dataframes"""
        total_train = pd.concat((total_train, scaled_train), axis=1)
        total_test = pd.concat((total_test, scaled_test), axis=1)

    if col_norm:
        """Normalize the non-categorical variables:"""
        normalizer = sklpp.Normalizer().fit(train_df[col_norm])
        normed_train = pd.DataFrame(normalizer.transform(train_df[col_norm]),\
            columns=col_norm)
        normed_test = pd.DataFrame(normalizer.transform(test_df[col_norm]), \
            columns=col_norm)

        """Add to the total dataframes"""
        total_train = pd.concat((total_train, normed_train), axis=1)
        total_test = pd.concat((total_test, normed_test), axis=1)

    """Do nothing with the categorically binary variables:"""
    binary_train = train_df[col_bin]
    binary_test = test_df[col_bin]

    """Combine the data types"""
    total_train = pd.concat((total_train, binary_train.reset_index(drop=True)),\
        axis=1)
    total_test = pd.concat((total_test, binary_test.reset_index(drop=True)),\
        axis=1)

    """Now encode the non-binary categorical data and append it to the DF"""
    for i in col_hot:
        train_dums = pd.get_dummies(train_df[i], prefix=i)
        test_dums = pd.get_dummies(test_df[i], prefix=i)

        """Have to make sure that the features are the same in both the training
        and testing data. Need to insert zeros columns for some cases:"""
        while train_dums.shape[1] != test_dums.shape[1]:
            """If the two don't match, find out which one's lacking, and add
            a column of zeros to that category."""
            if train_dums.shape[1] > test_dums.shape[1]:
                """Use the set '-' operator to find missing elements"""
                missing = set(train_dums.columns) - set(test_dums.columns)
                """Account for there possibly being multiple columns missing"""
                numbers = []
                for j in list(missing):
                    numbers.append(j[-1])

                """Cycle through the columns missing and add zeros"""
                numbers = [int(x) for x in numbers]
                numbers.sort()
                for number in numbers:
                    test_dums.insert(number, i+"_"+str(number), \
                        np.zeros(test_dums.shape[0], dtype=int))

            elif train_dums.shape[1] <= test_dums.shape[1]:
                """Use the set '-' operator to find missing elements. Identical
                to the previous string, only this time the training and test
                dataframes are flipped."""
                missing = set(test_dums.columns) - set(train_dums.columns)
                """Account for there possibly being multiple columns missing"""
                numbers = []
                for j in list(missing):
                    numbers.append(j[-1])

                """Cycle through the columns missing and add zeros"""
                numbers = [int(x) for x in numbers]
                numbers.sort()
                for number in numbers:
                    train_dums.insert(number, i+"_"+str(number), \
                        np.zeros(train_dums.shape[0], dtype=int))

        """Append these to the main DataFrame and proceed to the next"""
        total_train = pd.concat((total_train, train_dums.reset_index(drop=True)), \
            axis=1)
        total_test = pd.concat((total_test, test_dums.reset_index(drop=True)), \
            axis=1)

    return total_train, total_test


if __name__ == '__main__':
    # EMNIST_data_balanced()
    pass
