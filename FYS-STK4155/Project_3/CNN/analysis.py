import numpy as np

import model_creator
import grid_search
import preprocess
import config

def sort_by_accuracy(models):
    accuracies = [model["result"] for model in models]
    order = np.argsort(accuracies)[::-1]
    sorted_models = list(np.array(models)[order])
    return sorted_models

def tex_table_accuracies(sorted_models):
    out = ("\\begin{table}[H]\n\t"
           "\\centered\n\t"
           "\\begin{tabular}{c c c c c}\n\t\t"
           "Accuracy & Activation & Learning Rate & Batch Size & Kernel Size"
           " \\\\ \n\t\t\\hline \n\t\t")

    for n,model in enumerate(sorted_models):
        metadata = model['metadata']
        result = f"{model['result']:.4f}"
        out += (f"{result} & {metadata['activation_hid']} "
                f"& {metadata['learning_rate']} & {metadata['batch_size']} "
                f"& {metadata['kernel_size']} ")
        if n < len(sorted_models) - 1:
            out += "\\\\"
        out += "\n\t"
        if n < len(sorted_models) - 1:
            out += "\t"

    out += ("\\end{tabular}\n\t"
            "\\caption{Ranked categorical accuracies for the CNN with "
            "varying hyperparameters; for a constant layer configuration "
            f"{config.gs_layers[0]}, {config.gs_epochs[0]} epochs, and output "
            f"activation function {config.gs_activation_out[0]}."
            "}\n\\end{table}")

    return out

if __name__ == "__main__":

    data = preprocess.read_data()
    data = preprocess.one_hot(data)
    data = preprocess.reshape_4D(data)

    models = grid_search.load_grid()
    sorted_models = sort_by_accuracy(models)
    tex_table = tex_table_accuracies(sorted_models)
    best_model = sorted_models[0]
    # best_model = model_creator.train_CNN_continue(data, best_model, epochs = 1)
    print(tex_table)
