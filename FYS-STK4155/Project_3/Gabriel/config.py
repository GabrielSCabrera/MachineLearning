import numpy as np

data_dir = "./emnist/"

# Where the EMNIST data is stored
csv_names = ["emnist-balanced-train.csv", "emnist-balanced-test.csv"]

# Where the EMNIST data will be stored as .npy files
npy_names = ["train.npy", "test.npy"]

for i in range(len(csv_names)):
    csv_names[i] = data_dir + csv_names[i]
    npy_names[i] = data_dir + npy_names[i]

# Dataset labels for each file listed above (first training, then testing)
files_labels = ["train", "test"]

# Input Shape (expected image dimension)
input_shape = (28, 28, 1)

# Neural Network Settings
layers = (64, 32)                   # Layer Configuration
activation_hid = "relu"             # Activation Function (Hidden Layers)
activation_out = "softmax"          # Activation Function (Output Layers)
kernel_size = 3                     # Kernel Size
learning_rate = 0.05                # Learning Rate
epochs = 10                         # Number of Epochs
batch_size = 50                     # Size of Batches
loss = "categorical_crossentropy"   # Loss Function

CNN_save_name = "CNN_model.h5"      # Name to save CNN with
metrics = ["categorical_accuracy"]  # Accuracy metrics to evaluate model

# Grid Search CNN Settings
gs_label            =   "dataset"
gs_kernel_size      =   [3,4,5]
gs_activation_hid   =   ["relu", "tanh"]
gs_activation_out   =   ["sigmoid", "softmax"]
gs_layers           =   [[10,5], [50,2], [100,50]]
gs_learning_rate    =   list(np.logspace(-4, 0, 3, dtype = float))
gs_epochs           =   list(np.linspace(1, 26, 3, dtype = int))
gs_batch_size       =   list(np.linspace(80, 500, 3, dtype = int))
