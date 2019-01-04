import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.pyplot.switch_backend('agg')
import pandas as pd
import numpy as np
from keras.utils import np_utils
# import pydot
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from utils.ml_plotting import plot_roc_curve, plot_out_dists, plot_acc_loss, plot_output_over_time, plot_confusion_matrix, plot_kfold_roc
from utils.ml_tools import my_train_test_split, get_confusion_matrix, kfold_evaluation
from datetime import datetime
import os, sys
from keras_nets import custom_CNN, LeNet5, LSTM
##############################################################
## Script for training Neuarl Networks and producing plots. ##
## Requires sklearn, tensorflow, keras                      ##
##                                                          ##
## 2018 Alexander Sopio                                     ##
## Note: new code needs new utils.ml_plotting               ##
##############################################################

# ----- Configuration -----

np.random.seed(94)
loss_function = 'mean_squared_logarithmic_error' #'mean_squared_error' #'categorical_crossentropy'
vector_length = 25 * 25
optimizer = 'adam'
nb_epochs = 5

nb_samples = 1000
model_name = 'custom_CNN' #"LeNet5"
do_shuffle = True
save_trained_model = True
show_plots = True
input_filename = "/unix/cedar/asopio/lund_plane_ntuples/lund_images_25by25.npz"
timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H%M%S")

# if __name__ == "__main__":

print("loading {} feature vector dataset...".format(vector_length))
#Input images

#replacing "file" inside np.load with other files
zprime_data = np.load("input_files/lund_images_wprime_25by25.npz")
#zprime_data = np.load("/mnt/storage/lborgna/SamData/Sig_BigSig.root200000_400000.npy")

dijets_data = np.load("input_files/lund_images_dijets_25by25.npz")
#dijets_data = np.load("/mnt/storage/lborgna/SamData/BKG_BigBKG.root100000_200000.npy")


X_W = zprime_data['W']
X_top = zprime_data['top'] #not used anywhere
X_q = dijets_data['Q']
X_g = dijets_data['G']

X_QCD = X_q[:nb_samples] + X_g[:nb_samples]

#Label data
y_W = np.ones(len(X_W))
y_QCD = np.zeros(len(X_QCD))

#Pick nb_samples from dataset in equal proportion from both classes
X_W_samp = X_W [:nb_samples]
y_W_samp = y_W [:nb_samples]
X_QCD_samp = X_QCD[:nb_samples]
y_QCD_samp = y_QCD[:nb_samples]

X = np.append(X_W_samp, X_QCD_samp, axis=0)
y = np.append(y_W_samp, y_QCD_samp, axis=0)

# Input shape for CNN: 1 colour channel + 1 imagege channel (flattened)
X = X.reshape( (X.shape[0], X.shape[1], X.shape[2], 1) )

# Split data into training and testing samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=do_shuffle)

# Reshape target vectors
y = np_utils.to_categorical(y, 2)
y_train = np_utils.to_categorical(y_train, 2)
y_test = np_utils.to_categorical(y_test, 2)


input_shape = X[0].shape
print("Input vector shape:", input_shape)

# ----- Training-----

#Get model architecture
exec('neural_net = '+model_name)
model=neural_net.GetNetArchitecture(input_shape)
model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])


tb_callback = TensorBoard(log_dir='./save/logs/run-{}'.format(timestamp), histogram_freq=1, batch_size=32,
                          write_graph=True, write_grads=False, write_images=False, embeddings_freq=0,
                          embeddings_layer_names=None, embeddings_metadata=None)

model_history = model.fit(X_train, y_train, epochs=nb_epochs, batch_size=50,
                          validation_data=(X_test, y_test), shuffle = do_shuffle, callbacks=[tb_callback])

#Save model
saved_weights_path = "/mnt/storage/sdaley/save/models/{}_initial.hdf5".format(model_name)
if save_trained_model:
    print("saving model...")
    model.save_weights(saved_weights_path)
else:
    print("Model is not being saved!")


# ----- Evaluation ----
#Evaluate on test and training set
y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)

#Flatten target vectors
y_pred_1d = y_pred[:, 1]
y_pred_train_1d = y_pred_train[:, 1]
y_test_1d = y_test[:, 1]
y_train_1d = y_train[:, 1]

#do k - fold evaluation
kfolds = 4
plt.figure()
tprs, mean_tpr, mean_fpr, auc_scores = kfold_evaluation(model, model_name, saved_weights_path, nb_epochs,
                                              X, y, kfolds, random_over_sampling=False, return_mean_tpr=True)

#Save test results
history_df = pd.DataFrame(model_history.history)
history_df.to_pickle("/mnt/storage/sdaley/save/models/results1/train_history_{}_{}.pkl".format(model_name, timestamp))
score_df = pd.DataFrame({"test":y_test_1d, "pred":y_pred_1d})
score_df.to_pickle("/mnt/storage/sdaley/save/models/results1/scores_{}_{}.pkl".format(model_name, timestamp))

fpr, tpr, thresholds = roc_curve(y_test_1d, y_pred_1d)
roc_auc = roc_auc_score(y_test_1d, y_pred_1d)

y_decisions = np.round(y_pred_1d)

f1 = f1_score(y_test_1d, y_decisions)

# print(y_test_1d[:100])
# print(y_pred_1d[:100])

print("ROC AUC:", roc_auc)
print("F1 SCORE:", f1)

# ----- Plotting -----
# plot_model(model, to_file="plots/Layout_"+model_name+".png")
plot_acc_loss(model_history, model_name)
# Output distributions
plot_out_dists(y_pred_1d, y_test_1d, model_name + "_test", binarray=np.linspace(0,1,200), label0="QCD", label1="W")
plot_out_dists(y_pred_train_1d, y_train_1d, model_name + "_train", binarray=np.linspace(0,1,200), label0="QCD", label1="W")

# Roc Curves
plot_kfold_roc(tprs, mean_fpr, auc_scores, model_name, kfolds, baselines = [0.01, 0.1, 0.5])
plot_roc_curve(y_pred_1d, y_test_1d, model_name, use_rejection=True)
plot_roc_curve(y_pred_1d, y_test_1d, model_name, use_rejection=False)

plot_confusion_matrix(y_pred, y_test, model_name)
if show_plots:
    plt.show()

conf_mat = get_confusion_matrix(y_pred, y_test)
print("Confusio matrix:")
print(conf_mat)
