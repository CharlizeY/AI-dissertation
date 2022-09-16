#!/usr/bin/env python
"""
Trains the baseball-classifier.
This script is designed to be run on the DoC machines
"""

# ln[0]:
import argparse
import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.backend import clear_session

from nn.model_architectures import lookup_model_by_network_type
from nn.utils import train_model, train_concept_model_sequential, train_concept_model, calculate_metrics, \
    calculate_concept_metrics, get_concept_vector_dict, attn_prediction, load_labels_and_features

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

# In[1]:


from warnings import simplefilter
import time

# project_dir = '/vol/bitbucket/rp218/luke-for-roko'
# project_dir = '/vol/bitbucket/yy3219/roko-for-charlize'
project_dir = '/Users/Cherry0904/Desktop/roko-for-charlize'
# large_data_dir = "/vol/bitbucket/rp218/Thesis_Data"
large_data_dir = f"{project_dir}/Thesis_Data"

# Be aware of the fold id for logging
argparser = argparse.ArgumentParser()
argparser.add_argument("--fold", type=int, nargs='?', default=0)
args = argparser.parse_args()


# ## Load Data


# Parse the wrong format to the correct one
# path = os.path.join(project_dir, "Extracted_Concepts/final_dict_new_codex_no_pruning.pkl")
path = os.path.join(project_dir, "Extracted_Concepts/final_dict_new_codex.pkl")
luke_output = pd.read_pickle(path)

labels_keys = ['id', 'label', 'concepts']
labels_dict = {key: luke_output[key] for key in labels_keys}
labels_df_filtered = pd.DataFrame.from_dict(labels_dict)

print(labels_df_filtered)

# In[4]:


FEATURE_EXTRACTOR = 'Resnet50V2'

features_dir = f'{large_data_dir}/Feature_vectors_{FEATURE_EXTRACTOR}'
print(features_dir)

# In[5]:


X, labels = load_labels_and_features(labels_df_filtered, features_dir)


model_dir = f"{large_data_dir}/Models"
os.makedirs(model_dir, exist_ok=True)

concepts_text = pd.DataFrame.from_dict({"explanations": luke_output["explanations"]})

num_classes = 5
classes = np.array(['strike', 'ball', 'play', 'foul', 'out'])
n_concepts = 78
# n_concepts = 342

class_dict = {
    'strike': 0,
    'ball': 1,
    'play': 2,
    'foul': 3,
    'out': 4}

inv_class_dict = {v: k for k, v in class_dict.items()}

concept_matrix = labels['concepts'].values
concept_matrix = np.stack(concept_matrix, axis=0)
idx = np.argwhere(np.all(concept_matrix[..., :] == 0, axis=0))
concept_matrix = np.delete(concept_matrix, idx, axis=1)
concept_matrix = concept_matrix[:, :n_concepts]
print(concept_matrix.shape)

y = np.array([class_dict[label] for label in labels['label']])

y_binary = tf.keras.utils.to_categorical(y, num_classes)
print(y_binary.shape)

print(X.shape)
X_train0 = X[:1700, :, :]
y_train_binary = y_binary[:1700, :]
X_test0 = X[1700:, :, :]
y_test_binary = y_binary[1700:, :]
concept_train = concept_matrix[:1700, :]
concept_test = concept_matrix[1700:, :]

print(X_train0.shape)
print(y_train_binary.shape)
print(concept_train.shape)
print(X_test0.shape)
print(y_test_binary.shape)
print(concept_test.shape)

# ## Train Model

# In[7]:


# network_type = 'Conv1D'
# network_type = 'concept_Conv'
network_type = 'concept_Conv_attn'
# network_type = 'LSTM'
# network_type = 'concept_LSTM'
# network_type = 'concept_LSTM_attn'
# network_type = 'concept_attn'
# network_type = 'model_MLP'
# network_type = 'seq_concept_Conv_attn'

# specifying hyper-parameters
batch_size = 16
_, win_len, dim = X_train0.shape
n_concepts = concept_train.shape[1]

# In[8]:
print('building the model ...')
model = lookup_model_by_network_type(network_type, dim, win_len, num_classes, n_concepts)
if type(model) is tuple:
    _, full_model = model
    print(full_model.summary())
else:
    print(model.summary())

# In[ ]:


t = int(time.time())

if (network_type == 'Conv1D' or network_type == 'LSTM'):
    model, H = train_model(model, X_train0, y_train_binary, X_test0, y_test_binary,
                           model_dir, t, batch_size=batch_size, epochs=100, name=network_type)

elif network_type == 'seq_concept_Conv_attn':
    model, H = train_concept_model_sequential(model, X_train0, y_train_binary, concept_train, X_test0, y_test_binary,
                                              concept_test, model_dir, t, n_concepts, batch_size=batch_size, epochs=100,
                                              name=network_type)

elif network_type == 'concept_attn' or network_type == "model_MLP":
    model, H = train_model(model, concept_train, y_train_binary, concept_test, y_test_binary, model_dir, t,
                           batch_size=batch_size, epochs=100, name=network_type)


else:
    model, H = train_concept_model(model, X_train0, y_train_binary, concept_train,
                                   X_test0, y_test_binary, concept_test,
                                   model_dir, t, n_concepts, batch_size=batch_size, epochs=100, name=network_type)

# In[ ]:


print(f"Results for model at time {t}")
if (network_type == 'Conv1D' or network_type == 'LSTM'):
    cf_matrix, accuracy, macro_f1, mismatch, y_pred, = calculate_metrics(model, X_test0,
                                                                         y_test_binary)
    print('Accuracy : {}'.format(accuracy))
    print('F1-score : {}'.format(macro_f1))
    print(cf_matrix)

elif network_type == "concept_attn" or network_type == "model_MLP":
    cf_matrix, accuracy, macro_f1, mismatch, y_pred, = calculate_metrics(model, concept_test,
                                                                         y_test_binary)
    print('Accuracy : {}'.format(accuracy))
    print('F1-score : {}'.format(macro_f1))
    print(cf_matrix)

else:
    cf_matrix, accuracy, macro_f1, mismatch, y_pred, cf_concepts, accuracy_concepts = calculate_concept_metrics(model,
                                                                                                                X_test0,
                                                                                                                y_test_binary,
                                                                                                                concept_test)

    print('Accuracy : {}'.format(accuracy))
    print('F1-score : {}'.format(macro_f1))
    print(cf_matrix)
    print(cf_concepts)
    print(accuracy_concepts)

print()

# Store baseball results with concepts for rule-learning training
if network_type == "concept_Conv_attn":
    print("Storing predictions as a pickle")
    true_train_labels = classes[np.argmax(y_train_binary, axis=1)]
    concept_preds = attn_prediction(model, X_train0)
    output = get_concept_vector_dict(luke_output["explanations"], concept_preds, true_train_labels)
    pickle.dump(output, open(f"for_fastlas_examples_train_attn_{args.fold}.pkl", 'wb'))

    true_test_labels = classes[np.argmax(y_test_binary, axis=1)]
    concept_preds = attn_prediction(model, X_test0)
    output = get_concept_vector_dict(luke_output["explanations"], concept_preds, true_test_labels)
    pickle.dump(output, open(f"for_fastlas_examples_test_attn_{args.fold}.pkl", 'wb'))

# Delete old model to avoid OOM issues
del model
clear_session()
