"""
Runs the artemis-data classifier training
This script is designed to be run on the DoC machines.
# The dataset used with this code is stored at: https://gitlab.doc.ic.ac.uk/yy3219/wikiart_concrete
"""

import os
# import re
import time
import pathlib
from ast import literal_eval

import keras.preprocessing.image_dataset
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.applications import resnet_v2

# from sklearn.utils import class_weight

from nn.model_architectures import model_image_classification, model_image_classification_from_concepts
from nn.utils import ds_unzip, calculate_metrics, \
    train_model, train_concept_model, train_concept_model_sequential, train_concept_model_balanced_class, calculate_concept_metrics


def preprocess_data(images, labels):
    return resnet_v2.preprocess_input(images), labels


# def to_image_ids(file_paths:List[str]) -> List[str]:
def to_image_ids(file_paths):
    return [pathlib.PurePath(file_path).name[:-4] for file_path in file_paths]
    # return [re.match(r".*/([a-zA-Z0-9_]+)\.jpg", file_path).group(1) for file_path in file_paths]


# def idx_lookup(df: pd.DataFrame, row_key: str, value: str) -> int:
def idx_lookup(df, row_key, value):
    return df[row_key][df[row_key] == value].index[0]


network_name = "image_classifier_network"
# large_data_dir = "/vol/bitbucket/yy3219/roko-for-charlize/Thesis_Data"
# large_data_dir = "/Users/Cherry0904/Desktop/roko-for-charlize/Thesis_Data"
# model_dir = "/Users/Cherry0904/Desktop/roko-for-charlize/Thesis_Data/Models"
model_dir = "/vol/bitbucket/yy3219/roko-for-charlize/Thesis_Data/Models" # The directory to save the models


use_concepts = True
concepts_only_training = False
use_MLP = False
# use_resnet = False
sequential_training = False

n_concepts = 347+10 # Change it 290+10
dir = '/vol/bitbucket/yy3219/artemis_images_2000_ds'
# dir = '/vol/bitbucket/yy3219/pn_upsample_artemis_images_2000_ds' # The 2000-subset image folder on bitbucket
# dir = '/Users/Cherry0904/Desktop/artemis_concrete_ds'

dataset_train_split = 1500 # 1325 # 1082  # Change it: (0.75, 0.1, 0.15)
dataset_val_point = 1700 # 1501 # 1227   # Change it

dataset = keras.preprocessing.image_dataset.image_dataset_from_directory(dir, image_size=(224, 224), batch_size=1,
                                                                         shuffle=False)  # the returned y_data from one folder should all be 1
# print(dataset)
X_data, _, _ = ds_unzip(dataset, use_concepts=False)
# X_data, y_data, _ = ds_unzip(dataset, use_concepts=False)
X_data = resnet_v2.preprocess_input(X_data)

# Read the pickle file
project_dir = '/vol/bitbucket/yy3219/roko-for-charlize'
# project_dir = '/Users/Cherry0904/Desktop/roko-for-charlize'
path = os.path.join(project_dir, "Extracted_Concepts/artemis_2000_0.9_final_dict_old_codex.pkl")
luke_output = pd.read_pickle(path)

labels_keys = ['id', 'label', 'concepts']
labels_dict = {key: luke_output[key] for key in labels_keys}
labels = pd.DataFrame.from_dict(labels_dict) # a df containing image_name (id), label, concept array of 0 or 1
print(labels.shape)

# Add the binary image features to each concept array by matching the image_name
# binary_img_features_df_csv = '/Users/Cherry0904/Desktop/binary_img_features_each_image.csv'
binary_img_features_df_csv = '/vol/bitbucket/yy3219/roko-for-charlize/Thesis_Data/Data/binary_img_features_each_image.csv'
binary_img_features_df = pd.read_csv(binary_img_features_df_csv)
# binary_img_features_df['binary_feature_vector'] = binary_img_features_df['binary_feature_vector'].apply(literal_eval) # Turn the string into a list (not array)
new_df = pd.merge(labels, binary_img_features_df, on='id', how='left')
new_df = new_df.fillna(0) # Fill all the NaN values with 0
new_df['binary_feature_vector'] = new_df[['overall_red', 'overall_green', 'overall_blue',
       'large_hue_variation', 'high_saturation', 'high_global_contrast',
       'high_local_contrast', 'frequent_repeated_pattern',
       'large_number_of_lines', 'long_lines_on_average']].values.tolist()
new_df = new_df[['id', 'label', 'concepts', 'binary_feature_vector']]
# Append each binary image feature vector to the concept array
new_df['merged_concepts'] = new_df.apply(lambda x: np.append(np.asarray(x.binary_feature_vector), x.concepts), axis = 1)
# Replace the original concept array in 'labels' with the updated one 
labels['concepts'] = new_df['merged_concepts']
# print(labels.columns)

# Upsample the negative class by adding repetitive images (with '_copy" added to names) to the labels df
# labels_negative = labels.loc[labels['label'] == 'negative']
# labels_negative['id'] = labels_negative['id'] + '_copy'
# labels = pd.concat([labels, labels_negative])

# print(labels['id'].iloc[0])
# print(dataset.file_paths[0])

ds_img_ids = to_image_ids(dataset.file_paths)
concept_list_ids = [idx_lookup(labels, "id", img_id) for img_id in ds_img_ids]
# Return the list of concept arrays corresponding to the images in the repo
concept_list = [tf.convert_to_tensor(labels["concepts"].iloc[id]) for id in concept_list_ids] 
label_list = [labels["label"].iloc[id] for id in concept_list_ids]

# Make the output y a vector for classification
num_classes = 9
classes = np.array(['amusement', 'awe', 'excitement', 'anger', 'disgust', 'fear', 'sadness', 'something else'])

class_dict = {
    'amusement': 0,
    'awe': 1,
    'contentment': 2,
    'excitement': 3,
    'anger': 4,
    'disgust': 5, 
    'fear': 6, 
    'sadness': 7, 
    'something else': 8}

# num_classes = 2
# classes = np.array(['positive', 'negative'])
# class_dict = {'positive': 0, 'negative': 1}

inv_class_dict = {v: k for k, v in class_dict.items()} # Needed for visualisation of attention weights

# y = np.array([class_dict[label] for label in labels['label']]) # An array of integers in 0 to 8
# Return a list of binary vector form of y
y_binary = [tf.constant(tf.keras.utils.to_categorical(class_dict[label], num_classes), shape=[num_classes, ]) for label in label_list] 
# print(y_binary.shape) # May change this to prob vector instead of binary vector
# print(concept_list)
# print(y_binary)


y_data = [{"c_probs": c, "probs": y} for c, y in zip(concept_list, y_binary)] # augment the output y
# print(len(y_data))
# print(y_data)


def y_generator_fun():
    for y in y_data:
        # print(y)
        yield y

X_dataset = tf.data.Dataset.from_tensor_slices(X_data)
y_dataset = tf.data.Dataset.from_generator(y_generator_fun, output_signature=({
    "c_probs": tf.TensorSpec(shape=(n_concepts,)),
    "probs": tf.TensorSpec(shape=(num_classes,)) # "probs": tf.TensorSpec(shape=())
}))
preprocesed_dataset = tf.data.Dataset.zip((X_dataset, y_dataset)).batch(1)
# print(preprocesed_dataset)

X, y, c = ds_unzip(preprocesed_dataset, use_concepts=True)
# print(y.shape)

# Manually shuffling because tf dataset shuffle doesn't work
indices = tf.range(start=0, limit=tf.shape(X)[0], dtype=tf.int32)
shuffled_indices = tf.random.shuffle(indices, seed = 789)

X = tf.gather(X, shuffled_indices)
y = tf.gather(y, shuffled_indices)
c = tf.gather(c, shuffled_indices)

# X_train, y_train, c_train = ds_unzip(preprocesed_dataset, use_concepts=True)
# X_test, y_test, c_test = ds_unzip(shuffled_dataset, use_concepts=True)
X_train = X[:dataset_train_split]
X_val = X[dataset_train_split:dataset_val_point]
X_test = X[dataset_val_point:]
y_train = y[:dataset_train_split]
y_val = y[dataset_train_split:dataset_val_point]
y_test = y[dataset_val_point:]
c_train = c[:dataset_train_split]
c_val = c[dataset_train_split:dataset_val_point]
c_test = c[dataset_val_point:]

t = int(time.time())

if not sequential_training:
    # This is the full model pipeline
    if use_concepts and not concepts_only_training:
        # model, history = train_concept_model_sequential(model_tuple, X_train, y_train, c_train, X_test, y_test, c_test,
        #                                                 model_dir, t, n_concepts,
        #                                                 network_name, epochs=100, batch_size=32)
        _, model = model_image_classification(num_classes=9, n_concept_layer=n_concepts) # Change number of classes
        model, history = train_concept_model(model, X_train, y_train, c_train, X_val, y_val, c_val, model_dir, t,
                                            n_concepts, name="concept_full_artemis", epochs=50, batch_size=32)

        cf_matrix, accuracy, macro_f1, mismatch, y_pred, cf_concepts, accuracy_concepts \
            = calculate_concept_metrics(model, X_test, y_test, c_test.numpy())

        print('Results for model at time {}'.format(t))
        print('Accuracy : {}'.format(accuracy))
        print('F1-score : {}'.format(macro_f1))
        print(cf_matrix)
        print(cf_concepts)
        print(accuracy_concepts)


    # This is the model from concepts to labels with attention
    elif use_concepts and concepts_only_training:
        model = model_image_classification_from_concepts(num_classes=9, n_concepts=n_concepts)
        model, H = train_model(model, c_train, y_train, c_val, y_val, model_dir, t,
                            batch_size=32, epochs=50, name="concept_only_artemis")

        cf_matrix, accuracy, macro_f1, mismatch, y_pred, = calculate_metrics(model, c_test, y_test)
        print('Results for model at time {}'.format(t))
        print('Accuracy : {}'.format(accuracy))
        print('F1-score : {}'.format(macro_f1))
        print(cf_matrix)


    # This is the model from images to labels with no concepts
    else:
        _, model = model_image_classification(num_classes=9, n_concept_layer=n_concepts, use_concepts=False)
        model, H = train_model(model, X_train, y_train, X_val, y_val, model_dir, t,
                            batch_size=32, epochs=50, name="concept_free_artemis")

        cf_matrix, accuracy, macro_f1, mismatch, y_pred, = calculate_metrics(model, X_test, y_test)
        print('Results for model at time {}'.format(t))
        print('Accuracy : {}'.format(accuracy))
        print('F1-score : {}'.format(macro_f1))
        print(cf_matrix)


if sequential_training:
    concept_model, model = model_image_classification(num_classes=9, n_concept_layer=n_concepts) # Change number of classes
    model_tuple = concept_model, model
    model, history = train_concept_model_sequential(model_tuple, X_train, y_train, c_train, X_val, y_val, c_val, model_dir, t,
                                   n_concepts, name='concept_full_sequential_artemis', epochs=50, batch_size=32)
    cf_matrix, accuracy, macro_f1, mismatch, y_pred, cf_concepts, accuracy_concepts \
        = calculate_concept_metrics(model, X_test, y_test, c_test.numpy())
    print('Results for model at time {}'.format(t))
    print('Accuracy : {}'.format(accuracy))
    print('F1-score : {}'.format(macro_f1))
    print(cf_matrix)
    print(cf_concepts)
    print(accuracy_concepts)



# # TODO: remove duplication
# if use_concepts:
#     c_pred, y_pred = full_model.predict(X_test)
#     y_pred = y_pred.squeeze()
#     y_pred = (y_pred >= 0.5).astype(int)
#     y_true = y_test.numpy()
#
#     cf_matrix = confusion_matrix(y_true, y_pred)
#     accuracy = accuracy_score(y_true, y_pred)
#
#     print(f"Possible classes are {list(enumerate(dataset.class_names))}")
#     print("Confusion matrix:")
#     print(cf_matrix)
#     print(f"Accuracy is: {accuracy}")
#
#     c_test = c_test.numpy().flatten()
#     c_pred = c_pred.flatten()
#     c_pred[c_pred <= 0.5] = 0
#     c_pred[c_pred > 0.5] = 1
#     cf_concepts = confusion_matrix(c_test, c_pred)
#     precision = precision_score(c_test, c_pred)
#     print("Confusion matrix:")
#     print(cf_concepts)
#     print(f"Precision is: {precision}")
#
#     # Store to vector
#     concept_preds, _ = full_model.predict(train_ds)
#     true_train_labels = labels["label"].iloc[concept_list_ids[:1700]].values
#     print(concept_preds.shape)
#     print(true_train_labels.shape)
#
#     explanations = luke_output['explanations']
#     output = {
#         "probs": concept_preds,
#         "explanations": explanations,
#         "labels": true_train_labels,
#     }
#     pickle.dump(output, open("for_fastlas_examples_train.pkl", 'wb'))
#
#     concept_preds, _ = full_model.predict(test_ds)
#     true_train_labels = labels["label"].iloc[concept_list_ids[1700:]].values
#     print(concept_preds.shape)
#     print(true_train_labels.shape)
#
#     explanations = luke_output['explanations']
#     output = {
#         "probs": concept_preds,
#         "explanations": explanations,
#         "labels": true_train_labels,
#     }
#
#     pickle.dump(output, open("for_fastlas_examples_test.pkl", 'wb'))
#
#
# else:
#     y_pred = full_model.predict(X_test).squeeze()
#     y_pred = (y_pred >= 0.5).astype(int)
#     y_true = y_test.numpy()
#
#     cf_matrix = confusion_matrix(y_true, y_pred)
#     accuracy = accuracy_score(y_true, y_pred)
#
#     print(f"Possible classes are {list(enumerate(dataset.class_names))}")
#     print("Confusion matrix:")
#     print(cf_matrix)
#     print(f"Accuracy is: {accuracy}")
