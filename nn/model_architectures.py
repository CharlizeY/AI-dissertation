import tensorflow as tf
from keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Conv1D
from tensorflow.keras.layers import Input, MaxPooling1D, Multiply, LSTM, Attention, concatenate


def model_Conv1D(dim, win_len, num_classes, n_concepts, num_feat_map=128, p=0.3):
    model = Sequential()
    model.add(Conv1D(num_feat_map, kernel_size=3, activation='relu', padding='same',
                     input_shape=(win_len, dim), name='Conv_1'))
    model.add(MaxPooling1D(pool_size=4, name='Max_pool_1'))
    model.add(BatchNormalization(name='Bn_1'))
    model.add(Dropout(p, name='Drop_1'))
    model.add(Conv1D(16, kernel_size=3, activation='relu', padding='same', name='Conv_2'))
    model.add(BatchNormalization(name='Bn_2'))
    model.add(Dropout(p, name='Drop_2'))
    model.add(Flatten(name='flatten'))

    # Bottom as if we predicted concepts
    num_hidden_mlp = num_feat_map * 2
    model.add(Dense(num_hidden_mlp, activation='relu', input_shape=(n_concepts,), name='dense_1'))
    model.add(BatchNormalization(name='Bn_3'))
    model.add(Dropout(p, name='Drop_3'))
    model.add(Dense(num_hidden_mlp, activation='relu', name='dense_2'))
    model.add(BatchNormalization(name='Bn_4'))
    model.add(Dropout(p, name='Drop_4'))
    model.add(Dense(num_classes, activation='softmax', name='probs'))
    return model


def model_Conv1D_concepts(dim, win_len, num_classes, n_concepts, num_feat_map=64, p=0.3):
    inputs = Input(shape=(win_len, dim), name='Input_1')
    x = Conv1D(128, kernel_size=3, activation='relu', padding='same', name='Conv_1')(inputs)
    x = MaxPooling1D(pool_size=4, name='Max_pool_1')(x)
    x = BatchNormalization(name='Bn_1')(x)
    x = Dropout(p, name='Drop_1')(x)
    x = Conv1D(16, kernel_size=3, activation='relu', padding='same', name='Conv_2')(x)
    x = BatchNormalization(name='Bn_2')(x)
    x = Dropout(p, name='Drop_2')(x)
    x = Flatten(name='flatten')(x)
    concepts = Dense(n_concepts, name='concept_logits')(x)
    concepts = Activation('sigmoid', name='c_probs')(concepts)
    out = Dense(num_classes, name='logits')(concepts)
    out = Activation('softmax', name='probs')(out)

    model = Model(inputs=inputs, outputs=[concepts, out], name="Video_concepts")
    return model


def model_Conv1D_attn_concepts(dim, win_len, num_classes, n_concepts, num_feat_map=128, p=0.3):
    inputs = Input(shape=(win_len, dim), name='Input_1')
    x = Conv1D(num_feat_map, kernel_size=3, activation='relu', padding='same', name='Conv_1')(inputs)
    x = MaxPooling1D(pool_size=4, name='Max_pool_1')(x)
    x = BatchNormalization(name='Bn_1')(x)
    x = Dropout(p, name='Drop_1')(x)
    x = Conv1D(16, kernel_size=3, activation='relu', padding='same', name='Conv_2')(x)
    x = BatchNormalization(name='Bn_2')(x)
    x = Dropout(p, name='Drop_2')(x)
    x = Flatten(name='flatten')(x)
    concepts = Dense(n_concepts, name='concept_logits')(x)
    concepts = Activation('sigmoid', name='c_probs')(concepts)

    attention = Dense(n_concepts, name='attention_weights', activation='tanh')(concepts)
    attention = Activation('softmax', name='attn_score')(attention)

    out = Multiply(name='mul')([attention, concepts])

    out = Dense(num_classes, name='logits')(out)
    out = Activation('softmax', name='probs')(out)

    num_hidden_mlp = 2 * num_feat_map
    # out = Dense(num_hidden_mlp, activation='relu', name='dense_1')(out)
    # out = BatchNormalization(name='Bn_3')(out)
    # out = Dropout(p, name='Drop_3')(out)
    # out = Dense(num_hidden_mlp, activation='relu', name='dense_2')(out)
    # out = BatchNormalization(name='Bn_4')(out)
    # out = Dropout(p, name='Drop_4')(out)
    # out = Dense(num_classes, activation='softmax', name='probs')(out)

    model = Model(inputs=inputs, outputs=[concepts, out], name="Video_concepts")
    return model


def concept_predicition_Conv1d_model(dim, win_len, n_concepts, num_feat_map=128, p=0.3):
    inputs = Input(shape=(win_len, dim), name='Input_1')
    x = Conv1D(num_feat_map, kernel_size=3, activation='relu', padding='same', name='Conv_1')(inputs)
    x = MaxPooling1D(pool_size=4, name='Max_pool_1')(x)
    x = BatchNormalization(name='Bn_1')(x)
    x = Dropout(p, name='Drop_1')(x)
    x = Conv1D(16, kernel_size=3, activation='relu', padding='same', name='Conv_2')(x)
    x = BatchNormalization(name='Bn_2')(x)
    x = Dropout(p, name='Drop_2')(x)
    x = Flatten(name='flatten')(x)
    concepts = Dense(n_concepts, name='concept_logits')(x)
    concepts = Activation('sigmoid', name='c_probs')(concepts)

    return Model(inputs=inputs, outputs=concepts, name="Concept_prediction")


def seq_model_Conv1D_attn_concepts(dim, win_len, num_classes, n_concepts, num_feat_map=128, p=0.3):
    concept_prediction_model = concept_predicition_Conv1d_model(dim, win_len, n_concepts, num_feat_map, p)

    attention = Dense(n_concepts, name='attention_weights', activation='tanh')(concept_prediction_model.output)
    attention = Activation('softmax', name='attn_score')(attention)

    out = Multiply(name='mul')([attention, concept_prediction_model.output])

    num_hidden_mlp = 2 * num_feat_map
    out = Dense(num_classes, name='logits')(out)
    out = Activation('softmax', name = 'probs')(out)
    #
    #
    # out = Dense(num_hidden_mlp, activation='relu', name='dense_1')(out)
    # out = BatchNormalization(name='Bn_3')(out)
    # out = Dropout(p, name='Drop_3')(out)
    # out = Dense(num_hidden_mlp, activation='relu', name='dense_2')(out)
    # out = BatchNormalization(name='Bn_4')(out)
    # out = Dropout(p, name='Drop_4')(out)
    # out = Dense(num_classes, activation='softmax', name='probs')(out)

    full_model = Model(inputs=concept_prediction_model.inputs, outputs=[concept_prediction_model.output, out],
                       name="Video_concepts")

    return concept_prediction_model, full_model


def model_LSTM(dim, win_len, num_classes, num_hidden_lstm=128, p=0.3):
    inputs = Input(shape=(win_len, dim), name='Input_1')
    x = LSTM(num_hidden_lstm, return_sequences=True, name='lstm_1')(inputs)
    x = MaxPooling1D(pool_size=4, name='Max_pool_1')(x)
    x = BatchNormalization(name='Bn_1')(x)
    x = Dropout(p, name='Drop_1')(x)
    x = LSTM(16, return_sequences=True, name='lstm_s')(x)
    x = BatchNormalization(name='Bn_2')(x)
    x = Dropout(p, name='Drop_2')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(300, activation='relu')(x)
    x = BatchNormalization(name='Bn_3')(x)
    x = Dropout(p, name='Drop_3')(x)
    out = Dense(num_classes, name='logits')(x)
    out = Activation('softmax', name='probs')(out)

    model = Model(inputs=inputs, outputs=out, name="Video_concepts")
    return model


def model_LSTM_concepts(dim, win_len, num_classes, n_concepts, num_hidden_lstm=128, p=0.3):
    inputs = Input(shape=(win_len, dim), name='Input_1')
    x = LSTM(num_hidden_lstm, return_sequences=True, name='lstm_1')(inputs)
    x = MaxPooling1D(pool_size=4, name='Max_pool_1')(x)
    x = BatchNormalization(name='Bn_1')(x)
    x = Dropout(p, name='Drop_1')(x)
    x = LSTM(16, return_sequences=True, name='lstm_s')(x)
    x = BatchNormalization(name='Bn_2')(x)
    x = Dropout(p, name='Drop_2')(x)
    x = Flatten(name='flatten')(x)
    concepts = Dense(n_concepts, name='concept_logits')(x)
    concepts = Activation('sigmoid', name='c_probs')(concepts)
    out = Dense(num_classes, name='logits')(concepts)
    out = Activation('softmax', name='probs')(out)

    model = Model(inputs=inputs, outputs=[concepts, out], name="Video_concepts")
    return model


def model_LSTM_attn_concepts(dim, win_len, num_classes, n_concepts, num_hidden_lstm=128, p=0.3):
    inputs = Input(shape=(win_len, dim), name='Input_1')
    x = LSTM(num_hidden_lstm, return_sequences=True, name='Lstm_1')(inputs)
    x = MaxPooling1D(pool_size=4, name='Max_pool_1')(x)
    x = BatchNormalization(name='Bn_1')(x)
    x = Dropout(p, name='Drop_1')(x)
    x = LSTM(16, return_sequences=True, name='Lstm_2')(x)
    x = BatchNormalization(name='Bn_2')(x)
    x = Dropout(p, name='Drop_2')(x)

    x = Flatten(name='flatten')(x)
    concepts = Dense(n_concepts, name='concept_logits')(x)
    concepts = Activation('sigmoid', name='c_probs')(concepts)

    attention = Dense(n_concepts, name='attention_weights', activation='tanh')(concepts)
    attention = Activation('softmax', name='attn_score')(attention)

    out = Multiply(name='mul')([attention, concepts])
    out = Dense(num_classes, name='logits')(out)
    out = Activation('softmax', name='probs')(out)

    model = Model(inputs=inputs, outputs=[concepts, out], name="Video_concepts")
    return model


def model_MLP(n_concepts, num_classes, num_hidden_mlp=256, p=0.3):
    model = Sequential()
    model.add(Dense(num_hidden_mlp, activation='relu', input_shape=(n_concepts,), name='dense_1'))
    model.add(BatchNormalization(name='Bn_1'))
    model.add(Dropout(p, name='Drop_1'))
    model.add(Dense(num_hidden_mlp, activation='relu', name='dense_2'))
    model.add(BatchNormalization(name='Bn_2'))
    model.add(Dropout(p, name='Drop_2'))
    model.add(Dense(num_classes, activation='softmax', name='dense_out'))
    return model


# def model_attn_concepts(n_concepts: int, num_classes: int, num_hidden: int = 128):
#     # input: (batch_size, one_hot_vector_size)
#     # output: (batch_size, 1)

#     inputs = Input(shape=n_concepts, name='Input_1')

#     attention = Attention(name='attention_weights')([inputs, inputs, inputs])

#     out = Dense(num_hidden, name='dense_1', activation='relu')(attention)
#     out = Dense(num_hidden, name='dense_2', activation='relu')(out)
#     out = Dense(num_classes, name='logits')(out)
#     out = Activation('softmax', name='probs')(out)

#     model = Model(inputs=inputs, outputs=out, name="Video_concepts")
#     return model


# Returns (concept prediction model, full model) where the full model uses the concept prediction model
def model_image_classification(num_classes: int, n_concept_layer: int, num_hidden: int = 1024, use_concepts=True):
    concept_prediction_model = model_image_concept_prediction(n_concept_layer)

    attention = Dense(n_concept_layer, name='attention_weights', activation='tanh')(concept_prediction_model.output)
    attention = Activation('softmax', name='attn_score')(attention)
    out = Multiply(name='mul')([attention, concept_prediction_model.output])

    # out = Dense(num_hidden, name='dense_1', activation='relu')(out)
    # out = Dense(num_hidden, name='dense_2', activation='relu')(out)
    out = Dense(num_classes, name='probs', activation='sigmoid')(out)     # The sigmoid activation function fits for multi-label classification of binary labels

    if use_concepts:
        return concept_prediction_model, Model(concept_prediction_model.inputs,
                                               outputs=[concept_prediction_model.output, out])
    return concept_prediction_model, Model(concept_prediction_model.inputs, outputs=out)


def model_image_classification_append(num_classes: int, n_concept_layer: int, num_hidden: int = 1024, fine_tune_from_layer: int = 100):
    base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Only fine-tune from specific layer because early layers tend to have good general features
    for layer in base_model.layers[:fine_tune_from_layer]:
        layer.trainable = False

    flat1 = Flatten()(base_model.layers[-1].output)
    # print(tf.shape(flat1))

    concept_inputs = Input(shape=(n_concept_layer, ), name='c_inputs')
    
    merge = concatenate([flat1, concept_inputs])
    # print(tf.shape(merge))

    # attention = Dense(100352+n_concept_layer, name='attention_weights', activation='tanh')(merge)
    # attention = Activation('softmax', name='attn_score')(attention)
    # print(tf.shape(attention))
    # out = Multiply(name='mul')([attention, merge])

    out = Dense(num_classes, name='probs', activation='sigmoid')(merge) 
    # out = Dense(num_classes, name='probs', activation='sigmoid')(out)     # The sigmoid activation function fits for multi-label classification of binary labels

    return Model(inputs=[concept_inputs, base_model.inputs], outputs=out)


def model_image_classification_no_attention(num_classes: int, n_concept_layer: int, num_hidden: int = 1024, use_concepts=True):
    concept_prediction_model = model_image_concept_prediction(n_concept_layer)

    # attention = Dense(n_concept_layer, name='attention_weights', activation='tanh')(concept_prediction_model.output)
    # attention = Activation('softmax', name='attn_score')(attention)
    # out = Multiply(name='mul')([attention, concept_prediction_model.output])

    out = Dense(num_hidden, name='dense_1', activation='relu')(concept_prediction_model.output)
    out = Dense(num_hidden, name='dense_2', activation='relu')(out)
    out = Dense(num_classes, name='probs', activation='sigmoid')(out)     # The sigmoid activation function fits for multi-label classification of binary labels

    if use_concepts:
        return concept_prediction_model, Model(concept_prediction_model.inputs,
                                               outputs=[concept_prediction_model.output, out])
    return concept_prediction_model, Model(concept_prediction_model.inputs, outputs=out)


def model_image_classification_no_attention_raw_image(num_classes: int, n_concept_layer: int, num_hidden: int = 1024, use_concepts=True):
    concept_prediction_model = model_image_concept_prediction_raw_image(n_concept_layer)

    # attention = Dense(n_concept_layer, name='attention_weights', activation='tanh')(concept_prediction_model.output)
    # attention = Activation('softmax', name='attn_score')(attention)
    # out = Multiply(name='mul')([attention, concept_prediction_model.output])

    out = Dense(num_hidden, name='dense_1', activation='relu')(concept_prediction_model.output)
    out = Dense(num_hidden, name='dense_2', activation='relu')(out)
    out = Dense(num_classes, name='probs', activation='sigmoid')(out)     # The sigmoid activation function fits for multi-label classification of binary labels

    if use_concepts:
        return concept_prediction_model, Model(concept_prediction_model.inputs,
                                               outputs=[concept_prediction_model.output, out])
    return concept_prediction_model, Model(concept_prediction_model.inputs, outputs=out)


def model_image_classification_from_images(num_classes: int, n_concept_layer: int, num_hidden: int = 1024, use_concepts=True, fine_tune_from_layer: int = 100):
    base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Only fine-tune from specific layer because early layers tend to have good general features
    for layer in base_model.layers[:fine_tune_from_layer]:
        layer.trainable = False

    flat1 = Flatten()(base_model.layers[-1].output)

    # out = Dense(num_hidden, name='dense_1', activation='relu')(out)
    # out = Dense(num_hidden, name='dense_2', activation='relu')(out)
    out = Dense(num_classes, name='probs', activation='sigmoid')(out)     # The sigmoid activation function fits for multi-label classification of binary labels

    return Model(base_model.inputs, outputs=out)


def model_image_classification_from_raw_images(num_classes: int, n_concept_layer: int, num_hidden: int = 1024, use_concepts=True, fine_tune_from_layer: int = 100):
    inputs = Input(shape=(224, 224, 3), name='Input_1')
    flat1 = Flatten()(inputs)

    # out = Dense(num_hidden, name='dense_1', activation='relu')(out)
    # out = Dense(num_hidden, name='dense_2', activation='relu')(out)
    out = Dense(num_classes, name='probs', activation='sigmoid')(out)     # The sigmoid activation function fits for multi-label classification of binary labels

    return Model(inputs=inputs, outputs=out)


def model_image_classification_from_concepts(num_classes: int, n_concepts: int, num_hidden: int = 1024):
    inputs = Input(shape=n_concepts, name='Input_1')
    attention = Dense(n_concepts, name='attention_weights', activation='tanh')(inputs)
    attention = Activation('softmax', name='attn_score')(attention)
    out = Multiply(name='mul')([attention, inputs])

    out = Dense(num_hidden, name='dense_1', activation='relu')(out)
    out = Dense(num_hidden, name='dense_2', activation='relu')(out)
    out = Dense(num_classes, name='probs', activation='sigmoid')(out)

    return Model(inputs=inputs, outputs=out)


# Model used for concept prediction using raw image pixels
def model_image_concept_prediction_raw_image(n_concept_layer: int, fine_tune_from_layer: int = 100):
    # base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # # Only fine-tune from specific layer because early layers tend to have good general features
    # for layer in base_model.layers[:fine_tune_from_layer]:
    #     layer.trainable = False
    inputs = Input(shape=(224, 224, 3), name='Input_1')
    flat1 = Flatten()(inputs)

    concepts = Dense(n_concept_layer, name='concept_logits')(flat1)
    concepts = Activation('sigmoid', name='c_probs')(concepts)

    return Model(inputs=inputs, outputs=concepts)


# Model used for concept prediction of the artemis concepts
def model_image_concept_prediction(n_concept_layer: int, fine_tune_from_layer: int = 100):
    base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Only fine-tune from specific layer because early layers tend to have good general features
    for layer in base_model.layers[:fine_tune_from_layer]:
        layer.trainable = False

    flat1 = Flatten()(base_model.layers[-1].output)

    concepts = Dense(n_concept_layer, name='concept_logits')(flat1)
    concepts = Activation('sigmoid', name='c_probs')(concepts)

    return Model(base_model.inputs, outputs=concepts)


def lookup_model_by_network_type(network_type: str, dim: str, win_len: str, num_classes: int, n_concepts: int = None,
                                 num_hidden: int = None, p: float = None):
    if network_type == 'Conv1D':
        num_hidden = 128 if num_hidden is None else num_hidden
        p = 0.5 if p is None else p
        model = model_Conv1D(dim, win_len, num_classes, n_concepts, num_feat_map=num_hidden, p=p)

    elif network_type == 'concept_Conv':
        p = 0.5 if p is None else p
        model = model_Conv1D_concepts(dim, win_len, num_classes, n_concepts, p=p)

    elif network_type == 'concept_Conv_attn':
        num_hidden = 64 if num_hidden is None else num_hidden
        p = 0.5 if p is None else p
        model = model_Conv1D_attn_concepts(dim, win_len, num_classes, n_concepts, num_feat_map=num_hidden, p=p)

    elif network_type == 'LSTM':
        num_hidden = 512 if num_hidden is None else num_hidden
        p = 0.2 if p is None else p
        model = model_LSTM(dim, win_len, num_classes, num_hidden_lstm=num_hidden, p=p)

    elif network_type == 'concept_LSTM_attn':
        num_hidden = 512 if num_hidden is None else num_hidden
        p = 0.2 if p is None else p
        model = model_LSTM_attn_concepts(dim, win_len, num_classes, n_concepts, num_hidden_lstm=num_hidden, p=p)

    # elif network_type == 'concept_attn':
    #     model = model_attn_concepts(n_concepts, num_classes)

    elif network_type == 'model_MLP':
        p = 0.3 if p is None else p
        num_hidden = 256 if num_hidden is None else num_hidden
        model = model_MLP(n_concepts, num_classes, num_hidden, p)

    elif network_type == "seq_concept_Conv_attn":
        num_hidden = 128 if num_hidden is None else num_hidden
        p = 0.3 if p is None else p
        model = seq_model_Conv1D_attn_concepts(dim, win_len, num_classes, n_concepts, num_feat_map=num_hidden, p=p)

    else:
        raise NotImplementedError(f"{network_type} does not exist")

    return model
