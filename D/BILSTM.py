# import library
import os
import time
import string

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

#Task D main code
def run_D():
    print('\n\n******************************Running Task D: Bidirectional Long Short-Term Memory******************************')

    print('\nThe current path of Task D is', os.getcwd())

    # create saving path
    save_path = './Results/D'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # import implemented functions
    import NLP as NLP

    # read data
    training_df = pd.read_csv('Datasets/training.csv')
    validation_df = pd.read_csv('Datasets/validation.csv')
    test_df = pd.read_csv('Datasets/test.csv')

    # map category to label
    category = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    labels_dict = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}

    training_df['description'] = training_df['label'].map(labels_dict)
    validation_df['description'] = validation_df['label'].map(labels_dict)
    test_df['description'] = test_df['label'].map(labels_dict)

    # plot distribution of training, validation and test data
    print('\nPlot distribution of training, validation and test data')
    print('\nTraining data:')
    training_distribution = NLP.plot_distribution(training_df, 'description', 'Training')
    training_distribution.savefig(os.path.join(save_path, '1 Label Distribution of Training Data.png'))

    print('\nValidation data:')
    validation_distribution = NLP.plot_distribution(validation_df, 'description', 'Validation')
    validation_distribution.savefig(os.path.join(save_path, '2 Label Distribution of Validation Data.png'))

    print('\nTest data:')
    test_distribution = NLP.plot_distribution(test_df, 'description', 'Test')
    test_distribution.savefig(os.path.join(save_path, '3 Label Distribution of Test Data.png'))

    # data clearning and get maximum length
    print('\nStart data cleaning ...')
    print('Remove all URL links https?:\/\/\S+', )
    print('Removing all punctuation', string.punctuation)
    print('Convert all letters to lowercase')

    word2id, max_len = NLP.get_vocab(training_df['text'].tolist() + validation_df['text'].tolist())

    # data tokenization
    print('\nStart data tokenization ...')
    X_train = NLP.tokenization(word2id, training_df['text'].tolist(), max_len)
    X_valid = NLP.tokenization(word2id, validation_df['text'].tolist(), max_len)

    # labels one hot encoding
    onehot_encoder = OneHotEncoder(sparse_output=False)
    y_train = onehot_encoder.fit_transform(np.array(training_df['label'].tolist()).reshape(-1, 1))
    y_valid = onehot_encoder.fit_transform(np.array(validation_df['label'].tolist()).reshape(-1, 1))

    print('The shape of training dataset is', X_train.shape)
    print('The shape of validation dataset is', X_valid.shape)

    print('\nThe shape of training label is', y_train.shape)
    print('The shape of validation label is', y_valid.shape)

    # build BILSTM model
    print('\nBILSTM model')
    model_BILSTM = models.Sequential()
    model_BILSTM.add(layers.Embedding(input_dim=len(word2id), output_dim=100, input_length=max_len))
    model_BILSTM.add(layers.Dense(128, activity_regularizer=keras.regularizers.L2(0.01)))
    model_BILSTM.add(layers.Bidirectional(keras.layers.LSTM(units=64, return_sequences=True),
                                          activity_regularizer=keras.regularizers.L2(0.01)))
    model_BILSTM.add(layers.Dropout(0.2))
    model_BILSTM.add(
        layers.Bidirectional(keras.layers.LSTM(units=32, activity_regularizer=keras.regularizers.L2(0.01))))
    model_BILSTM.add(layers.Dense(6, activation='softmax'))
    model_BILSTM.summary()

    # compile model
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]

    model_BILSTM.compile(loss=keras.losses.categorical_crossentropy,
                         optimizer=keras.optimizers.Adam(0.001),
                         metrics=['accuracy'])

    # training model
    print('\nStart training')
    start = time.time()
    history_BILSTM = model_BILSTM.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=50,
                                      callbacks=callbacks)
    end = time.time()

    print('\nTime for training BILSTM model is: %.2f' % (end - start) + 's')

    # plot learning curve
    learning_curve_BILSTM, train_BILSTM, val_BILSTM = NLP.plot_learning_curve(history_BILSTM, 'BILSTM')
    train_BILSTM = round(train_BILSTM, 4)
    val_BILSTM = round(val_BILSTM, 4)
    learning_curve_BILSTM.savefig(os.path.join(save_path, '4 Learn Curve of BILSTM.png'))

    # prepare test data
    print('\nStart data tokenization of test data...')
    X_test = NLP.tokenization(word2id, test_df['text'].tolist(), max_len)
    y_test = test_df['label'].tolist()

    # show the shape of test data and test label
    print('The shape of validation dataset is', X_test.shape)
    print('The length of training label is', len(y_test))

    # evaluate test accuracy
    print('\nStart evaluation')
    y_pred_BILSTM = np.argmax(model_BILSTM.predict(X_test), axis=1)
    test_BILSTM = accuracy_score(y_test, y_pred_BILSTM)
    test_BILSTM = round(test_BILSTM, 4)
    print('The Test accuracy of BILSTM is %.4f' % test_BILSTM)

    # plot confusion matrix
    CM_BILSTM = pd.DataFrame(confusion_matrix(y_test, y_pred_BILSTM), index=category, columns=category)
    print('\nConfusion Matrix of BILSTM:')
    print(CM_BILSTM)

    print('\nClassification Report of BILSTM:')
    print(classification_report(y_test, y_pred_BILSTM))

    confusion_matrix_BILSTM = NLP.plot_CM_matrix(CM_BILSTM, 'Confusion Matrix of BILSTM')
    confusion_matrix_BILSTM.savefig(os.path.join(save_path, '5 Confusion Matrix of BILSTM.png'))

    # save results
    test_df['pred_BILSTM'] = y_pred_BILSTM
    test_df['pred_BILSTM_discription'] = test_df['pred_BILSTM'].map(labels_dict)
    test_df.to_csv(os.path.join(save_path, 'result_D.csv'))

    return train_BILSTM, val_BILSTM, test_BILSTM
