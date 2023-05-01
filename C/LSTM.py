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

#Task C main code
def run_C():
    print('\n\n******************************Running Task C: Long Short-Term Memory******************************')

    print('\nThe current path of Task C is', os.getcwd())

    # create saving path
    save_path = './Results/C'
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
    print('\nStart data tokenization of test data...')
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

    # build LSTM model
    print('\nLSTM model:')
    model_LSTM = models.Sequential()
    model_LSTM.add(layers.Embedding(input_dim=len(word2id), output_dim=100, input_length=max_len))
    model_LSTM.add(layers.LSTM(units=64, return_sequences=True, activity_regularizer=keras.regularizers.L2(0.01)))
    model_LSTM.add(layers.Dropout(0.2))
    model_LSTM.add(layers.LSTM(units=32, activity_regularizer=keras.regularizers.L2(0.01)))
    model_LSTM.add(layers.Dense(6, activation='softmax'))
    model_LSTM.summary()

    # compile model
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]

    model_LSTM.compile(loss=keras.losses.categorical_crossentropy,
                       optimizer=keras.optimizers.Adam(0.001),
                       metrics=['accuracy'])

    # training model
    print('\nStart training')
    start = time.time()
    history_LSTM = model_LSTM.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=50, callbacks=callbacks)
    end = time.time()

    print('\nTime for training LSTM model is: %.2f' % (end - start) + 's')

    # plot learning curve
    learning_curve_LSTM, train_LSTM, val_LSTM = NLP.plot_learning_curve(history_LSTM, 'LSTM')
    train_LSTM = round(train_LSTM, 4)
    val_LSTM = round(val_LSTM, 4)
    learning_curve_LSTM.savefig(os.path.join(save_path, '4 Learn Curve of LSTM.png'))

    # prepare test data
    print('\nStart data tokenization ...')
    X_test = NLP.tokenization(word2id, test_df['text'].tolist(), max_len)
    y_test = test_df['label'].tolist()

    # show the shape of test data and test label
    print('The shape of validation dataset is', X_test.shape)
    print('The length of training label is', len(y_test))

    # evaluate test accuracy
    print('\nStart evaluation')
    y_pred_LSTM = np.argmax(model_LSTM.predict(X_test), axis=1)
    test_LSTM = accuracy_score(y_test, y_pred_LSTM)
    test_LSTM = round(test_LSTM, 4)
    print('The Test accuracy of BILSTM is %.4f' % test_LSTM)

    # plot confusion matrix
    CM_LSTM = pd.DataFrame(confusion_matrix(y_test, y_pred_LSTM), index=category, columns=category)
    print('\nConfusion Matrix of LSTM:')
    print(CM_LSTM)

    print('\nClassification Report of LSTM:')
    print(classification_report(y_test, y_pred_LSTM))

    confusion_matrix_LSTM = NLP.plot_CM_matrix(CM_LSTM, 'Confusion Matrix of LSTM')
    confusion_matrix_LSTM.savefig(os.path.join(save_path, '5 Confusion Matrix of LSTM.png'))

    # save results
    test_df['pred_LSTM'] = y_pred_LSTM
    test_df['pred_LSTM_discription'] = test_df['pred_LSTM'].map(labels_dict)
    test_df.to_csv(os.path.join(save_path, 'result_C.csv'))

    return train_LSTM, val_LSTM, test_LSTM
