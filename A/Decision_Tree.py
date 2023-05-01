# import library
import os
import time

import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn import tree
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

#Task A main code
def run_A():
    print('\n******************************Running Task A: Decision Tree******************************')

    print('\nThe current path of Task A is', os.getcwd())
    # create saving path
    save_path = './Results/A'
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

    # combined training and validation datasets
    combined_df = pd.concat([training_df, validation_df], ignore_index=True)

    # process and cleaning combined dataset
    print('\nProcess and cleaning training and validation dataset')
    combined_df['text'] = NLP.process_DT(combined_df['text'])

    # split combined dataset into training and testing datasets
    cv = TimeSeriesSplit(n_splits=5, test_size=2000)

    for train_index, valid_index in cv.split(combined_df):
        training_df = combined_df.iloc[train_index]
        validation_df = combined_df.iloc[valid_index]

    # vectorize the text using TF-IDF
    vectorizer = TfidfVectorizer()
    X_train_valid = vectorizer.fit_transform(combined_df['text'])
    X_train = vectorizer.transform(training_df['text'])
    X_valid = vectorizer.transform(validation_df['text'])

    y_train_valid = combined_df['label']
    y_train = training_df['label']
    y_valid = validation_df['label']

    print('\nThe shape of combined dataset is', X_train_valid.shape)
    print('The shape of training dataset is', X_train.shape)
    print('The shape of validation dataset is', X_valid.shape)

    print('\nThe shape of combined label is', y_train_valid.shape)
    print('The shape of training label is', y_train.shape)
    print('The shape of validation label is', y_valid.shape)

    # build decision tree model
    def TreeClassifier(X_train, y_train, X_test, k):

        tree_params = {'criterion': 'entropy', 'min_samples_split': k}
        clf = tree.DecisionTreeClassifier(**tree_params)

        clf.fit(X_train, y_train)
        Y_pred = clf.predict(X_test)
        return Y_pred

    # hyper-parameter tunning of minimum samples split
    score_list_tree = []
    limit = range(55, 305, 5)

    print('\nHyper-parameter tunning of minimum samples split')
    for i in tqdm(limit):
        Y_pred_Tree = TreeClassifier(X_train, y_train, X_valid, i)
        score = accuracy_score(y_valid, Y_pred_Tree)
        score_list_tree.append(score)

    # find the highest validation accuracy
    validation_accuracy = plt.figure(figsize=(8, 6))
    plt.plot(limit, score_list_tree, 'b-o', linewidth=1.5)
    plt.xlabel('Minimum Samples Split', fontproperties='Times New Roman', size=15)
    plt.ylabel('Validation Accuracy', fontproperties='Times New Roman', size=15)
    plt.title('Validation Accuracy of Different Minimum Samples Split', fontproperties='Times New Roman', size=16)
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.grid()
    plt.show()
    validation_accuracy.savefig(os.path.join(save_path, '4 Validation Accuracy of Different Minimum Samples Split.png'))

    print('\n')
    best_score = max(score_list_tree)
    split = []
    for j in range(len(score_list_tree)):
        if score_list_tree[j] == best_score:
            split.append(limit[j])
            print('When Samples Split is ' + str(limit[j]) + ', Validation Accuracy has the Highest Value')
    print('\nChoose the Minimum Sample:', split[0], 'and plot learning curve...')

    # plot learning curve of the minimum samples split with the highest validation accuracy
    tree_params = {'criterion': 'entropy', 'min_samples_split': split[0]}
    DT_model = tree.DecisionTreeClassifier(**tree_params)
    DT_learning_curve, train_DT, val_DT = NLP.plot_learning_curve_DT(DT_model, 'Learn Curve of Decision Tree', X_train_valid,
                                                   y_train_valid, cv)
    train_DT = round(train_DT, 4)
    val_DT = round(val_DT, 4)
    DT_learning_curve.savefig(os.path.join(save_path, '5 Learn Curve of Decision Tree.png'))

    # prepare test data
    print('\nPrepare test data')
    test_df['text'] = NLP.process_DT(test_df['text'])
    X_test = vectorizer.transform(test_df['text'])
    y_test = test_df['label']

    # show the shape of test data and test label
    print('The shape of test dataset is', X_test.shape)
    print('The shape of test label is', y_test.shape)

    # evaludte test accuracy
    start = time.time()
    y_pred_DT = TreeClassifier(X_train, y_train, X_test, split[0])
    end = time.time()
    test_DT = accuracy_score(y_test, y_pred_DT)
    test_DT = round(test_DT, 4)
    print('\nTime for training decision tree model is: %.2f' % (end - start) + 's')

    print('The Test Accuracy of Decision Tree is %.4f' % test_DT)

    # plot confusion matrix
    CM_DT = pd.DataFrame(confusion_matrix(y_test, y_pred_DT), index=category, columns=category)
    print('\nConfusion Matrix of Decision Tree:')
    print(CM_DT)

    print('\nClassification Report of Decision Tree:')
    print(classification_report(y_test, y_pred_DT))

    DT_confusion_matrix = NLP.plot_CM_matrix(CM_DT, 'Confusion Matrix of Decision Tree')
    DT_confusion_matrix.savefig(os.path.join(save_path, '6 Confusion Matrix of Decision Tree.png'))

    # save results
    test_df['pred_DT'] = y_pred_DT
    test_df['pred_DT_discription'] = test_df['pred_DT'].map(labels_dict)
    test_df.to_csv(os.path.join(save_path, 'result_A.csv'))

    return train_DT, val_DT, test_DT
