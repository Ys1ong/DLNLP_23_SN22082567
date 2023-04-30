import re
import string

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import learning_curve, TimeSeriesSplit

def plot_distribution(data, label, data_name):
    # Count the number of occurrences of each category
    counts = data[label].value_counts()
    
    # Create the canvas and subplots
    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 5))

    # Plot the first subplot
    sns.barplot(x = counts.index, y = counts.values, ax = axs[0])
    axs[0].set_xlabel('Category', fontproperties = 'Times New Roman', size = 14)
    axs[0].set_ylabel('Count', fontproperties = 'Times New Roman', size = 14)
    axs[0].set_title('Bar Chart', fontproperties = 'Times New Roman', size = 15)

    # Plot the second subplot
    axs[1].pie(counts, labels = counts.index, autopct = '%1.2f%%',
               textprops = {'fontsize': 14, 'family': 'Times New Roman'})
    axs[1].set_title('Pie Chart', fontproperties = 'Times New Roman', size = 15)

    # Display the plots
    fig.suptitle('Label Distribution of ' + data_name + ' Data', fontproperties = 'Times New Roman', size = 16)
    plt.show()
    
    # Print the count and proportion of each category
    for category, count in counts.items():
        proportion = count / len(data)
        print(f'{category}: {count} ({proportion:.2%})')
    
    return fig
        
def process_DT(data):
    print('Remove all URL links https?:\/\/\S+', )
    print('Removing all punctuation', string.punctuation)
    print('Convert all letters to lowercase')
    
    processed = []
    for tweet in tqdm(data):

        tweet = re.sub(r'https?:\/\/\S+', '', tweet)
        tweet = tweet.translate(str.maketrans('', '', string.punctuation))
        tweet = tweet.lower()

        processed.append(tweet)
    
    return processed

def process_data(data):
    
    data = re.sub(r'https?:\/\/\S+', '', data)
    data = data.translate(str.maketrans('', '', string.punctuation))
    data = data.lower()
        
    tokenizer = TweetTokenizer(preserve_case = False, strip_handles = True, reduce_len = True)
    data_tokens = tokenizer.tokenize(data)
    return data_tokens

def get_vocab(data):
    
    word2id = {'<PAD>': 0, '<UNK>': 1, '<EOS>': 2}
    
    max_len = 0
    for tweet in data:
        processed_tweet = process_data(tweet)
        if len(processed_tweet) > max_len:
            max_len = len(processed_tweet)

        for word in processed_tweet:
            if word not in word2id :
                word2id[word] = len(word2id)
                
    return word2id, max_len

def tokenization(word2id, tweets, max_len):
    
    numericalized_tweets = []
    
    for tweet in tqdm(tweets):
        processed_tweet = process_data(tweet)
        numericalized_tweet = []
        
        for word in processed_tweet:
            if word in word2id:
                numericalized_tweet.append(word2id[word])
            else:
                numericalized_tweet.append(word2id['<UNK>'])
                
        if len(numericalized_tweet) > max_len:
            numericalized_tweet = numericalized_tweet[:max_len]
        else:
            while len(numericalized_tweet) < max_len:
                numericalized_tweet.append(word2id['<PAD>'])
            
        numericalized_tweets.append(numericalized_tweet)
        
    return np.array(numericalized_tweets)

def plot_learning_curve_DT(estimator, title, X, y, cv, ylim = [0.5, 1.0], n_jobs = 1, train_sizes = np.linspace(.1, 1.0, 5)):
    fig = plt.figure(figsize = (8, 6))
    plt.title(title, fontproperties = 'Times New Roman', size = 16)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples", fontproperties = 'Times New Roman', size = 15)
    plt.ylabel("Score", fontproperties = 'Times New Roman', size = 15)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv = cv, n_jobs = n_jobs, train_sizes = train_sizes)
    train_scores_mean = np.mean(train_scores, axis = 1)
    train_scores_std = np.std(train_scores, axis = 1)
    test_scores_mean = np.mean(test_scores, axis = 1)
    test_scores_std = np.std(test_scores, axis = 1)
    plt.grid()
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha = 0.1, color = 'r')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha = 0.1, color = 'g')
    plt.plot(train_sizes, train_scores_mean, 's--', color = 'r', label = 'Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color = 'g', label = 'Cross-Validation score')
    
    plt.legend(prop={'family': 'Times New Roman', 'size': 15})
    plt.show()
    
    print('The Training Accuracy of decision tree is %.4f' %train_scores_mean[-1])
    print('The Validation Accuracy of decision tree is %.4f' %test_scores_mean[-1])
    
    return fig

def plot_learning_curve(hisroty, model_name):
    fig = plt.figure(figsize = (8, 6))
    plt.plot(hisroty.history['accuracy'], label = 'training accuracy')
    plt.plot(hisroty.history['val_accuracy'], label = 'validation accuracy')
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.xlabel('Epoch', fontproperties = 'Times New Roman', size = 15)
    plt.ylabel('Accuracy', fontproperties = 'Times New Roman', size = 15)
    plt.title('Learning Curve of ' + model_name, fontproperties = 'Times New Roman', size = 16)
    plt.legend(loc = 'lower right', prop = {'family': 'Times New Roman', 'size': 15})
    plt.grid()
    plt.show()

    print('The Training Accuracy of ' + model_name + ' is %.4f' %hisroty.history['accuracy'][-1])
    print('The Validation Accuracy of ' + model_name + ' is %.4f' %hisroty.history['val_accuracy'][-1])
    return fig

def plot_CM_matrix(matrix, title):
    fig = plt.figure(figsize = (6, 6))
    sns.heatmap(matrix, annot = True, annot_kws = {"size": 16}, fmt = 'g', cmap = 'binary', 
                vmin = 0, vmax = 100, cbar = False, linecolor = 'black', linewidths = 0.5)
    plt.xlabel('Predicted Label', fontproperties = 'Times New Roman', size = 15)
    plt.ylabel('True Label', fontproperties = 'Times New Roman', size = 15)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.title(title, fontproperties = 'Times New Roman', size = 16)
    plt.show()
    
    return fig
