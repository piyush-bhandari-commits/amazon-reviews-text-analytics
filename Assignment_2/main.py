import numpy as np
from pprint import pprint
import os 
import sys
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def read_csv(data_path):
  with open(data_path) as f:
    data = f.readlines()
    regex_data = [re.findall(r'\w+',line) for line in data]
    clean_data = [" ".join(line) for line in regex_data]
  return clean_data


def get_labels(list_data):
  label = [0]*round(len(list_data)/2) + [1]*round(len(list_data)/2)
  return label


def train_classifier(model, vect, x_train, y_train, x_valid, y_valid):

  # count_vect = CountVectorizer(min_df=2, ngram_range=ngram_range)

  vectorizer = vect.fit(x_train)

  x_train_vect = vectorizer.transform(x_train)
  x_valid_vect = vectorizer.transform(x_valid)

  trained_model = model.fit(x_train_vect, y_train)
  predictions = trained_model.predict(x_valid_vect)

  metrics = {
        'Accuracy': accuracy_score(y_valid, predictions),
        'Precision': precision_score(y_valid, predictions),
        'Recall': recall_score(y_valid, predictions),
        'F1': f1_score(y_valid, predictions),
    }
  return metrics


def main(dir_path):
    
    print ('\n Checking path to assignment 1 data...\n ')
    print (dir_path)
    
    print ('\n Reading the CSV files with stopwords...\n')

    data_with_stopwords = {}
    data_with_stopwords['train'] = read_csv(os.path.join(dir_path, 'train_data_with_stopwords.csv'))
    data_with_stopwords['valid'] = read_csv(os.path.join(dir_path, 'valid_data_with_stopwords.csv'))
    data_with_stopwords['test'] = read_csv(os.path.join(dir_path, 'test_data_with_stopwords.csv'))
    data_with_stopwords['train_label'] = get_labels(data_with_stopwords['train'])
    data_with_stopwords['valid_label'] = get_labels(data_with_stopwords['valid'])
    data_with_stopwords['test_label'] = get_labels(data_with_stopwords['test'])

    print('\n Reading the CSV files without stopwords...\n')

    data_without_stopwords = {}
    data_without_stopwords['train'] = read_csv(os.path.join(dir_path, 'train_data_without_stopwords.csv'))
    data_without_stopwords['valid'] = read_csv(os.path.join(dir_path, 'valid_data_without_stopwords.csv'))
    data_without_stopwords['test'] = read_csv(os.path.join(dir_path, 'test_data_without_stopwords.csv'))
    data_without_stopwords['train_label'] = get_labels(data_without_stopwords['train'])
    data_without_stopwords['valid_label'] = get_labels(data_without_stopwords['valid'])
    data_without_stopwords['test_label'] = get_labels(data_without_stopwords['test'])

    print ('\n Reading CSV files completed...\n')

    print ('\n Training classifier with stopwords data...\n')

    classifier = MultinomialNB()
    result = dict()

    print('\n-------Unigram Features---------\n')
    unigram_vect_with_stopwords = CountVectorizer(min_df=2 , ngram_range=(1,1))
    result['unigram_with_stopwords'] = train_classifier(classifier, unigram_vect_with_stopwords,
                                                    data_with_stopwords['train'], 
                                                    data_with_stopwords['train_label'],
                                                    data_with_stopwords['test'],
                                                    data_with_stopwords['test_label'])
    print('\n-------Bigram Features---------\n')
    bigram_vect_with_stopwords = CountVectorizer(min_df=2, ngram_range=(2, 2))
    result['bigram_with_stopwords'] = train_classifier(classifier, bigram_vect_with_stopwords,
                                                       data_with_stopwords['train'],
                                                       data_with_stopwords['train_label'],
                                                       data_with_stopwords['test'],
                                                       data_with_stopwords['test_label'])
    print('\n-------Both Unigram-Bigram Features---------\n')
    unigram_bigram_vect_with_stopwords = CountVectorizer(min_df=2, ngram_range=(1, 2))
    result['unigram_bigram_with_stopwords'] = train_classifier(classifier, unigram_bigram_vect_with_stopwords,
                                                               data_with_stopwords['train'],
                                                               data_with_stopwords['train_label'],
                                                               data_with_stopwords['test'],
                                                               data_with_stopwords['test_label'])

    print('\n Completed training classifier with stopwords data...\n')

    print('\n Training classifier without stopwords data...\n')

    print('\n-------Unigram Features---------\n')
    unigram_vect_without_stopwords = CountVectorizer(min_df=2, ngram_range=(1, 1))
    result['unigram_without_stopwords'] = train_classifier(classifier, unigram_vect_without_stopwords,
                                                        data_without_stopwords['train'],
                                                        data_without_stopwords['train_label'],
                                                        data_without_stopwords['test'],
                                                        data_without_stopwords['test_label'])
    print('\n-------Bigram Features---------\n')
    bigram_vect_without_stopwords = CountVectorizer(min_df=2, ngram_range=(2, 2))
    result['bigram_without_stopwords'] = train_classifier(classifier, bigram_vect_without_stopwords,
                                                       data_without_stopwords['train'],
                                                       data_without_stopwords['train_label'],
                                                       data_without_stopwords['test'],
                                                       data_without_stopwords['test_label'])
    print('\n-------Both Unigram-Bigram Features---------\n')
    unigram_bigram_vect_without_stopwords = CountVectorizer(min_df=2, ngram_range=(1, 2))
    result['unigram_bigram_without_stopwords'] = train_classifier(classifier, unigram_bigram_vect_without_stopwords,
                                                               data_without_stopwords['train'],
                                                               data_without_stopwords['train_label'],
                                                               data_without_stopwords['test'],
                                                               data_without_stopwords['test_label'])

    print('\n Completed training classifier without stopwords data...\n')

    pprint(result)

if __name__ == "__main__":
    main(sys.argv[1])