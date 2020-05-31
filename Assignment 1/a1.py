import nltk
nltk.download('popular')
import re
from sklearn.utils import shuffle
import random
import numpy as np


def remove_punct(text):
    text_nopunct = "".join([char for char in text if char not in '\',!"#$%&()*+/:;<=>@[\\]^`{|}~\t\n']) 
    return text_nopunct


def tokenize(line):
    tokens = re.split('\s+', line)
    return tokens

def remove_stopwords(tokenized_list):
    stopword = nltk.corpus.stopwords.words('english')
    text = [word for word in tokenized_list if word not in stopword]
    return text


def create_split (dataset):

    train_data = []
    validation_data = []
    test_data = []

    test_size= int (0.10*len(dataset))
    validate_size= int (0.10*len(dataset))
    train_size= int (0.80*len(dataset))

    list_indices = list(range(len(dataset)))
    shuffled_list_indices = list(shuffle(list_indices))
    train_indices = random.sample(shuffled_list_indices, train_size)
    #rest_of_data = [index for index in shuffled_list_indices if index not in train_indices]
    rest_of_data = list(set(shuffled_list_indices) - set(train_indices))
    validation_indices = random.sample(rest_of_data, validate_size)
    #test_indices = [index for index in rest_of_data if index not in validation_indices]
    test_indices = list(set(rest_of_data) - set(validation_indices))

    for index in train_indices:
      train_data.append(dataset[index])
    
    for index in validation_indices:
      validation_data.append(dataset[index])

    for index in test_indices:
      test_data.append(dataset[index])

    return train_data, validation_data, test_data

def create_csv(data, name):
  np.savetxt('{}.csv'.format(name), np.array(data) , delimiter=",", fmt='%s')


def main():
    
    print ("Reading text files...")
    neg_raw_data = open('/Users/piyushbhandari/Desktop/msci-text-analytics-s20/Assignment_1/neg.txt').read()
    pos_raw_data = open('/Users/piyushbhandari/Desktop/msci-text-analytics-s20/Assignment_1/pos.txt').read()
    
    print ("\n Splitting at new line...")
    neg_list = neg_raw_data.split("\n")
    pos_list = pos_raw_data.split("\n")

    print ("\n Removing Special Characters...")
    clean_neg = [remove_punct(text.lower()) for text in neg_list]
    clean_pos = [remove_punct(text.lower()) for text in pos_list]

    print ("\n Tokenizing the corpus...")
    neg_tokens_with_stopwords = [tokenize(line) for line in clean_neg]
    pos_tokens_with_stopwords = [tokenize(line) for line in clean_pos]

    print ("\n Removing the stopwords...")
    neg_tokens_without_stopwords = [remove_stopwords(line) for line in neg_tokens_with_stopwords]
    pos_tokens_without_stopwords = [remove_stopwords(line) for line in pos_tokens_with_stopwords]

    print ("\n Splitting the dataset into train, validation and test...")
    dataset = neg_tokens_with_stopwords
    neg_train_with_stopwords, neg_valid_with_stopwords, neg_test_with_stopwords = create_split(dataset)

    dataset = pos_tokens_with_stopwords
    pos_train_with_stopwords, pos_valid_with_stopwords, pos_test_with_stopwords = create_split(dataset)

    dataset = neg_tokens_without_stopwords
    neg_train_without_stopwords, neg_valid_without_stopwords, neg_test_without_stopwords = create_split(dataset)

    dataset = pos_tokens_without_stopwords
    pos_train_without_stopwords, pos_valid_without_stopwords, pos_test_without_stopwords = create_split(dataset)

    print ("\n Create CSV Files...")
    create_csv(neg_train_with_stopwords, 'neg_train_with_stopwords')
    create_csv(neg_valid_with_stopwords, 'neg_valid_with_stopwords')
    create_csv(neg_test_with_stopwords, 'neg_test_with_stopwords')

    create_csv(pos_train_with_stopwords, 'pos_train_with_stopwords')
    create_csv(pos_valid_with_stopwords, 'pos_valid_with_stopwords')
    create_csv(pos_test_with_stopwords, 'pos_test_with_stopwords')

    create_csv(neg_train_without_stopwords, 'neg_train_without_stopwords')
    create_csv(neg_valid_without_stopwords, 'neg_valid_without_stopwords')
    create_csv(neg_test_without_stopwords, 'neg_test_without_stopwords')

    create_csv(pos_train_without_stopwords, 'pos_train_without_stopwords')
    create_csv(pos_valid_without_stopwords, 'pos_valid_without_stopwords')
    create_csv(pos_test_without_stopwords, 'pos_test_without_stopwords')

if __name__ == '__main__':
    main()
    
