import re
import sys
from gensim.models import Word2Vec
import numpy as np
from pprint import pprint


def read_file(data_path):
    """
    Method to read csv from the data folder in assignment 1
    """
    with open(data_path) as f:
        data = f.readlines()
        regex_data = [re.findall(r'\w+', line.lower()) for line in data]
        clean_data = [" ".join(line) for line in regex_data]
    return clean_data


def main(path_sample_txt):
    np.random.seed(42)
    sample_data = read_file(path_sample_txt)
    pprint(sample_data)

    # Loading the word2vec model form data folder
    w2v_model = Word2Vec.load('data/w2v.model')
    all_data = {}
    for word in sample_data:
        similar_words = w2v_model.wv.most_similar(positive=[word], topn=20)
        all_data[word] = similar_words

    pprint(all_data)


if __name__ == '__main__':
    main(sys.argv[1])
