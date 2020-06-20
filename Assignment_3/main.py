import re
import sys
import os
from gensim.models import Word2Vec
import numpy as np
from pprint import pprint


def read_file(data_path):
    with open(data_path) as f:
        data = f.readlines()
        regex_data = [re.findall(r'\w+', line.lower()) for line in data]
        # clean_data = [" ".join(line) for line in regex_data]
    return regex_data


def main(dir_path):

    np.random.seed(42)

    data = {
            'negative': read_file(os.path.join(dir_path, 'neg.txt')),
            'positive': read_file(os.path.join(dir_path, 'pos.txt'))
            }

    all_data = data['negative'] + data['positive']
    print('\n Data read successfully from files... \n')

    print('\n Model training in progress... \n')
    model = Word2Vec(all_data, min_count=1, size=50, workers=3, window=3, sg=1, seed=42)
    model.save('data/w2v.model')
    print('\n Model saved ... \n')

    print('\n Most similar words to "good" are... \n')
    good_similar = model.wv.most_similar(positive=['good'], topn=20)
    pprint(good_similar)

    print('\n Most similar words to "bad" are... \n')
    bad_similar = model.wv.most_similar(positive=['bad'], topn=20)
    pprint(bad_similar)


if __name__ == '__main__':
    main(sys.argv[1])
