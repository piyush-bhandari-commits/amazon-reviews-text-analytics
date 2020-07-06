import re
import sys
import numpy as np
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def read_file(data_path):
    """
    Method to read csv from the data folder in assignment 1
    """
    with open(data_path) as f:
        data = f.readlines()
        regex_data = [re.findall(r'\w+', line.lower()) for line in data]
        clean_data = [" ".join(line) for line in regex_data]
    return clean_data


def main(path_sample_txt, model_activation):
    np.random.seed(42)
    x_test = read_file(path_sample_txt)

    x_train = read_file('../Assignment_1/data/train_data_with_stopwords.csv')
    print('Train data loaded successfully..')

    # Loading the word2vec model form data folder
    model = load_model('data/nn_{}.h5'.format(model_activation))
    print('Model loaded successfully...')

    print('Tokenizing sample data..')
    features = 200
    tokenizer = Tokenizer(num_words=features)
    tokenizer.fit_on_texts(x_train)

    x_test_seq = tokenizer.texts_to_sequences(x_test)
    x_test_pad = pad_sequences(x_test_seq, maxlen=25)
    print (x_test_pad.shape)

    print('Predicting labels..')
    y_prob = model.predict(x_test_pad)
    y_predict = [np.argmax(item) for item in y_prob]
    print (y_predict)
    print('Done...')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
