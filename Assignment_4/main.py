import numpy as np
import re
import sys
import os
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, Dropout


def get_labels(list_data):
    label = [0] * round(len(list_data) / 2) + [1] * round(len(list_data) / 2)
    return label


def read_file(data_path):
    with open(data_path) as f:
        data = f.readlines()
        regex_data = [re.findall(r'\w+', line.lower()) for line in data]
        clean_data = [" ".join(line) for line in regex_data]
    return clean_data


def create_model(word_model, embedding_matrix, x_train_pad, y_train, hidden_layer_activation):
    batch = 32
    epochs = 10
    model = Sequential()
    model.add(Embedding(len(word_model.wv.vocab) + 1, 350,
                        input_length=x_train_pad.shape[1],
                        weights=[embedding_matrix],
                        trainable=False))
    model.add(Flatten())
    model.add(Dense(2, activation=hidden_layer_activation))
    # model.add(Dropout(0.1))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['acc'])
    model.fit(x_train_pad, y_train, batch, epochs)
    return model


def main(dir_path):
    print('\n Reading data...\n')
    # x_train = read_file('../Assignment_1/data'
    #                     '/train_data_with_stopwords.csv')
    x_train = read_file(os.path.join(dir_path, 'train_data_with_stopwords.csv'))
    y_train = get_labels(x_train)
    y_train = to_categorical(np.asarray(y_train))

    # x_valid = read_file('../Assignment_1/data'
    #                     '/valid_data_with_stopwords.csv')
    x_valid = read_file(os.path.join(dir_path, 'valid_data_with_stopwords.csv'))
    y_valid = get_labels(x_valid)
    y_valid = to_categorical(np.asarray(y_valid))

    print('\n Loading Word2Vec model...\n')
    word_model = Word2Vec.load('../Assignment_3/data/w2v_min_count_1_350.model')

    print('\n Creating the embedding matrix...\n')
    embedding_matrix = np.zeros((len(word_model.wv.vocab) + 1, 350))
    for i, vec in enumerate(word_model.wv.vectors):
        embedding_matrix[i] = vec
    # print('\n Embedding Matrix Shape: {}\n'.format(embedding_matrix.shape))
    print('\n Tokenizing and padding the data...\n')

    features = 200
    tokenizer = Tokenizer(num_words=features)
    tokenizer.fit_on_texts(x_train)

    word_index = tokenizer.word_index
    x_train_seq = tokenizer.texts_to_sequences(x_train)
    x_train_pad = pad_sequences(x_train_seq)

    x_valid_seq = tokenizer.texts_to_sequences(x_valid)
    x_valid_pad = pad_sequences(x_valid_seq)

    print('\n Training model with relu activation...\n')
    hidden_layer_activation = 'relu'
    nn_model = create_model(word_model, embedding_matrix, x_train_pad, y_train, hidden_layer_activation)
    nn_model.save('data/nn_relu.h5')

    print('\n Training model with sigmoid activation...\n')
    hidden_layer_activation = 'sigmoid'
    nn_model = create_model(word_model, embedding_matrix, x_train_pad, y_train, hidden_layer_activation)
    nn_model.save('data/nn_sigmoid.h5')
    print('\n Done...\n')

    print('\n Training model with tanh activation...\n')
    hidden_layer_activation = 'tanh'
    nn_model = create_model(word_model, embedding_matrix, x_train_pad, y_train, hidden_layer_activation)
    nn_model.save('data/nn_tanh.h5')
    print('\n Done...\n')

    return


if __name__ == '__main__':
    main(sys.argv[1])
