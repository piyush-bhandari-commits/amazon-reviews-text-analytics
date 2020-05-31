import nltk
nltk.download('popular')
import re
from sklearn.utils import shuffle
import random
import numpy as np













if __name__ == '__main__':
    
    neg_raw_data = open('/content/drive/My Drive/Colab Notebooks/MSCI 641 Text Analytics/neg.txt').read()
    pos_raw_data = open('/content/drive/My Drive/Colab Notebooks/MSCI 641 Text Analytics/pos.txt').read()
    