# Assignment 2 Report

The following sections of the report highlight the accuracies obtained from the experiments run and the reason for the performance.

## Accuracy on Test Set
| Stopwords Removed | Text Features | Accuracy |
| :-------------: |:-------------:| :-----:|
| Yes      | Unigrams | 0.8041 |
| Yes      | Bigrams  | 0.7879 |
| Yes      | Unigrams + Bigrams | 0.8217 |
| No      | Unigrams | 0.8044 |
| No      | Bigrams  | 0.8212 |
| No      | Unigrams + Bigrams |0.8298|

## Analysis

**1. Which condition performed better: with or without stopwords?**

The model performed better when the stopwords are retained in the corpus. The reason for this can be attributed to meaning captured by the stopwords in the reviews. When we removed stopwords from the text, the reviews would have become similar. For Example: 'I didn't like the product' (Negative). When we remove stopwords from the review, it becomes 'like product', which the model might classify as a positive review. For this reason, the data with stopwords captured more meaning from the reviews, which lead to higher accuracy across all text features.  

**2. Which condition performed better: unigrams, bigrams or unigrams+bigrams?**

The model accuracy for unigram+bigram features is highest for both corpus with stopwords and corpus without stopwords. The reason for this can be attrubuted to higher number of features for classification. When we consider both unigram and bigram tokens for classification, alongwith the individual word occurances, we capture meaning with bigram tokens. For Example : There might be number of reviews with bigram 'good product', which is a helpful feature for model for classification alongwith unigram 'good'.
 
