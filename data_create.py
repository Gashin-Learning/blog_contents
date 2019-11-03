import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import pickle

import nltk
# Please download for pos_tag
#nltk.download('averaged_perceptron_tagger')
#nltk.download('punkt')

def create_data():
    vectorizer = CountVectorizer(min_df=0.005, max_df=0.1, stop_words="english")

    categories = ['rec.sport.baseball', 'talk.religion.misc',
                  'comp.graphics', 'sci.space']
    # read data
    newsgroups_train = fetch_20newsgroups(categories=categories, shuffle=True, remove=('headers', 'footers', 'quotes'))
    X = vectorizer.fit_transform(newsgroups_train.data).toarray()
    X_words = np.array(vectorizer.get_feature_names())
    y = newsgroups_train.target

    # select only "Noun" or "Proper Noun"
    POS = [nltk.pos_tag(word)[0][1] for word in X_words]
    cond_NN = np.array(list(map(lambda x:True if x=='NN' or x=="NNP" or x=="NNS" else False, POS)))
    X_cond = X[:,cond_NN]
    X_words_cond = np.array(X_words)[cond_NN]

    # select the documents which has at least 1 word.
    cond_d_has_1word = np.where(np.sum(X_cond, axis=1)>0)[0]
    X_cond = X_cond[cond_d_has_1word,:]
    y = y[cond_d_has_1word]

    print('X: (documents, words) ',X_cond.shape)

    return X_cond, X_words_cond, y
