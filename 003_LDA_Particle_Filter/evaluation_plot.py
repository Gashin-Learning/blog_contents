import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from wordcloud import WordCloud

def plot_wordcloud(n_kv, news_process1_words, topic_num):
    fig, ax = plt.subplots(1, topic_num, figsize=(12, 6))
    for cate in range(topic_num):
        category_words = news_process1_words[(-n_kv[:,cate]).argsort()][:20]

        text=''
        word_freqency = np.repeat(category_words, (np.sort(n_kv[:,cate])[::-1][:20]*100).astype('int'))
        np.random.shuffle(word_freqency)
        for word in word_freqency:
            text+=' ' + word

        wordcloud = WordCloud(width=480, height=320).generate(text)

        ax[cate].imshow(wordcloud, interpolation='bilinear')
        ax[cate].set_title('topic{}'.format(cate), fontsize=24)
    plt.tight_layout()


def plot_tSNE_plot(n_dk_for_evaluation, y):

    n_dk_decrease = TSNE(n_components=2, random_state=0).fit_transform(n_dk_for_evaluation)
    fig,ax=plt.subplots(figsize=(9,6))
    plt.scatter(x=n_dk_decrease[:,0], y=n_dk_decrease[:,1], c=y, marker='.', alpha=.7)
    plt.colorbar();
