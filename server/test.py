from flask import Flask, render_template, request
from get_text_from_url_demo import *
from feedforward_cbow_prints import *
from sklearn.neighbors import NearestNeighbors


PRETRAINED_VOCAB_PATH = "./glove.6B/glove.6B.100d.txt"



def get_knn(url, dict1):
    article = get_text(url)
    keyword = dict1.keys()
    val = [dict1[k] for k in dict1.keys()]
    val = np.array(val)
    nbrs = NearestNeighbors(n_neighbors=20, algorithm='ball_tree').fit(val)
    y = get_cwob_score_labels_v2(dict1, article)
    keyw = [keyword[i] for i in nbrs.kneighbors(y[0])[1][0]]
    return keyw

if __name__ == '__main__':
    url = 'https://www.thesun.co.uk/tvandshowbiz/6004503/lauren-goodger-accused-of-photoshopping-again-as-she-posts-lingerie-selfie/'
    article = get_text(url)
    dict1, length_vocab = vocab_dict_glove(PRETRAINED_VOCAB_PATH)
    keyword = dict1.keys()
    val = [dict1[k] for k in dict1.keys()]
    val = np.array(val)
    nbrs = NearestNeighbors(n_neighbors=20, algorithm='ball_tree').fit(val)
    y = get_cwob_score_labels_v2(dict1, article)

    for i in nbrs.kneighbors(y[0])[1][0]:
        print keyword[i]


