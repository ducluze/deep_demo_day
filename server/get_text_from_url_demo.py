#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import codecs
import requests
from goose import Goose
#from tqdm import tqdm
#import numpy as np
import re
from bs4 import BeautifulSoup


'''script to get the text from an url: python get_text_from_url.py URL'''


def cleanhtml(raw_html): #pour nettoyer les merdes dans le texte genre des balises manquantes ou des encodages chelou

    emoji_pattern = re.compile(
        u"(\ud83d[\ude00-\ude4f])|"  # emoticons
        u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
        u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
        u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
        u"(\ud83c[\udde0-\uddff])"  # flags (iOS)
        "+", flags=re.UNICODE)

    cleanr = re.compile('<.*?>|\\n|&quot;|\xa0')
    cleanbody = re.sub(cleanr, ' ', raw_html)
    cleanbody = re.sub(emoji_pattern, ' ', cleanbody)
    #cleanbody.encode('utf-8', 'replace')
    #cleanbody.replace('?', ' ')
    return ''.join([i if ord(i) < 128 else '' for i in cleanbody])


def get_text(url):
    part_url = url.split('/')
    content = requests.get(url).content
    soup = BeautifulSoup(content, "html5lib")
    list_p = soup.find_all('p')
    article = ""
    for text in list_p:
        article += text.getText()
    article = cleanhtml(article)
    article.replace("'", " ")
    return article
    # if "articles_scrapped" not in os.listdir('.'): # pour l'enregistrement des fichiers
    #     os.makedirs("articles_scrapped")
    # PATH_SAVED = "articles_scrapped/{}.txt".format(part_url[2])
    # f = codecs.open(PATH_SAVED, 'w', encoding='utf-8')
    # f.write(article)
    # print('Text scrapped successfully at -->',PATH_SAVED)
    # return(PATH_SAVED)

def get_text_goose(url):
    g = Goose()
    article = g.extract(url=url)
    return article

if __name__ == "__main__":
    get_text(sys.argv[1])
