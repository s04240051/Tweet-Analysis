import csv
import sys
import os
import statistics
import time
import argparse
import numpy as np
import pandas as pd

import nltk
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

lmtzr = WordNetLemmatizer()
stops = set(stopwords.words("english"))
#anew = "../lib/vad-nrc.csv"
anew = "/home/hzhu/project/ANEW/SentimentAnalysis//lib/EnglishShortened.csv"
avg_V = 5.06    # average V from ANEW dict
avg_A = 4.21
avg_D = 5.18

def analyze_line(texts, mode='mean'):
    '''
    args:
    texts (arrray_like):
        tweet text list
    mode (str):
        in "mean", "median", "mika"
    '''
    info_container = []
    columns = [
        'Valence',
        'Arousal',
        'Dominance',
        'Average VAD',
        'Sentiment Label',
        '# Words Found',
        'Found Words',
        'All Words',
    ]
    for text_line in texts:
        s = tokenize.word_tokenize(text_line.lower())
        #print("S" + str(i) +": " + s)
        all_words = []
        found_words = []
        total_words = 0
        v_list = []  # holds valence scores
        a_list = []  # holds arousal scores
        d_list = []  # holds dominance scores

        # search for each valid word's sentiment in ANEW
        words = nltk.pos_tag(s)
        for index, p in enumerate(words):
            # don't process stops or words w/ punctuation
            w = p[0]
            pos = p[1]
            if w in stops or not w.isalpha():
                continue

            # check for negation in 3 words before current word
            j = index-1
            neg = False
            while j >= 0 and j >= index-3:
                if words[j][0] == 'not' or words[j][0] == 'no' or words[j][0] == 'n\'t':
                    neg = True
                    break
                j -= 1

            # lemmatize word based on pos
            if pos[0] == 'N' or pos[0] == 'V':
                lemma = lmtzr.lemmatize(w, pos=pos[0].lower())
            else:
                lemma = w

            all_words.append(lemma)

            # search for lemmatized word in ANEW
            with open(anew) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if row['Word'].casefold() == lemma.casefold():
                        if neg:
                            found_words.append("neg-"+lemma)
                        else:
                            found_words.append(lemma)
                        v = float(row['valence'])
                        a = float(row['arousal'])
                        d = float(row['dominance'])

                        if neg:
                            # reverse polarity for this word
                            v = 5 - (v - 5)
                            a = 5 - (a - 5)
                            d = 5 - (d - 5)

                        v_list.append(v)
                        a_list.append(a)
                        d_list.append(d)

        if len(found_words) == 0:  # no words found in ANEW for this sentence
            
            info_container.append(
                [
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    0,
                    np.nan,
                    all_words,
                ]
            )
            
        else:  # output sentiment info for this sentence
            # get values
            if mode == 'median':
                sentiment = statistics.median(v_list)
                arousal = statistics.median(a_list)
                dominance = statistics.median(d_list)
            elif mode == 'mean':
                sentiment = statistics.mean(v_list)
                arousal = statistics.mean(a_list)
                dominance = statistics.mean(d_list)
            elif mode == 'mika':
                # calculate valence
                if statistics.mean(v_list) < avg_V:
                    sentiment = max(v_list) - avg_V
                elif max(v_list) < avg_V:
                    sentiment = avg_V - min(v_list)
                else:
                    sentiment = max(v_list) - min(v_list)
                # calculate arousal
                if statistics.mean(a_list) < avg_A:
                    arousal = max(a_list) - avg_A
                elif max(a_list) < avg_A:
                    arousal = avg_A - min(a_list)
                else:
                    arousal = max(a_list) - min(a_list)
                # calculate dominance
                if statistics.mean(d_list) < avg_D:
                    dominance = max(d_list) - avg_D
                elif max(d_list) < avg_D:
                    dominance = avg_D - min(a_list)
                else:
                    dominance = max(d_list) - min(d_list)
            else:
                raise Exception('Unknown mode')
                
                # set sentiment label
            label = 'neutral'
            if sentiment > 6:
                label = 'positive'
            elif sentiment < 4:
                label = 'negative'
            info_container.append(
                [
                    sentiment,
                    arousal,
                    dominance,
                    np.mean([sentiment, arousal, dominance]),
                    label,
                    ("%d out of %d" % (len(found_words), len(all_words))),
                    found_words,
                    all_words,
                ]
            )
    return pd.DataFrame(info_container, columns=columns)