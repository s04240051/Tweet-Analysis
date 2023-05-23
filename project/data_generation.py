import os
import json
from collections import defaultdict
import csv
from tqdm import tqdm
import csv
import pandas as pd
import nltk
import re 
from textblob import TextBlob
from collections import Counter, defaultdict
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textstat.textstat import textstatistics
import textstat
from ANEW_util import analyze_line
import spacy

class DATA_REBUILD:
    def __init__(
            self,
            file_name,
            source_file,
            save_fold=None,
    ):

        self.tweets = self.open_json(os.path.join(file_name, source_file))
        self.name_dict = self.id_idx_map()
        self.save_fold = save_fold
        (
            self.wholedata_fold,
            self.time_text_fold,
        ) = (
            os.path.join(file_name, self.save_fold[0]),
            os.path.join(file_name, self.save_fold[1]),
        )

    def open_json(self, file_name):
        tweets = []
        for line in open(file_name, 'r'):
            tweets.append(json.loads(line))
        return tweets

    def id_idx_map(self):
        name_dict = defaultdict(list)
        name_list = [line['data'][0]['author_id'] for line in self.tweets]
        for i, id in enumerate(name_list):
            name_dict[id].append(i)
        return name_dict

    def pipeline(self):
        for id, indexes in tqdm(self.name_dict.items()):

            media_list, tweet_list = [], []
            time, text = [], []

            user_inf = self.get_user_inf(
                id, self.tweets[list(indexes)[0]])
            out_name = f"{user_inf['username']}_{user_inf['name']}_{id}"
            assert user_inf is not None, 'fail to get user information'

            for index in indexes:
                sub_tweets = self.tweets[index]['includes']
                tweet_list.extend(sub_tweets['tweets'])
                media_list.extend(
                    sub_tweets['media'] if 'media' in sub_tweets else [])
                text.extend([sub_tweets['tweets'][i]['text']
                            for i in range(len(sub_tweets['tweets']))])
                time.extend([sub_tweets['tweets'][i]['created_at']
                             for i in range(len(sub_tweets['tweets']))])
            if self.save_fold is not None:
                out_container = {
                    "user": user_inf,
                    "tweets": tweet_list,
                    "media": media_list,
                }
                self.save(out_container, zip(time, text), out_name)

    def save(self, whole_data, time_text, out_name):

        whole_data_path, time_text_path = (
            os.path.join(self.wholedata_fold, out_name+'.json'),
            os.path.join(self.time_text_fold, out_name+'.csv'),
        )
        whole_data = json.dumps(whole_data, sort_keys=True, indent=0, separators=(',', ':'))
        with open(whole_data_path, "w") as f_w:
            f_w.write(whole_data)
        with open(time_text_path, 'w', encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            for time, text in time_text:
                writer.writerow([time, text])

    def get_user_inf(self, id, tweet):
        for u in tweet['includes']['users']:
            if id == u['id']:
                return u
        return None
    
class SENTIMENT:
    def __init__(self, dataset_folder, candidators):
        '''
        Class to do the text sentiment analysis
        arg:
        ----------
        dataset_folder : str
            path to the dataset ./name_text
        candidators : list
            a list of first name of the candidators

        '''
        self.dataset_folder = dataset_folder
        self.file_name_list = []
        self.sid = SentimentIntensityAnalyzer()
        for name in os.listdir(self.dataset_folder):
            sub_result = sum([name.startswith(item) for item in candidators])
            if sub_result>0:
                self.file_name_list.append(name)
        
    def open_file(self, path1):
        out = []
        with open(path1,"r", encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            
            for line in reader:
                out.append(line)
        df = pd.DataFrame(out, columns=["time", "text"])
        return df

    def clean_tweet(self, tweet):
        '''
        Utility function to clean the text in a tweet by removing 
        links and special characters using regex.
        '''
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

    def vader_index(self, tweets):
        sub_dict = defaultdict(list)
        keys = ['neg', 'neu', 'pos', 'compound']
        for text in tweets.values:
            score = self.sid.polarity_scores(text)
            for key in keys:
                sub_dict[key].append(score[key]) 
            if score['compound'] >= 0.05:
                cl = 1
            elif 0.05<score['compound']<0.05:
                cl = 0
            else:
                cl = -1
            sub_dict["sentiment_class_vader"].append(cl)
        return pd.DataFrame(sub_dict)

    def read_info(self, df):
        df["flesch_reading"] = df["text"].apply(textstat.flesch_reading_ease)
        df["smog_index"] = df["text"].apply(textstat.smog_index)
        df["flesch_kincaid"] = df["text"].apply(textstat.flesch_kincaid_grade) 
        df["coleman_liau"] = df["text"].apply(textstat.coleman_liau_index)
        df["automated_readability_index"] = df["text"].apply(textstat.automated_readability_index)
        df["dale_chall_readability"] = df["text"].apply(textstat.dale_chall_readability_score)
        return df
    
    def data_pipeline(self, save_path, mode='mean', rewrite=True):
        '''
        Main data process function
        Args
        ---------
        save_path : str
            absolute path of folder to save the output files
        mode : determines how sentiment values for a sentence are computed (median or mean). Default is mean
        rewrite: whether to rewite the previous result. Default is True. 
        '''

        for name in tqdm(self.file_name_list):
            path = os.path.join(self.dataset_folder, name)
            out_path = os.path.join(save_path, name)
            if os.path.isfile(out_path) and not rewrite:
                continue 
            df = self.open_file(path)
            df['text'] = df['text'].apply(self.clean_tweet)
            df = pd.concat([df, self.vader_index(df['text'])], axis=1)
            df = self.read_info(df)
            df = pd.concat([df, analyze_line(df['text'].values, mode=mode)], axis=1)

            df.to_csv(out_path, index=False)

if __name__ == "__main__":
    '''
    #better use argparse to control. Anyway, use this block to generate the dataset.
    file_name = './emma/'
    data_build = DATA_REBUILD(
        file_name=file_name,
        source_file="emma_queries_13_23.json",
        save_fold=('name_file', 'name_text'),
    )
    data_build.pipeline()
    '''
    #use this block of code to generate the sentiment analysis
    dataset_folder = './emma/name_text/'
    out_path = './emma/extracted_features'
    candidators = ['Barack', 'Mitt', 'Hillary', 'Donald', 'Joe', 'POTUS']
    data_builder = SENTIMENT(dataset_folder, candidators)
    data_builder.data_pipeline(save_path=out_path, rewrite=False)
