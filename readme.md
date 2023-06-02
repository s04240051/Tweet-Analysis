# Twitter Data Analysis
---
## Dataset Directory Structure
`./emma_queries_10_13.json` and `./emma_queries_13_23.json` are the original dataset obtained from crawler and only the second one is processed now. `./name_file` contains the meta data about every candidator extracted from original dataset. `./name_text` contains the datasets with only tweet text and posted time. `./extracted_features` contains the tweet text sentiment analysis for the selected candidators. 
 ```
emma
 ┣ extracted_feature
 ┃ ┗ JoeBiden_Joe Biden_939091.csv
 ┣ name_file
 ┃ ┣ APechtold_Alexander Pechtold_16708690.json
 ┃ ┣ BarackObama_Barack Obama_813286.json
 ┃ ┣ BernieSanders_Bernie Sanders_216776631.json
 ┃ ┗ ...
 ┣ name_text
 ┃ ┣ APechtold_Alexander Pechtold_16708690.csv
 ┃ ┣ BarackObama_Barack Obama_813286.csv
 ┃ ┣ BernieSanders_Bernie Sanders_216776631.csv
 ┃ ┣ DassenLaurens_Laurens Dassen_1050373998587056128.csv
 ┃ ┗ ...
 ┣ emma_queries_10_13.json
 ┗ emma_queries_13_23.json
```
## Sentiment Analysis Result
This section will describe the information about the columns used by the files in the `./extracted_feature`. There are 21 columns in total by now. The tasks include sentiment, arousal, valence, Dominance, and readability. In this folder, each .csv file represent a condidator's tweet. 

|Tweet info|Introduction|
|:-:|:-:|
|time|created time of tweet|
|text|tweet text include the emoji|
|**Columns from Vader**|**Check this [link](https://github.com/cjhutto/vaderSentiment)**|
|neg|negative score (float) [0,1]|
|neu|neutral score (float) [0,1]|
|pos|positive score (float) [0,1]|
|compound|summarize the valence scores of each word in the lexicon (float) [-1,1]|
|sentiment_class_vader|pos: compound>=0.05 neu: -0.05<compound<0.05 neg: compound<=-0.05|
|**Readability score**|**Check this [blog](https://www.geeksforgeeks.org/readability-index-pythonnlp/) and [document](https://textacy.readthedocs.io/en/0.11.0/api_reference/text_stats.html#textacy.text_stats.readability.automated_readability_index)**|
|flesch_reading|Readability test used as a general-purpose standard in several languages (float)|
|smog_index|estimates the number of years of education required to understand a tex (float)|
|flesch_kincaid|Readability test used widely in education, whose value estimates the U.S. (float)|
|coleman_liau|Readability test whose value estimates the number of years of education required to understand a text (float)|
|automated_readability_index|Readability test for English-language texts, particularly for technical writing|
|dale_chall_readability|Compute the percentage of words NOT on the Dale–Chall word list of 3, 000 easy words （float)|
|**Score from ANEW**|**Check this [repository](https://github.com/bagustris/text-vad/tree/master)**|
|Valence||
|Arousal||
|Dominance||
|Average VAD||
|Sentiment label||
|# Words Found|Found_words out of all words|
|Found_words|lemmatized word in ANEW|
|Found Words|all lemmatize words based on pos-tag|
|like_count|int|
|quote_count|int|
|reply_count|int|
|retweet_count|int|

