from datetime import datetime

from textblob import TextBlob
import tweepy
import pandas as pd
import numpy as np
import sys
import re

api_key = ""
api_key_secret = ""
access_token = ""
access_token_secret = ""

auth_handler = tweepy.OAuthHandler(consumer_key=api_key, consumer_secret=api_key_secret)
auth_handler.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth_handler, wait_on_rate_limit=True)
twits = []
search_term = "#Survivor2021"
tweet_amount = 150
tarihler = ['2021-05-31', '2021-06-01', '2021-06-02', '2021-06-03', '2021-06-04', '2021-06-05', '2021-06-06',
            '2021-06-07']
data = None
for i in range(0, 7):
    tweets = tweepy.Cursor(api.search, q=search_term, lang="tr", since=tarihler[i], until=tarihler[i + 1],
                           tweet_mode="extended").items(tweet_amount)
    sayi = 0
    for tweet in tweets:
        sayi += 1
        twits.append(tweet)
        print(tweet.full_text)
    data = pd.DataFrame([twit.full_text for twit in twits], columns=['Tweetler'])
    data['Tarih'] = np.array([twit.created_at for twit in twits])
print(data)

data.to_csv(r'survivor.csv', encoding="utf-8-sig", index=False, sep='~')
