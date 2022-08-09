import os
import pandas as pd

basepath = 'Twitter/neg/'
basepath2 = 'Twitter/pos/'

df = pd.DataFrame(columns=['Tweetler', 'Value'])

for entry in os.listdir(basepath):
    if os.path.isfile(os.path.join(basepath, entry)):
        with open(basepath + entry, encoding='utf-8_sig') as f:
            lines = f.readlines()
        df = df.append({'Tweetler': lines[0], 'Value': 0}, ignore_index=True)

for entry2 in os.listdir(basepath2):
    if os.path.isfile(os.path.join(basepath2, entry2)):
        with open(basepath2 + entry2, encoding='utf-8_sig') as f:
            lines = f.readlines()
        df = df.append({'Tweetler': lines[0], 'Value': 1}, ignore_index=True)

print(df)
df.to_csv('tweets.csv', sep=';', encoding='utf-8_sig', index=False)
