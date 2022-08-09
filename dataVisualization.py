import pickle
import jpype
import pandas as pd
import numpy as np
import re
from string import punctuation, digits
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

nltk.download('stopwords')
import matplotlib.pyplot as plt

# tweetler
df = pd.read_csv(r'survivor.csv', sep='~', names=['Tweetler', 'Tarihler'], skiprows=1)

# modeli çağırma
with open('SVMModel', 'rb') as f:
    model = pickle.load(f)

# ön işleme adımları
jpype.startJVM("Java/jre1.8.0_291/bin/server/jvm.dll",
               "-Djava.class.path=E:\zemberek-full.jar", "-ea")


def clean_text(text):
    text = text.lower()  # Metnin hepsini küçük karakter yapma
    text = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))', '', text)  # URL leri kaldırma
    text = re.sub(r'@[^\s]+ ', '', text)  # Kullanıcıları kaldırma
    text = re.sub(r'#', '', text)  # Hashtag leri kaldırma
    text = re.sub(r'(rt )', '', text)  # rt leri kaldırma
    regrex_pattern = re.compile(pattern="["
                                        u"\U0001F600-\U0001F64F"  # emoji listesi
                                        u"\U0001F300-\U0001F5FF"
                                        u"\U0001F680-\U0001F6FF"
                                        u"\U0001F1E0-\U0001F1FF"
                                        u"\U00002702-\U000027B0"
                                        u"\U000024C2-\U0001F251"
                                        u"\U0001f926-\U0001f937"
                                        u'\U00010000-\U0010ffff'
                                        u"\u200d"
                                        u"\u2640-\u2642"
                                        u"\u2600-\u2B55"
                                        u"\u23cf"
                                        u"\u23e9"
                                        u"\u231a"
                                        u"\u3030"
                                        u"\ufe0f"
                                        "]+", flags=re.UNICODE)
    text = re.sub(regrex_pattern, '', text)  # Emojileri kaldırma
    text = re.sub(r'(\n)', '', text)  # \n leri kaldırma
    text = re.sub(r'(’)', '', text)  # ’ kaldırma
    cevirici = str.maketrans('', '', punctuation)  # noktalama işaretlerini kaldırma
    text = text.translate(cevirici)
    cevirici = str.maketrans('', '', digits)  # rakamları kaldırma
    text = text.translate(cevirici)
    return text


def stopwordKaldirma(dataframe):
    stopWord = set(stopwords.words('turkish'))
    dataframe['durakKelimesizHali'] = dataframe['Tokanization'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in (stopWord)]))

    dataframe['durakKelimesizHali'] = dataframe['durakKelimesizHali'].apply(
        lambda x: ' '.join([w for w in x.split() if len(w) > 3]))
    return dataframe['durakKelimesizHali']


def normalization(dataframe):
    from os.path import join
    from jpype import JClass, JString
    TurkishMorphology: JClass = JClass('zemberek.morphology.TurkishMorphology')
    TurkishSentenceNormalizer: JClass = JClass(
        'zemberek.normalization.TurkishSentenceNormalizer'
    )
    Paths: JClass = JClass('java.nio.file.Paths')

    normalizer = TurkishSentenceNormalizer(
        TurkishMorphology.createWithDefaults(),
        Paths.get("normalization"),
        Paths.get(
            join("lm.2gram.slm")
        )
    )

    tweetler = []
    for tw in dataframe['Tweetler']:
        try:
            normal = str(normalizer.normalize(JString(tw)))
            tweetler.append(normal)
        except:
            tweetler.append(tw)

    dataframe['Normalization'] = [tweet for tweet in tweetler]
    return dataframe['Normalization']


def lemmazations(dataframe):
    from jpype import JClass
    TurkishMorphology: JClass = JClass('zemberek.morphology.TurkishMorphology')
    Paths: JClass = JClass('java.nio.file.Paths')

    WordAnalysis: JClass = JClass('zemberek.morphology.analysis.WordAnalysis')

    morphology: TurkishMorphology = TurkishMorphology.createWithDefaults()

    lemmazation = []

    for tweet in dataframe['StopWords']:
        if (len(tweet) != 0):
            tw = tweet.split()
            kokListesi = []
            for tww in tw:
                analysis = morphology.analyzeSentence(tww)

                # Resolving the ambiguity
                results = morphology.disambiguate(tww, analysis).bestAnalysis()
                i = 1
                kok = []

                for result in results:
                    while i < len(result.formatLong()):
                        if result.formatLong()[i] == ":":
                            if ''.join(kok) != 'unk':
                                kokListesi.append((''.join(kok).lower()))
                            else:
                                kokListesi.append(tww)
                            kok = []
                            i = 1
                            break
                        kok.append((result.formatLong()[i]).lower())
                        i += 1
            lemmazation.append(str((' '.join(kokListesi).lower())))
        else:
            lemmazation.append(tweet)
    dataframe['Lemmazation'] = [lemmas for lemmas in lemmazation]
    return dataframe['Lemmazation']


df['Normalization'] = normalization(df)
df['Tokanization'] = df['Normalization'].apply(clean_text)
df['StopWords'] = stopwordKaldirma(df)
df['Lemmazation'] = lemmazations(df)

# vektör dönüşümü
tfIdfVektorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii')
x_df = tfIdfVektorizer.fit_transform(df['Lemmazation'])

# Görselleştirme
value = []
for x_predict in x_df:
    sonuc = model.predict(x_predict)[0]
    value.append(sonuc)

df['Value'] = [val for val in value]
print(value.size)

dates = np.array(df['Tarihler'])
indices = np.argsort(dates)
window = 750

dates = dates[indices][window:]
values = np.array(df['value'])[indices]
windows = pd.Series(values).rolling(window)
moving_averages = windows.mean()[window:]

plt.figure(figsize=(12, 6))
plt.plot(dates, moving_averages, color='blue', label='Average Sentiment')
plt.title('Analysis of Turkish Tweets about Survivor2021')
plt.xlabel('Date')
plt.ylabel('Sentiment Score')
plt.legend()

jpype.shutdownJVM()
