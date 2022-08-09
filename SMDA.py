import re
from string import punctuation, digits

import jpype
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn import naive_bayes, svm, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

nltk.download('stopwords')
import re, os, pickle
from collections import Counter
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from string import punctuation, digits
import itertools
import pandas as pd
import numpy as np
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from string import punctuation, digits
from tensorflow.keras.layers import LSTM, Dense, Flatten
from sklearn.preprocessing import scale
from gensim.models import KeyedVectors
from tensorflow.keras.models import Sequential
from tensorflow import keras

jpype.startJVM("Java/jre1.8.0_291/bin/server/jvm.dll",
               "-Djava.class.path=E:\zemberek-full.jar", "-ea")

df = pd.read_csv(r'tweets.csv', sep=';', names=['Tweetler', 'Value'], skiprows=1)


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


def naiveBayes(x_train, y_train, x_test, y_test):
    clf = naive_bayes.MultinomialNB()
    clf.fit(x_train, y_train)
    print("************** Naive Bayes **************")
    print(classification_report(y_test, clf.predict(x_test)))
    print(confusion_matrix(y_test, clf.predict(x_test)))


def kNClassifier(x_train, y_train, x_test, y_test):
    neigh = KNeighborsClassifier(n_neighbors=9)
    neigh.fit(x_train, y_train)
    print("************** KNClassifier **************")
    print(classification_report(y_test, neigh.predict(x_test)))
    print(confusion_matrix(y_test, neigh.predict(x_test)))


def supportVectorMachines(x_train, y_train, x_test, y_test):
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    print("************** SVM **************")
    print(classification_report(y_test, clf.predict(x_test)))
    print(confusion_matrix(y_test, clf.predict(x_test)))


def logisticReg(x_train, y_train, x_test, y_test):
    clf = LogisticRegression(random_state=0)
    clf.fit(x_train, y_train)
    print("************** Logistic Regression **************")
    print(classification_report(y_test, clf.predict(x_test)))
    print(confusion_matrix(y_test, clf.predict(x_test)))


def decisionTree(x_train, y_train, x_test, y_test):
    clf = tree.DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    print("************** Decision Tree **************")
    print(classification_report(y_test, clf.predict(x_test)))
    print(confusion_matrix(y_test, clf.predict(x_test)))


def randomForest(x_train, y_train, x_test, y_test):
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(x_train, y_train)
    print("************** Random Forest **************")
    print(classification_report(y_test, clf.predict(x_test)))
    print(confusion_matrix(y_test, clf.predict(x_test)))


df['Normalization'] = normalization(df)
df['Tokanization'] = df['Normalization'].apply(clean_text)
df['StopWords'] = stopwordKaldirma(df)
df['Lemmazation'] = lemmazations(df)

"""tfIdfVektorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii')
x_train = tfIdfVektorizer.fit_transform(df_train['Lemmazation'])
x_test = tfIdfVektorizer.transform(df_test['Lemmazation'])"""

"""naiveBayes(x_train, y_train, x_test, y_test)
kNClassifier(x_train, y_train, x_test, y_test)
supportVectorMachines(x_train, y_train, x_test, y_test)
logisticReg(x_train, y_train, x_test, y_test)
decisionTree(x_train, y_train, x_test, y_test)
randomForest(x_train, y_train, x_test, y_test)"""

WPT = nltk.WordPunctTokenizer()
stop_word_list = nltk.corpus.stopwords.words('turkish')

word_vectors = KeyedVectors.load_word2vec_format(r"trmodel", binary=True)


def buildWordVector(text, size=400):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += word_vectors[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


tokensTrain = []
for i in range(0, len(df['Lemmazation'])):
    tokensTrain.append(WPT.tokenize(df['Lemmazation'][i]))

train_vecs = np.concatenate([buildWordVector(z) for z in tokensTrain])
train_vecs = scale(train_vecs)

train_vecs = np.reshape(train_vecs, (train_vecs.shape[0], 1, train_vecs.shape[1]))

y_df = []

for i in range(0, len(df['Value'])):
    if df['Value'][i] == 0:
        y_df.append(0)
    else:
        y_df.append(1)

y_df = np.array(y_df)

train_vecs, test_vecs, y_train, y_test = train_test_split(train_vecs, y_df, test_size=0.2, random_state=1, shuffle=True)

model = Sequential()
model.add(LSTM(200, input_shape=(1, train_vecs.shape[2])))
model.add(Dense(units=1, activation='sigmoid'))

opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()

mc = keras.callbacks.ModelCheckpoint("best_val_loss/", monitor='val_loss', verbose=0, save_best_only=True, mode='min',
                                     save_weights_only=True)

es = keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=10)
train_vecs, validation_vecs, y_train, y_validation = train_test_split(train_vecs, y_train, test_size=0.2,
                                                                      random_state=1, shuffle=True)
model.fit(train_vecs, y_train,
          epochs=15, batch_size=2, verbose=2,
          validation_data=(validation_vecs, y_validation),
          callbacks=[es, mc])

(model.load_weights("best_val_loss/"))

y_pred = model.predict(test_vecs,
                       batch_size=1,
                       verbose=1, steps=None)

y_pred = y_pred > 0.5

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_pred=y_pred, y_true=y_test)
print('True Positives: ', cm[1, 1])
print('False Positives: ', cm[0, 1])
print('True Negatives: ', cm[0, 0])
print('False Negatives: ', cm[1, 0])

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %f' % accuracy)
jpype.shutdownJVM()
