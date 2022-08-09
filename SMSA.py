import jpype
import pandas as pd
import numpy as np
import re
from string import punctuation, digits
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import naive_bayes, svm, tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import nltk

nltk.download('stopwords')

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

tfIdfVektorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii')
x_df = tfIdfVektorizer.fit_transform(df['Lemmazation'])
y_df = []

for i in range(0, len(df['Value'])):
    if df['Value'][i] == 0:
        y_df.append(0)
    else:
        y_df.append(1)

y_df = np.array(y_df)

x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=1, shuffle=True)

naiveBayes(x_train, y_train, x_test, y_test)
kNClassifier(x_train, y_train, x_test, y_test)
supportVectorMachines(x_train, y_train, x_test, y_test)
logisticReg(x_train, y_train, x_test, y_test)
decisionTree(x_train, y_train, x_test, y_test)
randomForest(x_train, y_train, x_test, y_test)

jpype.shutdownJVM()
