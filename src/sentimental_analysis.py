import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split

np.set_printoptions(precision=2)


def preprocessor(text):
    #clean data by removing any html script and moving all emojis to the end of the data string.
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) +\
        ' '.join(emoticons).replace('-', '')
    return text

def tokenizer(text):
    #convert text to array of words
    return text.split()

def tokenizer_porter(text):
    #stem and split the text into tokens
    return [porter.stem(word) for word in text.split()]


#read dataset of reviews in as df
df= pd.read_csv("C:/Users/Dell/PycharmProjects/Sentiment_Analysis_with_scikit-learn_project/data/movie_data.csv")
print(df.head(10))
print(df['review'][0])

#initilize CountVector
count=CountVectorizer()
docs=np.array(['The sun is shining',
               'The weather is sweet',
               'The sun is shining, the weather is sweet, and one and one is two'])

#compute bag of words
bag=count.fit_transform(docs)
print(count.vocabulary_)
print(bag.toarray())


tfidf= TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())

print(df.loc[0, 'review'][-50:])

print(preprocessor(df.loc[0, 'review'][-50:]))
print(preprocessor("<a/>><html> ;D amazing things :) test this"))
df['review']=df['review'].apply(preprocessor)

porter = PorterStemmer()

print(tokenizer("the test that does some testing thus continue testing"))
print(tokenizer_porter("the test that does some testing thus continue testing"))

#nltk.download('stopwords')

stop = stopwords.words('english')
print([w for w in tokenizer_porter("the test that does some testing thus continue testing")[-10:] if w not in stop])



tfidf=TfidfVectorizer(strip_accents=None,
                     lowercase=False,
                     preprocessor=None,
                     tokenizer=tokenizer_porter,
                     use_idf=True,
                     norm='l2',
                     smooth_idf=True)
y=df.sentiment.values
x=tfidf.fit_transform(df.review)


#spilt data into 50% taining and testing
X_train, X_test, Y_train, Y_test = train_test_split(x, y, random_state=1,test_size=0.5, shuffle=False)

#build model and save using pickle to local drive
clf = LogisticRegressionCV(cv=5,
                          scoring='accuracy',
                          n_jobs=-1,
                          verbose=3,
                          max_iter=300).fit(X_train, Y_train)
saved_model= open('saved_model.sav', 'wb')
pickle.dump(clf, saved_model)
saved_model.close()

#use model to predict values for test data.
filename = 'saved_model.sav'
saved_clf = pickle.load(open(filename, 'rb'))
saved_clf.score(X_test, Y_test)