import pandas as pd
messages=pd.read_csv('SpamClassifier-master/smsspamcollection/SMSSpamCollection',
                    sep='\t',names=["label","message"])
messages
## Data Cleaning And Preprocessing
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
wordlemmatize=WordNetLemmatizer()
corpus=[]
for i in range(0,len(messages)):
    review=re.sub('[^a-zA-z]',' ',messages['message'][i])
    review=review.lower()
    review=review.split()
    review=[wordlemmatize.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)
corpus
# Tf_Idf technique implemented
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(max_features=100)
X=tfidf.fit_transform(corpus).toarray()
import numpy as np
np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))

# n-grams algoritms
tfidf=TfidfVectorizer(max_features=100,ngram_range=(2,2))
X=tfidf.fit_transform(corpus).toarray()
tfidf.vocabulary_
