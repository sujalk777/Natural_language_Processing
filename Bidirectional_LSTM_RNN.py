## Dataset: https://www.kaggle.com/competitions/fake-news/data?select=train.csv
import pandas as pd
df=pd.read_csv('train.csv',engine='python',error_bad_lines=False)
df.head()
df.isnull().sum()
df.shape
df=df.dropna()
df.head()
## get the independent features
X=df.drop('label',axis=1)
y=df['label']
## Check whether dataset is balanced or not
y.value_counts()
import tensorflow as tf
tf.__version__
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional

## vocabulary size
voc_size=5000
messages=X.copy()
import nltk
import re
from nltk.corpus import stopwords
nltk.download('stopwords')
messages.reset_index(inplace=True)
### Dataset Preprocessing
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    print(i)
    review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
corpus
onehot_repr=[one_hot(words,voc_size) for words in corpus]
onehot_repr
sent_length=20
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)

embedded_docs
## Creating model
embedding_vector_features=40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(Bidirectional(LSTM(100)))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())
import numpy as np
X_final=np.array(embedded_docs)
y_final=np.array(y)
## train test split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)
## Model Training
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=32)
y_pred=model.predict(X_test)
import numpy as np

y_pred=np.where(y_pred>=0.5,1,0)
from sklearn.metrics import confusion_matrix,accuracy_score
confusion_matrix(y_test,y_pred)
print(accuracy_score(y_test,y_pred))
