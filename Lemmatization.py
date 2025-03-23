## Q&A,chatbots,text summarization
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
'''
POS- Noun-n
verb-v
adjective-a
adverb-r
'''
lemmatizer.lemmatize("going",pos='v')
words=["eating","eats","eaten","writing","writes","programming","programs","history","finally","finalized"]
for word in words:
    print(word+"---->"+lemmatizer.lemmatize(word,pos='v'))
lemmatizer.lemmatize("goes",pos='v')
lemmatizer.lemmatize("fairly",pos='v'),lemmatizer.lemmatize("sportingly")

