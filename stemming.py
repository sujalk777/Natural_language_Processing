## Classification Problem
## Comments of product is a positive review or negative review
## Reviews----> eating, eat,eaten [going,gone,goes]--->go

words=["eating","eats","eaten","writing","writes","programming","programs","history","finally","finalized"]
# Porter Stemmer
from nltk.stem import PorterStemmer
stemming=PorterStemmer()
for word in words:
    print(word+"---->"+stemming.stem(word))
stemming
stemming.stem("sitting")
# RegexpStemmer class
from nltk.stem import RegexpStemmer
reg_stemmer=RegexpStemmer('ing$|s$|e$|able$', min=4)
reg_stemmer.stem('eating')
reg_stemmer.stem('ingeating')

# Snowball Stemmer
from nltk.stem import SnowballStemmer
snowballsstemmer=SnowballStemmer('english')
for word in words:
    print(word+"---->"+snowballsstemmer.stem(word))
stemming.stem("fairly"),stemming.stem("sportingly")
snowballsstemmer.stem("fairly"),snowballsstemmer.stem("sportingly")
snowballsstemmer.stem('goes')
stemming.stem('goes')
