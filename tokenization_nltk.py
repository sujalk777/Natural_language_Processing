corpus="""Hello Welcome,to Krish Naik's NLP Tutorials.
Please do watch the entire course! to become expert in NLP.
"""
print(corpus)
##  Tokenization
## Sentence-->paragraphs
from nltk.tokenize import sent_tokenize
documents=sent_tokenize(corpus)
type(documents)
for sentence in documents:
    print(sentence)
## Tokenization 
## Paragraph-->words
## sentence--->words
from nltk.tokenize import word_tokenize
word_tokenize(corpus)
for sentence in documents:
    print(word_tokenize(sentence))
from nltk.tokenize import wordpunct_tokenize
wordpunct_tokenize(corpus)
from nltk.tokenize import TreebankWordTokenizer
tokenizer=TreebankWordTokenizer()
tokenizer.tokenize(corpus)
