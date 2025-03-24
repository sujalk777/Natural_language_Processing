import gensim
from gensim.models import Word2Vec, KeyedVectors
import gensim.downloader as api

wv = api.load('word2vec-google-news-300')

vec_king = wv['king']
vec_king
vec_king.shape
wv['cricket']
wv.most_similar('cricket')
wv.most_similar('happy')
wv.similarity("hockey","sports")
vec=wv['king']-wv['man']+wv['woman']
vec
wv.most_similar([vec])

     
