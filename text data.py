# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 20:04:51 2022

@author: Rahul
"""

import pandas as pd
import numpy as np
import string
import re
import nltk
from textblob import TextBlob
nltk.download('punkt')
nltk.download('stopwords') 
from nltk.corpus import stopwords

df=pd.read_csv("D:\\DS\\books\\ASSIGNMENTS\\Text Mining\\negative-words.txt",error_bad_lines=False,encoding="latin-1",header=None)
df1=df.drop(df.index[0:26],axis=0)
df1

df=pd.read_csv("D:\\DS\\books\\ASSIGNMENTS\\Text Mining\\positive-words.txt",error_bad_lines=False,encoding="latin-1",header=None)
df2=df.drop(df.index[0:26],axis=0)
df2

# Concating positive and negative word text
df=pd.concat([df1,df2],axis=0)
# Naming the column
df.columns={'X'}
df

def calpolarity(x):
    return TextBlob(x).sentiment.polarity

def calSubjectivity(x):
    return TextBlob(x).sentiment.subjectivity

def segmentation(x):
    if x > 0:
        return "positive"
    if x== 0:
        return "neutral"
    else:
        return "negative"
    
df['polarity']=df["X"].apply(calpolarity)
df['subjectivity']=df["X"].apply(calSubjectivity)
df['segmentation']=df["polarity"].apply(segmentation)

df.head()

# Analysis and visualization
df.pivot_table(index=['segmentation'],aggfunc={"segmentation":'count'})
    
# Removes both the leading and the trailing characters
df = [X.strip() for X in df.X] 

# Removes empty strings
book = [X for X in df if X] 
book[0:10]
 
# Joining the list into one string/text
text = ' '.join(book)
text

# Punctuation
no_punc_text = text.translate(str.maketrans('', '', string.punctuation)) 
no_punc_text

# Tokenization
from nltk.tokenize import word_tokenize
text_tokens = word_tokenize(no_punc_text)
print(text_tokens[0:50])
  
# Removing stopwords
my_stop_words = stopwords.words('english')
no_stop_tokens = [word for word in text_tokens if not word in my_stop_words]

# Noramalize the data
lower_words = [X.lower() for X in no_stop_tokens]
print(lower_words[0:40])

# Stemming the data
from nltk.stem import PorterStemmer
ps = PorterStemmer()
stemmed_tokens = [ps.stem(word) for word in lower_words]
print(stemmed_tokens[0:10])

import spacy
nlp=spacy.load("en_core_web_sm")
# lemmas being one of them, but mostly POS, which will follow later
doc = nlp(' '.join(no_stop_tokens))
print(doc[0:40])   
    
lemmas = [token.lemma_ for token in doc]
print(lemmas[0:25])

# Feature Extraction
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(stemmed_tokens)
pd.DataFrame.from_records([vectorizer.vocabulary_]).T.sort_values(0,ascending=False).head(20)
print(vectorizer.vocabulary_)
print(vectorizer.get_feature_names()[50:100])
print(X.toarray()[50:100])
print(X.toarray().shape)   

# Bigrams and Trigrams 
vectorizer_ngram_range = CountVectorizer(analyzer='word',ngram_range=(1,3),max_features = 100)
bow_matrix_ngram =vectorizer_ngram_range.fit_transform(book)
bow_matrix_ngram
print(vectorizer_ngram_range.get_feature_names())
print(bow_matrix_ngram.toarray())


# TFidf vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer_n_gram_max_features = TfidfVectorizer(norm="l2",analyzer='word', ngram_range=(1,3), max_features = 10)
tf_idf_matrix_n_gram_max_features =vectorizer_n_gram_max_features.fit_transform(book)
print(vectorizer_n_gram_max_features.get_feature_names())
print(tf_idf_matrix_n_gram_max_features.toarray())

# WordCloud
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Define a function to plot word cloud
def plot_cloud(wordcloud):
    # Set figure size
    plt.figure(figsize=(15, 30))
    # Display image
    plt.imshow(wordcloud) 
    # No axis details
    plt.axis("off");
    
wordcloud = WordCloud(width = 3000, height = 2000, background_color='black', max_words=100,colormap='Set2').generate(text)
plot_cloud(wordcloud)
plt.show()

















    
    
    
    
    
    
    
    
    
    
    
    