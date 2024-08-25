#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import textwrap
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem  import WordNetLemmatizer,PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


nltk.download('punkt')
nltk.download('stopwords')


# In[4]:


df = pd.read_csv('bbc_data.csv')
df.head()


# In[6]:


doc = df[df.labels== 'business']['data'].sample(random_state=42)


# In[7]:


def wrap(x):
    return textwrap.fill(x,replace_whitespace=False,fix_sentence_endings=True)


# In[8]:


print(wrap(doc.iloc[0]))


# In[13]:


text = doc.iloc[0]
if "\n" in text:
    sents = nltk.sent_tokenize(text.split("\n", 1)[1])
else:
    # Handle the case where there is no newline or only one line
    sents = nltk.sent_tokenize(text)


# In[20]:


feature = TfidfVectorizer(
    stop_words = stopwords.words('english'),
    norm = 'l1' ,
)


# In[21]:


x = feature.fit_transform(sents)


# In[22]:


def get_sentence(tfidf_row):
    #return the average of non zeros
    #of the tfidf vectorizer representation of a sentence
    x = tfidf_row[tfidf_row != 0]
    return x.mean()


# In[23]:


scores = np.zeros(len(sents))
for i in range(len(sents)):
    score = get_sentence(x[i,:])
    scores[i] = score


# In[24]:


sort = np.argsort(-scores)


# In[25]:


print('generated summary: ')
for i in sort[:5]:
    print(wrap(" %.2f: %s" % (scores[i],sents[i])))


# In[26]:


doc.iloc[0].split("\n", 1)[0]


# In[27]:


def summarize(text):
    sents = nltk.sent_tokenize(text)
    x=feature.fit_transform(sents)
    scores = np.zeros(len(sents))
    for i in range(len(sents)):
        score = get_sentence(x[i,:])
        scores[i] = score
        
    
    sort = np.argsort(-scores)
    
    for i in sort[:5]:
        print(wrap("%.2f: %s" % (scores[i],sents[i])))
    


# In[28]:


doc.iloc[0].split("\n", 1)[0]


# In[29]:


print(wrap(doc.iloc[0]))


# In[ ]:




