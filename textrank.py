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


# In[3]:


df = pd.read_csv('bbc_data.csv')
df.head()


# In[4]:


doc = df[df.labels== 'business']['data'].sample(random_state=42)


# In[5]:


def wrap(x):
    return textwrap.fill(x,replace_whitespace=False,fix_sentence_endings=True)


# In[6]:


print(wrap(doc.iloc[0]))


# In[11]:


sents = nltk.sent_tokenize(doc.iloc[0])


# In[12]:


feature = TfidfVectorizer(
    stop_words = stopwords.words('english'),
    norm = 'l1' ,
)


# In[13]:


x = feature.fit_transform(sents)


# In[15]:


from sklearn.metrics.pairwise import cosine_similarity
s = cosine_similarity(x)


# In[16]:


s.shape


# In[17]:


len(sents)


# In[18]:


#normalize similarity matrix
s /= s.sum(axis=1, keepdims = True)


# In[19]:


s[0].sum()


# In[21]:


#uniform transition matrix
u = np.ones_like(s) / len(s)


# In[22]:


u[0].sum()


# In[23]:


#smoothed similarity method
factor = 1.5
s = (1-factor) * s + factor * u


# In[24]:


s[0].sum()


# In[26]:


#find the limiting /stationary distribution
eigenvals,eigenvecs = np.linalg.eig(s.T)


# In[27]:


eigenvals


# In[29]:


eigenvecs[:,0]


# In[32]:


eigenvecs[:,0].dot(s)


# In[33]:


eigenvecs[:,0] / eigenvecs[:,0].sum()


# In[35]:


limiting_dist = np.ones(len(s)) / len(s)
threshold = 1e-8
delta = float('inf')
iters = 0
while delta>threshold:
    iters += 1
    
    p = limiting_dist.dot(s) #markov transition
    
    delta = np.abs(p - limiting_dist).sum()
    
    #update limitig distribution
    limiting_dist = p
    
print(iters)
    


# In[36]:


limiting_dist


# In[37]:


limiting_dist.sum()


# In[38]:


np.abs(eigenvecs[:,0] / eigenvecs[:,0].sum() - limiting_dist).sum()


# In[39]:


scores = limiting_dist


# In[40]:


sort_ = np.argsort(-scores)


# In[43]:


print('generated summary: ')
# If sort_ is 2D, and you want the 6th column
for i in sort_:
    print(wrap(" %.2f: %s" % (scores[i], sents[i])))


# In[44]:


doc.iloc[0].split("\n")[0]


# In[46]:


def summarize(text, factor = 0.15):
    sents = nltk.sent_tokenize(text)
    feature = TfidfVectorizer(
        stop_words = stopwords.words('english'),
    norm = 'l1' ,
)
    x = feature.fit_transform(sents)
    s = cosine_similarity(x)
    s /= s.sum(axis=1, keepdims = True)
    u = np.ones_like(s) / len(s)
    s = (1-factor) * s + factor * u
    eigenvals,eigenvecs = np.linalg.eig(s.T)
    scores = eigenvecs[:,0] / eigenvecs[:,0].sum()
    sort_ = np.argsort(-scores)
    for i in sort_:
        print(wrap(" %.2f: %s" % (scores[i], sents[i])))


# In[52]:


doc = df[df.labels == 'entertainment']['data'].sample(random_state=123)
summarize(doc.iloc[0])


# In[53]:


doc.iloc[0].split("\n")[0]


# In[ ]:




