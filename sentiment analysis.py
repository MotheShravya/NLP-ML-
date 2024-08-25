#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


np.random.seed(1)
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score,confusion_matrix
from sklearn.linear_model import LogisticRegression


# In[3]:


df = pd.read_csv('Tweets.csv')
df.head()


# In[4]:


df = df[['airline_sentiment','text']].copy()


# In[5]:


df.head()


# In[6]:


df['airline_sentiment'].hist()


# In[7]:


target = {'positive': 1,'negative':0,'neutral':2}
df['target'] = df['airline_sentiment'].map(target)


# In[8]:


df.head()


# In[9]:


df_train,df_test = train_test_split(df)


# In[10]:


df_train.head()


# In[11]:


vectorize = TfidfVectorizer(max_features=2000)


# In[12]:


xtrain = vectorize.fit_transform(df_train['text'])


# In[13]:


xtrain


# In[20]:


xtest = vectorize.transform(df_test['text'])


# In[21]:


xtest


# In[22]:


ytrain = df_train['target']
ytest = df_test['target']


# In[23]:


model = LogisticRegression(max_iter=500)
model.fit(xtrain,ytrain)
print('train: ',model.score(xtrain,ytrain))
print('test: ',model.score(xtest,ytest))


# In[24]:


ptrain = model.predict_proba(xtrain)
ptest = model.predict_proba(xtest)
print('train: ',roc_auc_score(ytrain,ptrain,multi_class='ovo'))
print('test: ',roc_auc_score(ytest,ptest,multi_class='ovo'))


# In[28]:


prtrain = model.predict(xtrain)
prtest = model.predict(xtest)


# In[29]:


cm = confusion_matrix(ytrain,prtrain,normalize='true')
cm


# In[30]:


def plot_cm(cm):
    classes = ['negative','positive','neutral']
    df_cm = pd.DataFrame(cm,index=classes,columns=classes)
    ax = sns.heatmap(df_cm,annot = True,fmt='g')
    ax.set_xlabel('predicted')
    ax.set_ylabel('target')
    
plot_cm(cm)


# In[32]:


cm_test = confusion_matrix(ytest,prtest,normalize='true')
plot_cm(cm_test)


# In[34]:


binary_target = [target['positive'],target['negative']]
df_b_train = df_train[df_train['target'].isin(binary_target)]
df_b_test = df_test[df_test['target'].isin(binary_target)]


# In[35]:


df_b_train.head()


# In[36]:


xtrain = vectorize.fit_transform(df_b_train['text'])
xtest = vectorize.transform(df_b_test['text'])


# In[37]:


ytrain = df_b_train['target']
ytest = df_b_test['target']


# In[38]:


model = LogisticRegression(max_iter = 500)
model.fit(xtrain,ytrain)
print('train: ',model.score(xtrain,ytrain))
print('test: ',model.score(xtest,ytest))


# In[39]:


prtrain = model.predict_proba(xtrain)[:,1]
prtest = model.predict_proba(xtest)[:,1]
print('train: ',roc_auc_score(ytrain,prtrain))
print('test: ',roc_auc_score(ytest,prtest))



# In[40]:


model.coef_


# In[43]:


plt.hist(model.coef_[0],bins=30);


# In[42]:


word_index = vectorize.vocabulary_
word_index


# In[45]:


#lets looks at the weights for each word
#try it with diff threshold
threshold = 2
print('most positive words: ')
for word,index in word_index.items():
    weight = model.coef_[0][index]
    if weight > threshold:
        print(word,weight)


# In[ ]:




