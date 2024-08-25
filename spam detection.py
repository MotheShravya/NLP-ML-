#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score,confusion_matrix
from sklearn.naive_bayes import MultinomialNB
# from wordcloud import WordCloud


# In[7]:


pip install wordcloud


# In[8]:


from wordcloud import WordCloud


# In[9]:


df = pd.read_csv('spam.csv',encoding='ISO-8859-1')


# In[10]:


df.head


# In[11]:


df.head()


# In[15]:


#dropping unnecessary columns
df = df.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1)


# In[16]:


df.head()


# In[17]:


df.columns=['labels','data']


# In[18]:


df.head()


# In[19]:


df['labels'].hist()


# In[20]:


#create binary classes
df['b_labels']=df['labels'].map({'ham':0,'spam':1})
y = df['b_labels'].to_numpy()


# In[22]:


#split up the data
df_train,df_test,ytrain,ytest=train_test_split(df['data'],y,test_size=0.3)


# In[24]:


#calculating features using countvectorizer
feature = CountVectorizer(decode_error='ignore')
xtrain = feature.fit_transform(df_train)
xtest = feature.transform(df_test)


# In[25]:


xtrain


# In[33]:


#create a model,trainit, print scores
model = MultinomialNB()
model.fit(xtrain,ytrain)
print("train accuracy: ",model.score(xtrain,ytrain))
print("test accuracy: ",model.score(xtest,ytest))


# In[36]:


ptrain = model.predict(xtrain)
ptest = model.predict(xtest)
print("f1: ",f1_score(ytrain,ptrain))
print("f1test: ",f1_score(ytest,ptest))


# In[38]:


prob_train = model.predict_proba(xtrain)[:,1]
prob_test = model.predict_proba(xtest)[:,1]
print("train: ",roc_auc_score(ytrain,prob_train))
print("test: ",roc_auc_score(ytest,prob_test))


# In[39]:


cm = confusion_matrix(ytrain,ptrain)
cm


# In[40]:


def plot_cm(cm):
    classes=['ham','spam']
    df_cm = pd.DataFrame(cm,index=classes,columns=classes)
    ax = sns.heatmap(df_cm,annot=True,fmt='g')
    ax.set_xlabel('predicted')
    ax.set_ylabel('target')
    
plot_cm(cm)


# In[41]:


cm_test=confusion_matrix(ytest,ptest)
plot_cm(cm_test)


# In[42]:


#visualize the data
def visualize(label):
    words = ''
    for msg in df[df['labels'] == label]['data']:
        msg = msg.lower()
        words += msg + ' '
    wordcloud = WordCloud(width=600,height=400).generate(words)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    
        
    


# In[44]:


visualize('spam')


# In[45]:


visualize('ham')


# In[46]:


#see wt is we are getting wrond
x = feature.transform(df['data'])
df['predictions'] = model.predict(x)


# In[47]:


#things that should be spam
sneaky = df[(df['predictions'] == 0) & (df['b_labels'] == 1)]['data']
for msg in sneaky:
    print(msg)


# In[ ]:





# In[49]:


#things that should not be spam
not_spam = df[(df['predictions'] == 1) & (df['b_labels'] == 0)]['data']
for msg in not_spam:
    print(msg)


# In[ ]:




