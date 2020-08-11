#!/usr/bin/env python
# coding: utf-8

# In[129]:


import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt


# In[130]:


bank=pd.read_csv('bank-full.csv',sep=";")


# In[131]:


bank.head()


# In[132]:


print("{rows}".format(rows = len(bank)))


# In[133]:


bank.isnull().sum()


# In[134]:


bank.describe()


# In[136]:


sns.countplot(x=bank[bank['loan']=="yes"]['loan'],hue='marital',data=bank)


# In[17]:


sns.pairplot(bank,hue='y')


# In[137]:


sns.countplot(x=bank[bank['housing']=="yes"]['housing'],hue='marital',data=bank)


# In[162]:


data1=bank.groupby('education').apply(lambda x:(x[x['loan']=="yes"]['loan']).count())
plt.subplot(1,2,1)
data1.plot(kind='bar' , figsize= (20,5))
plt.ylabel("loan")

#print(data1)
data2=bank.groupby('education').apply(lambda x:(x[x['loan']=="yes"]['loan']).count())
plt.subplot(1,2,2)
data2.plot(kind='bar')
plt.ylabel("loan")
#print(data1)

plt.show()


# In[139]:


ax=sns.stripplot(x="education",y="balance",hue="marital",data=bank)


# In[164]:



data1=bank.groupby('month').apply(lambda x:(x[x['loan']=="yes"]['loan']).count())
plt.subplot(1,2,1)
data1.plot(kind='bar' , figsize=(20,5))
plt.ylabel("loan")
data2=bank.groupby('month').apply(lambda x:(x[x['loan']=="no"]['loan']).count())
plt.subplot(1,2,2)
data2.plot(kind='bar')
plt.ylabel("loan")
#print(data2)
plt.show()


# In[127]:


ax = sns.boxplot(y=bank["duration"], x = bank['y'])


# In[116]:


ax = sns.boxplot(y=bank["balance"], x = bank['y'])


# In[13]:


correlation=bank.corr()
plt.figure(figsize=(14,8))
sns.heatmap(correlation,annot=True,linewidth=0,vmin=-1,cmap="RdBu_r")


# In[15]:


ax=sns.violinplot(x="balance",y="education",hue="marital",data=bank)


# In[61]:


label_encoder = LabelEncoder()
bank['job'] = label_encoder.fit_transform(bank['job'])
bank['marital'] =label_encoder.fit_transform(bank['marital'])
bank['education'] = label_encoder.fit_transform(bank['education'])
bank['default'] = label_encoder.fit_transform(bank['default'])
bank['housing'] = label_encoder.fit_transform(bank['housing'])
bank['loan'] = label_encoder.fit_transform(bank['loan'])
bank['month'] = label_encoder.fit_transform(bank['month'])
#bank['day_of_week'] = label_encoder.fit_transform(bank['day_of_week'])
bank['poutcome'] = label_encoder.fit_transform(bank['poutcome'])
bank['y'] = label_encoder.fit_transform(bank['y'])


# In[62]:


bank.dtypes


# In[63]:


from sklearn.model_selection import train_test_split


# In[64]:


bank['contact'] = label_encoder.fit_transform(bank['contact'])


# In[65]:


def Age(ag):
    if ag > 60 : return 1
    elif 60 >ag >=45:
        return 2
    elif 45 > ag >=30:
        return 3 
    elif 30 > ag >=15 :
        return 4 
    else :
        return 5
bank['age']=bank['age'].map(Age)


# In[90]:


def Pdays(pd):
    if pd == 871 : return 1
     
    else :
        return 0
bank['pdays']=bank['pdays'].map(Pdays)


# In[91]:


bank.head()


# In[67]:


bank.info()


# In[68]:


bank.groupby('age').y.value_counts()


# In[89]:


bank.groupby('loan').y.value_counts()


# In[69]:


bank.groupby('pdays').y.value_counts()


# In[92]:


bank = bank.drop('poutcome', axis=1)


# In[93]:


x = bank.drop('y', axis=1)
Y = bank['y']

X_train, X_test, y_train, y_test = train_test_split(x,Y,test_size=0.30,random_state=11)


# In[153]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[95]:


model =RandomForestClassifier(n_estimators = 1000 , criterion = 'entropy' , random_state=3)
model.fit(X_train , y_train)
predicted = model.predict(X_test)


# In[97]:


from sklearn import metrics 

print('Accuracy:',round(metrics.accuracy_score(y_test,predicted),5))


# In[74]:


from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 


# In[75]:


results = confusion_matrix(y_test, predicted) 
print(results)
print( classification_report(y_test, predicted)) 


# In[76]:


from sklearn.svm import SVC


# In[154]:


model =  KNeighborsClassifier(n_neighbors=3)
model.fit(X_train , y_train )
predicted2 = model.predict(X_test)


# In[155]:


print('Accuracy:',round(metrics.accuracy_score(y_test,predicted2),5))


# In[156]:


results = confusion_matrix(y_test, predicted2) 
print(results)
print( classification_report(y_test, predicted2)) 


# In[159]:


from sklearn.tree import DecisionTreeClassifier
model =  DecisionTreeClassifier()
model.fit(X_train , y_train )
predicted3 = model.predict(X_test)


# In[160]:


print('Accuracy:',round(metrics.accuracy_score(y_test,predicted3),5))


# In[161]:


results = confusion_matrix(y_test, predicted3) 
print(results)
print( classification_report(y_test, predicted3)) 


# In[ ]:




