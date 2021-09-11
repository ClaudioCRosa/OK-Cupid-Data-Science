#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

df = pd.read_csv('profiles.csv')
#print(len(df.location))
loc_col = df.location

true_list = []
for loc in loc_col:
    true_list.append("california" in loc)
#print(true_list)
#All profiles in the sample are from Calofornia in the US
#print(df.income)
print(df.head(10))
#print(df.religion.unique())


# In[2]:


list(df.columns)
#Column description:
#age: Age of the individual (continuous var.)
#body_type: Self-described body type (categorical var.)
#diet: Type of diet the individual follow (categorical var.)
#drinks: Relationship the profile has with drinking (categorical var.)
#drugs: Relationship the profile has with doing drugs (categorical var.)
#education: Education attained by the individual (categorical var.)
#essays (0-9): Answers to certain questions (string)
#ethinicity: Individual's ethnicity (categorical var.)
#height: Individual's height (continuous var.)
#income: Yearly income (continous)
#job: The individual's employment (categorical var.)
#last_online: Date of the times time the individual was online (date)
#location: Individual's location in California (categorical var.)
#offspring: Individual's stance on having children (categorical var.)
#orientation: Individual's sexual orientation (categorical var.)
#pets: If the individual has pets (categorical var.)
#religion: Religion of the individual (categorical var.)
#sex: Individual's sex (categorical var.)
#sign: Individual's sign (categorical var.)
#smokes: Relationship the profile has with smoking (categorical var.)
#speaks: Languages spoken by the individual (categorical var.)
#status: If the individual is currently in a relationship (categorical var.)


# In[3]:


#Cleaning/transforming the data
#Signs:
#print(df.sign.value_counts())
#print(df.head())
df["clean_signs"] = df["sign"].str.split().str.get(0)
#print(df.clean_signs.value_counts())
#Income:
df["defined_income"] = df["income"].apply(lambda x: 1 if x != -1 else 0)
df["defined_income_labels"] = df["defined_income"].apply(lambda x: "provided in the profile" if x == 1 else "unknown")
#Religion:
df["clean_rel"] = df["religion"].str.split().str.get(0)

print(df.head())


# In[4]:


#Only around one fifth of the profiles include their income
plt.pie(df.defined_income.value_counts(), autopct="%1.0f%%")
plt.legend(df.defined_income_labels)
plt.show()
print(df.defined_income.value_counts())


# In[5]:


#There is a prevalence of younger addults in the sample. 
#There are more men than women in the sample
#However, distribution-wise, there is no noticeable different distributions between the given genders
ax = sns.violinplot(y=df.age, x= df.sex)
ax.set_xlabel("Gender")
ax.set_ylabel("Age")
ax.set_title("Profile age distribution")
plt.show()
plt.clf()
ax2 = sns.displot(data=df, x="age", hue="sex", bins=30)
plt.title("Profile age distribution")
plt.show()
plt.clf()


# In[6]:


ax1 = sns.countplot(data=df, x=df.drinks)
ax1.set_title("Drinking among the profiles")
plt.show()
plt.clf()
ax2 = sns.countplot(data=df, x=df.drugs)
ax2.set_title("Drug usage among the profiles")
plt.show()
ax3 = sns.countplot(data=df, x=df.smokes)
ax3.set_title("Smoking among the profiles")
plt.show()


# In[7]:


ax2 = sns.countplot(data=df, x=df.clean_rel)
plt.xticks(rotation = 25)
plt.title("Religious affiliation of the participants")
plt.show()
plt.clf()
ax3 = sns.countplot(data=df, x=df.orientation, hue=df.sex)
ax3.set_title("Sexual orientation of the participants")
plt.show()


# In[8]:


df = df.drop(df[df.income == -1].index)
cols = ["body_type", "diet", "drinks", "drugs", "education", "height", "job", "location", "clean_rel", "sex", "clean_signs", "smokes", "income", "age"]
df = df[cols].dropna()
df.shape


# In[9]:


for col in cols[:-2]:
    df = pd.get_dummies(df, columns=[col], prefix = [col])


# In[10]:


print(df.head())


# In[11]:


col_length = len(df.columns)

X = df.iloc[:, 1:col_length]
Y = df.iloc[:, 0:1]

from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 1)

Y_train = Y_train.to_numpy().ravel()
Y_test = Y_test.to_numpy().ravel()


# In[12]:


from sklearn.linear_model import LogisticRegression

reg = LogisticRegression(max_iter=10000)
reg.fit(X_train, Y_train)


# In[13]:


reg.predict(X_test)
reg.score(X_test, Y_test)
#A score of 37% indicates that the model while somewhat flawed is better than random chance at guessing one's income from one's profile information


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
performance = []
for k in range(1, 101):
    kmodel = KNeighborsClassifier(n_neighbors = k)
    kmodel.fit(X_train, Y_train)
    performance.append(kmodel.score(X_test, Y_test))
    
    
plt.plot(range(1, 101), performance)
plt.show()

print(max(performance))
#The K-neighbours model does not perform better then the logistic regression. It guesses the person's income correctly
#given the profile information slightly more than one third of the times


# In[ ]:


from sklearn.tree import DecisionTreeClassifier

tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, Y_train) 
tree_model.score(X_test, Y_test)

#The decision tree performs worse than the two models above guessing correctly only one fourth of the times


# In[ ]:





# In[ ]:




