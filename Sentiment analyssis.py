#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import nltk.classify.util
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.classify import NaiveBayesClassifier
import numpy as np
import re
import string
import nltk
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


os.chdir ("F:/Ana1/2AI")


# In[4]:


os.listdir(os.getcwd())


# In[65]:


csv = "1429_1.csv"
df = pd.read_csv(r'F:\Ana1\2AI\1429_1.csv\1429_1.csv')
df.head(4)


# In[6]:


data = df.copy()
data.describe()


# In[11]:


data.info()


# In[68]:


check =  permanent[permanent["reviews.rating"].isnull()]
check.head()


# In[10]:


senti= permanent[permanent["reviews.rating"].notnull()]
permanent.head()


# In[12]:


data["asins"].unique()


# In[13]:


asins_unique = len(data["asins"].unique())
print("Number of Unique ASINs: " + str(asins_unique))


# In[14]:


# **Visualizing the distributions of numerical variables:**

data.hist(bins=50, figsize=(20,15)) # builds histogram and set the number of bins and fig size (width, height)
plt.show()


# In[17]:


senti["senti"].value_counts().plot.bar()


# In[18]:


from sklearn.model_selection import StratifiedShuffleSplit
print("Before {}".format(len(data)))
dataAfter = data.dropna(subset=["reviews.rating"]) # removes all NAN in reviews.rating
print("After {}".format(len(dataAfter)))
dataAfter["reviews.rating"] = dataAfter["reviews.rating"].astype(int)


# In[19]:


split = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
for train_index, test_index in split.split(dataAfter, dataAfter["reviews.rating"]): 
    strat_train = dataAfter.reindex(train_index)
    strat_test = dataAfter.reindex(test_index)


# In[20]:


len(strat_train)


# In[21]:


# **Check to see if train/test sets were stratified proportionately in comparison to raw data.**

strat_train["reviews.rating"].value_counts()/len(strat_train) # value_count() counts all the values based on column


# In[22]:


len(strat_test)


# In[23]:


strat_test["reviews.rating"].value_counts()/len(strat_test)


# In[25]:


# # 4 Data Exploration (Training Set)

reviews = strat_train.copy()
reviews.head(3)


# In[26]:


len(reviews["name"].unique()), len(reviews["asins"].unique())


# In[27]:


# Working hypothesis: there are only 35 products based on the training data ASINs
# - One for each ASIN, but more product names (47)
# - ASINs are what's important here since we're concerned with products. There's a one to many relationship between ASINs and names
# - A single ASIN can have many names due to different vendor listings
# - There could also a lot of missing names/more unique names with slight variations in title (ie. 8gb vs 8 gb, NAN for product

reviews.info()


# In[28]:


reviews.groupby("asins")["name"].unique()


# In[29]:


# Lets see all the different names for this product that have 2 ASINs
different_names = reviews[reviews["asins"] == "B00L9EPT8O,B01E6AO69U"]["name"].unique()
for name in different_names:
    print(name)


# In[30]:


# **Confirmed our hypothesis that each ASIN can have multiple names. Therefore we should only really concern ourselves with which ASINs do well, not the product names.**

reviews[reviews["asins"] == "B00L9EPT8O,B01E6AO69U"]["name"].value_counts()


# In[31]:


fig = plt.figure(figsize=(16,10))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212, sharex = ax1)
reviews["asins"].value_counts().plot(kind="bar", ax=ax1, title="ASIN Frequency")
np.log10(reviews["asins"].value_counts()).plot(kind="bar", ax=ax2, title="ASIN Frequency (Log10 Adjusted)") 
plt.show()


# In[32]:


# Entire training dataset average rating
reviews["reviews.rating"].mean()


# In[33]:


asins_count_ix = reviews["asins"].value_counts().index
plt.subplots(2,1,figsize=(16,12))
plt.subplot(2,1,1)
reviews["asins"].value_counts().plot(kind="bar", title="ASIN Frequency")
plt.subplot(2,1,2)
sns.pointplot(x="asins", y="reviews.rating", order=asins_count_ix, data=reviews)
plt.xticks(rotation=90)
plt.show()


# In[34]:


plt.subplots (2,1,figsize=(16,12))
plt.subplot(2,1,1)
reviews["asins"].value_counts().plot(kind="bar", title="ASIN Frequency")
plt.subplot(2,1,2)
sns.pointplot(x="asins", y="reviews.doRecommend", order=asins_count_ix, data=reviews)
plt.xticks(rotation=90)
plt.show()


# In[35]:


corr_matrix = reviews.corr()
corr_matrix
# Here we can analyze reviews.ratings with asins


# In[36]:


reviews.info()


# In[37]:


counts = reviews["asins"].value_counts().to_frame()
counts.head()


# In[38]:


avg_rating = reviews.groupby("asins")["reviews.rating"].mean().to_frame()
avg_rating.head()


# In[39]:


table = counts.join(avg_rating)
table.head(30)


# In[40]:


plt.scatter("asins", "reviews.rating", data=table)
table.corr()


# In[41]:


def sentiments(rating):
    if (rating == 5) or (rating == 4):
        return "Positive"
    elif rating == 3:
        return "Neutral"
    elif (rating == 2) or (rating == 1):
        return "Negative"
# Add sentiments to the data
strat_train["Sentiment"] = strat_train["reviews.rating"].apply(sentiments)
strat_test["Sentiment"] = strat_test["reviews.rating"].apply(sentiments)
strat_train["Sentiment"][:20]


# In[43]:


# Prepare data
X_train = strat_train["reviews.text"]
X_train_targetSentiment = strat_train["Sentiment"]
X_test = strat_test["reviews.text"]
X_test_targetSentiment = strat_test["Sentiment"]
print(len(X_train), len(X_test))


# In[44]:


# Replace "nan" with space
X_train = X_train.fillna(' ')
X_test = X_test.fillna(' ')
X_train_targetSentiment = X_train_targetSentiment.fillna(' ')
X_test_targetSentiment = X_test_targetSentiment.fillna(' ')

# Text preprocessing and occurance counting
from sklearn.feature_extraction.text import CountVectorizer 
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train) 
X_train_counts.shape


# In[45]:


from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer(use_idf=False)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape


# In[46]:


# 6.3 Building a Pipeline from the Extracted Features
# We will use **Multinominal Naive Bayes** as our Classifier
# - Multinominal Niave Bayes is most suitable for word counts where data are typically represented as **word vector counts** (number of times outcome number X[i,j] is observed over the n trials), while also ignoring non-occurrences of a feature i
# - Naive Bayes is a simplified version of Bayes Theorem, where all features are assumed conditioned independent to each other (the classifiers), P(x|y) where x is the feature and y is the classifier


from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
clf_multiNB_pipe = Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), ("clf_nominalNB", MultinomialNB())])
clf_multiNB_pipe.fit(X_train, X_train_targetSentiment)


# In[49]:


import numpy as np
predictedMultiNB = clf_multiNB_pipe.predict(X_test)
np.mean(predictedMultiNB == X_test_targetSentiment)


# In[50]:


#Logistic Regression Classifier

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
clf_logReg_pipe = Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), ("clf_logReg", LogisticRegression())])
clf_logReg_pipe.fit(X_train, X_train_targetSentiment)

import numpy as np
predictedLogReg = clf_logReg_pipe.predict(X_test)
np.mean(predictedLogReg == X_test_targetSentiment)


# In[51]:


#Support Vector Machine Classifier

from sklearn.svm import LinearSVC
clf_linearSVC_pipe = Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), ("clf_linearSVC", LinearSVC())])
clf_linearSVC_pipe.fit(X_train, X_train_targetSentiment)

predictedLinearSVC = clf_linearSVC_pipe.predict(X_test)
np.mean(predictedLinearSVC == X_test_targetSentiment)


# In[52]:


#Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier
clf_decisionTree_pipe = Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), 
                                  ("clf_decisionTree", DecisionTreeClassifier())])
clf_decisionTree_pipe.fit(X_train, X_train_targetSentiment)

predictedDecisionTree = clf_decisionTree_pipe.predict(X_test)
np.mean(predictedDecisionTree == X_test_targetSentiment)


# In[53]:


#Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier
clf_randomForest_pipe = Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), ("clf_randomForest", RandomForestClassifier())])
clf_randomForest_pipe.fit(X_train, X_train_targetSentiment)

predictedRandomForest = clf_randomForest_pipe.predict(X_test)
np.mean(predictedRandomForest == X_test_targetSentiment)


# In[54]:


# - Here we will run a **Grid Search** of the best parameters on a grid of possible values, instead of tweaking the parameters of various components of the chain (ie. use_idf in tfidftransformer)
# - We will also run the grid search with LinearSVC classifier pipeline, parameters and cpu core maximization
# - Then we will fit the grid search to our training data set
# - Next we will use our final classifier (after fine-tuning) to test some arbitrary reviews
# - Finally we will test the accuracy of our final classifier (after fine-tuning)

from sklearn.model_selection import GridSearchCV
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],    
             'tfidf__use_idf': (True, False), 
             } 
gs_clf_LinearSVC_pipe = GridSearchCV(clf_linearSVC_pipe, parameters, n_jobs=-1)
gs_clf_LinearSVC_pipe = gs_clf_LinearSVC_pipe.fit(X_train, X_train_targetSentiment)
new_text = ["The tablet is good, really liked it.", # positive
            "The tablet is ok, but it works fine.", # neutral
            "The tablet is not good, does not work very well."] # negative

X_train_targetSentiment[gs_clf_LinearSVC_pipe.predict(new_text)]


# In[55]:


# **Results:**
# - After testing some arbitrary reviews, it seems that our features is performing correctly with Positive, Neutral, Negative results

predictedGS_clf_LinearSVC_pipe = gs_clf_LinearSVC_pipe.predict(X_test)
np.mean(predictedGS_clf_LinearSVC_pipe == X_test_targetSentiment)


# In[69]:


# - Here we see that the best mean score of the grid search is 93.65% which is very close to our accuracy level of 94.08%
# - Our best estimator here is also displayed
# - Lastly, our best parameters are true for use_idf in tfidf, and ngram_range between 1,2

for performance_analysis in (gs_clf_LinearSVC_pipe.best_score_, 
                             gs_clf_LinearSVC_pipe.best_estimator_, 
                             gs_clf_LinearSVC_pipe.best_params_):
        print(performance_analysis)


# In[70]:


# Below is the summary of the classification report:
# - Precision: determines how many objects selected were correct
# - Recall: tells you how many of the objects that should have been selected were actually selected
# - F1 score measures the weights of recall and precision (1 means precision and recall are equally important, 0 otherwise)
# - Support is the number of occurrences of each class

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

print(classification_report(X_test_targetSentiment, predictedGS_clf_LinearSVC_pipe))
print('Accuracy: {}'. format(accuracy_score(X_test_targetSentiment, predictedGS_clf_LinearSVC_pipe)))


# In[58]:


from sklearn import metrics
metrics.confusion_matrix(X_test_targetSentiment, predictedGS_clf_LinearSVC_pipe)


# In[ ]:




