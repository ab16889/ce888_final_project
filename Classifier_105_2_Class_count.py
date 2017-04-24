
# coding: utf-8

# In[42]:

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
#get_ipython().magic(u'matplotlib inline')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
import re
from pandas import DataFrame, Series
import json
import pickle
import matplotlib.pyplot as plt
import itertools



from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN

import nltk.stem as st
from nltk.stem import WordNetLemmatizer
from utils import load_sent_word_net
from sklearn.base import BaseEstimator
from sklearn.pipeline import FeatureUnion

from sklearn.model_selection import TimeSeriesSplit

from sklearn.pipeline import Pipeline

from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import GridSearchCV


from sklearn.metrics import confusion_matrix, f1_score, precision_score, make_scorer, accuracy_score, recall_score, classification_report


# In[38]:

#for use in browser
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        t = "(%.2f)"%(cm[i, j])
        #print t
#         plt.text(j, i, t,
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[2]:

data = pd.read_csv('Combined_News_DJIA.csv')
#print(data.head())
sent_word_net = load_sent_word_net()


# In[3]:

#vix_data = pd.read_csv('VIX_up_dn.csv')
#three_class = pd.read_csv('three_Class_dow.csv')


# three_class.head()

# In[4]:

#y_df_vix = vix_data['Change'].copy()
#y_df_three = three_class['Change'].copy()
#del y_df_vix['Date']


# y_df_three.head()

# In[5]:

data = data.dropna()

X_df = data.copy()
del X_df['Label']

del X_df['Date']

y_df = data['Label'].copy()


# In[6]:

trainheadlines = []
for row in range(0,len(X_df.index)):
    trainheadlines.append(''.join(str(x) for x in X_df.iloc[row,2:27]))


# In[7]:

#trainheadlines


# In[8]:

#X_df.head()


# In[10]:

#X_df.shift(1).head()


# In[11]:

X_df['combined'] = data.iloc[:,2:27].apply(lambda row: ''.join(str(row.values)), axis=1)

X_df_new = X_df['combined'].shift(1)

#X_train = X_df_new[:1500]
#X_test = X_df_new[1501:1985]
X_train = Series(trainheadlines[1:1500])#starting from 1 as 0 in X_df_new is now NaN
X_test = Series(trainheadlines[1501:1985])
y_train = y_df[1:1500]
y_test = y_df[1501:1985]
#y_train_vix = y_df_vix[1:1500]
#y_test_vix = y_df_vix[1501:1985]
#y_train_three = y_df_three[1:1500]
#y_test_three = y_df_three[1501:1985]


# y_train_three.head()

# In[12]:


poscache_filename = "poscache.json"
try:
    poscache = json.load(open(poscache_filename, "r"))
except IOError:
    poscache = {}


# In[13]:

emo_repl = {
    # positive emoticons
    "&lt;3": " good ",
    ":d": " good ",  # :D in lower case
    ":dd": " good ",  # :DD in lower case
    "8)": " good ",
    ":-)": " good ",
    ":)": " good ",
    ";)": " good ",
    "(-:": " good ",
    "(:": " good ",

    # negative emoticons:
    ":/": " bad ",
    ":&gt;": " sad ",
    ":')": " sad ",
    ":-(": " bad ",
    ":(": " bad ",
    ":S": " bad ",
    ":-S": " bad ",
}

emo_repl_order = [k for (k_len, k) in reversed(
    sorted([(len(k), k) for k in list(emo_repl.keys())]))]

re_repl = {
    r"\br\b": "are",
    r"\bu\b": "you",
    r"\bhaha\b": "ha",
    r"\bhahaha\b": "ha",
    r"\bdon't\b": "do not",
    r"\bdoesn't\b": "does not",
    r"\bdidn't\b": "did not",
    r"\bhasn't\b": "has not",
    r"\bhaven't\b": "have not",
    r"\bhadn't\b": "had not",
    r"\bwon't\b": "will not",
    r"\bwouldn't\b": "would not",
    r"\bcan't\b": "can not",
    r"\bcannot\b": "can not",
}


# In[14]:

class LinguisticVectorizer(BaseEstimator):

    def get_feature_names(self):
        return np.array(['sent_neut', 'sent_pos', 'sent_neg',
                         'nouns', 'adjectives', 'verbs', 'adverbs',
                         'allcaps', 'exclamation', 'question'])

    def fit(self, documents, y=None):
        return self

    def _get_sentiments(self, d):
        
        
        #sent = tuple(nltk.word_tokenize(d.decode('utf-8')))
        sent = tuple(nltk.word_tokenize(d))
        if poscache is not None:
            if d in poscache:
                tagged = poscache[d]
            else:
                poscache[d] = tagged = nltk.pos_tag(sent)
        else:
            tagged = nltk.pos_tag(sent)

        pos_vals = []
        neg_vals = []

        nouns = 0.
        adjectives = 0.
        verbs = 0.
        adverbs = 0.

        for w, t in tagged:
            p, n = 0, 0
            sent_pos_type = None
            if t.startswith("NN"):
                sent_pos_type = "n"
                nouns += 1
            elif t.startswith("JJ"):
                sent_pos_type = "a"
                adjectives += 1
            elif t.startswith("VB"):
                sent_pos_type = "v"
                verbs += 1
            elif t.startswith("RB"):
                sent_pos_type = "r"
                adverbs += 1

            if sent_pos_type is not None:
                sent_word = "%s/%s" % (sent_pos_type, w)

                if sent_word in sent_word_net:
                    p, n = sent_word_net[sent_word]

            pos_vals.append(p)
            neg_vals.append(n)

        l = len(sent)
        sum_pos_val = np.sum(pos_vals)
        sum_neg_val = np.sum(neg_vals)
        avg_pos_val = np.mean(pos_vals)
        avg_neg_val = np.mean(neg_vals)
        

        return [1 - avg_pos_val - avg_neg_val, sum_pos_val, sum_neg_val,
                nouns / l, adjectives / l, verbs / l, adverbs / l]

    def transform(self, documents):
        obj_val, pos_val, neg_val, nouns, adjectives, verbs, adverbs = np.array(
            [self._get_sentiments(d) for d in documents]).T

        allcaps = []
        exclamation = []
        question = []

        for d in documents:
            allcaps.append(
                np.sum([t.isupper() for t in d.split() if len(t) > 2]))

            exclamation.append(d.count("!"))
            question.append(d.count("?"))

        result = np.array(
            [obj_val, pos_val, neg_val, nouns, adjectives, verbs, adverbs, allcaps,
             exclamation, question]).T

        return result


# ling1 = LinguisticVectorizer()
# count1 = CountVectorizer()

# X_df_ling = ling1.transform(trainheadlines)

# X_df_ling[0]

# sentiment_array_pos = X_df_ling[:,1]

# sentiment_array_neg = X_df_ling[:,2]

# sentiment_array_net = (sentiment_array_pos/sentiment_array_neg) + 0.25

# sentiment_array_net

# np.savetxt("net_sentiment.csv", sentiment_array_net, delimiter=',')

# #pd.set_printoptions(threshold=np.inf)
# print(X_train.iloc[0].values)

# In[15]:

def preprocessor(tweet):
        #print(tweet)
        tweet = tweet.lower()
        #wordnet_lemmatizer = WordNetLemmatizer()
        out = tweet
        #for word in tweet.split():
         #   out += ' ' + wordnet_lemmatizer.lemmatize(word)

        for k in emo_repl_order:
            out = out.replace(k, emo_repl[k])
        for r, repl in re_repl.items():
            out = re.sub(r, repl, out)

        return out.replace("-", " ").replace("_", " ")


# In[29]:

def create_union_model_svc(params=None):
    

    tfidf_ngrams = TfidfVectorizer(preprocessor=preprocessor,
                                   ngram_range=(1,3),
                                   stop_words='english',
                                   analyzer="word")
    count_vectorizer = CountVectorizer(
                                       binary=False, 
                                       preprocessor=preprocessor, 
                                       ngram_range=(1,3), 
                                       stop_words='english',
                                        #lowercase=False,
                                    min_df=5,
                                      )
    hash_vectorizer = HashingVectorizer(
                                       binary=False, 
                                       preprocessor=preprocessor, 
                                       ngram_range=(1,5), 
                                       stop_words='english'
                                      )
    ling_stats = LinguisticVectorizer()
    #all_features = FeatureUnion([('ling', ling_stats)])
    #print(all_features)
    #all_features = FeatureUnion([('count', count_vectorizer)])
    all_features = FeatureUnion([('ling', ling_stats), ('count', count_vectorizer)])
    #all_features = FeatureUnion( [('ling', ling_stats), ('tfid', tfidf_ngrams)])
    #all_features = FeatureUnion([('tfidf', tfidf_ngrams)])
    #all_features = FeatureUnion([('ling', ling_stats)])
    #clf = MultinomialNB()
    #clf_random = RandomForestClassifier(n_jobs=3, random_state=0, n_estimators=10)
    #clf = ExtraTreesClassifier(n_jobs=3, random_state=0, n_estimators=10)
    #clf = SVC( C=100.0, kernel='poly', degree=3, random_state=0) 
    clf = SVC()
    #clf = KNN(n_neighbors=5, metric="manhattan")
    #clf = MLPClassifier(hidden_layer_sizes=(100,100), random_state=0, max_iter=1000, shuffle=False)
    pipeline = Pipeline([('all', all_features), 
                         #('feature_selector', SelectKBest(chi2, 750)  ),
                         ('clf', clf)])

    if params:
        pipeline.set_params(**params)

    return pipeline


def create_union_model_random(params=None):
    count_vectorizer = CountVectorizer(
                                       binary=False, 
                                       preprocessor=preprocessor, 
                                       ngram_range=(1,3), 
                                       stop_words='english',
                                        #lowercase=False,
                                    min_df=5,
                                      )
    hash_vectorizer = HashingVectorizer(
                                       binary=False, 
                                       preprocessor=preprocessor, 
                                       ngram_range=(1,5), 
                                       stop_words='english'
                                      )
    ling_stats = LinguisticVectorizer()
    #all_features = FeatureUnion([('ling', ling_stats)])
    #print(all_features)
    #all_features = FeatureUnion([('count', count_vectorizer)])
    all_features = FeatureUnion([('ling', ling_stats), ('count', count_vectorizer)])

    clf = RandomForestClassifier(n_jobs=3, random_state=0, n_estimators=10)

    pipeline = Pipeline([('all', all_features), 
                         #('feature_selector', SelectKBest(chi2, 750)  ),
                         ('clf', clf)])

    if params:
        pipeline.set_params(**params)

    return pipeline

def create_union_model_mlp(params=None):
    count_vectorizer = CountVectorizer(
                                       binary=False, 
                                       preprocessor=preprocessor, 
                                       ngram_range=(1,3), 
                                       stop_words='english',
                                        #lowercase=False,
                                    min_df=5,
                                      )
    hash_vectorizer = HashingVectorizer(
                                       binary=False, 
                                       preprocessor=preprocessor, 
                                       ngram_range=(1,5), 
                                       stop_words='english'
                                      )
    ling_stats = LinguisticVectorizer()
    #all_features = FeatureUnion([('ling', ling_stats)])
    #print(all_features)
    #all_features = FeatureUnion([('count', count_vectorizer)])
    all_features = FeatureUnion([('ling', ling_stats), ('count', count_vectorizer)])
    clf = MLPClassifier(hidden_layer_sizes=(100,100), random_state=0, max_iter=1000, shuffle=False)

    

    pipeline = Pipeline([('all', all_features), 
                         #('feature_selector', SelectKBest(chi2, 750)  ),
                         ('clf', clf)])

    if params:
        pipeline.set_params(**params)

    return pipeline



#classes = np.unique(y_train_three)


# In[377]:

param_grid_svc = param_grid = dict(

                      #clf__n_estimators=[10,20,30,50,75,100],
                      clf__C=np.logspace(0,3,num=20),
                      #clf__hidden_layer_sizes=[ (10,), (20,), (30,), (40,), (50,), (100,)],
                      all__count__ngram_range=[(1,1),(1,2), (1,3), (1,4), (1,5), (2,2), (2,3)],
                      #all__count__min_df = [2,3,4,5,6,7,8],
                      #all__count__binary=[False]
                      #vect__ngram_range=[(1, 1), (1, 2), (1, 3)],
                      #clf__alpha=[0, 0.01, 0.05, 0.1, 0.5, 1],
                      )

param_grid_random = param_grid = dict(

                      clf__n_estimators=[5,10,15,20,30,50,75,100],
                      #clf__C=np.logspace(0,3,num=2),
                      #clf__hidden_layer_sizes=[ (10,), (20,), (30,), (40,), (50,), (100,)],
                      all__count__ngram_range=[(1,1),(1,2), (1,3), (1,4), (1,5), (2,2), (2,3)],
                      #all__count__min_df = [2,3,4,5,6,7,8],
                      #all__count__binary=[False]
                      #vect__ngram_range=[(1, 1), (1, 2), (1, 3)],
                      #clf__alpha=[0, 0.01, 0.05, 0.1, 0.5, 1],
                      )

param_grid_mlp = param_grid = dict(

                      #clf__n_estimators=[5,10,15,20,30,50,75,100],
                      #clf__C=np.logspace(0,3,num=2),
                      clf__hidden_layer_sizes=[ (10,), (20,), (30,), (40,), (50,), (100,), (5,5), (10,10), (20,20), (30,30)],
                      all__count__ngram_range=[(1,1),(1,2), (1,3), (1,4), (1,5), (2,2), (2,3)],
                      #all__count__min_df = [2,3,4,5,6,7,8],
                      #all__count__binary=[False]
                      #vect__ngram_range=[(1, 1), (1, 2), (1, 3)],
                      #clf__alpha=[0, 0.01, 0.05, 0.1, 0.5, 1],
                      )
def __grid_search_model(clf_factory, X, Y, param_grid):
    print("Starting grid search now")
    cv = TimeSeriesSplit(n_splits=10)

    
    f1_args = dict(average='macro')
    grid_search = GridSearchCV(clf_factory(),
                               param_grid=param_grid,
                               cv=cv,
                               n_jobs=10,
                               #scoring=make_scorer(f1_score, {'average':'macro'}),
                               scoring=make_scorer(f1_score, **f1_args),
                               #scoring=make_scorer(precision_score, {'average':'macro'}),
                               verbose=10)
    grid_search.fit(X, Y)
    
    clf = grid_search.best_estimator_
    print(clf)

    return clf




#union_model = create_union_model()
best_svc_two_ling_count = __grid_search_model(create_union_model_svc, X_train, y_train, param_grid=param_grid_svc)
print(best_svc_two_ling_count.steps)
best_svc_two_ling_count.fit(X_train, y_train )
svc_two_pred = best_svc_two_ling_count.predict(X_test)
print(classification_report(y_true=y_test, y_pred=svc_two_pred))
print(confusion_matrix(y_true=y_test, y_pred=svc_two_pred))
#print(random_pred)
pickle.dump(best_svc_two_ling_count, open("best_svc_ling_count_two.p", "wb"))

##start Random forest grid search

best_random_two_ling_count = __grid_search_model(create_union_model_random, X_train, y_train, param_grid=param_grid_random)
print(best_random_two_ling_count.steps)
best_random_two_ling_count.fit(X_train, y_train )
random_two_pred = best_random_two_ling_count.predict(X_test)
print(classification_report(y_true=y_test, y_pred=random_two_pred))
print(confusion_matrix(y_true=y_test, y_pred=random_two_pred))
#print(random_pred)
pickle.dump(best_random_two_ling_count, open("best_random_ling_two_count.p", "wb"))

#start MPL grid search

best_mlp_two_ling_count = __grid_search_model(create_union_model_mlp, X_train, y_train, param_grid=param_grid_mlp)
print(best_mlp_two_ling_count.steps)
best_mlp_two_ling_count.fit(X_train, y_train)
mlp_two_pred = best_mlp_two_ling_count.predict(X_test)
print(classification_report(y_true=y_test, y_pred=mlp_two_pred))
print(confusion_matrix(y_true=y_test, y_pred=mlp_two_pred))
#print(random_pred)
pickle.dump(best_mlp_two_ling_count, open("best_mlp_ling_two_count.p", "wb"))






