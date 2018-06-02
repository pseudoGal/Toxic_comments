import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split

#from skmultilearn.problem_transform import LabelPowerset
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import hamming_loss
from sklearn.metrics import roc_auc_score

#for train data

train = pd.read_csv('train.csv')
train_id = train.id
x_comments = train.comment_text
print(x_comments.shape)
print(type(x_comments))
del train['id']
del train['comment_text']
train_length = x_comments.shape[0]
#test data

test = pd.read_csv('test.csv')
test_id = test.id
del test['id']
y_comments = test.comment_text
y_comments.fillna('unknown',inplace=True)
del test['comment_text']
y_comment_final = y_comments[pd.notnull(y_comments)]


#collect total unique words in test & train in data

corpus = pd.concat([x_comments,y_comment_final])
#countVectorizer
vectorizer = CountVectorizer()
vect = vectorizer.fit_transform(corpus)
#extrct toenized words
#analyze = vectorizer.build_analyzer()
#print(analyze(str(x_comments)))
#tfidf
transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(vect)
#split train-test
x_comment = tfidf[:train_length]
y_comment = tfidf[train_length:]
#numerical model
x_train,x_test,y_train,y_test = train_test_split(x_comment,train,test_size= 0.32,random_state = 42)

#classifier = LabelPowerset(MultinomialNB())
#classifier.fit(x_train, y_train)
#predictions = classifier.predict(x_test)

classif = OneVsRestClassifier(SVC(kernel='linear'))
classif.fit(x_train,y_train)
x_predict = classif.predict(x_test)
x_predict = pd.DataFrame(x_predict)
h_loss = hamming_loss(x_predict, y_test)
roc_auc = roc_auc_score(x_predict, y_test)
#classifier
y_predict = classif.predict(y_comment)
y_predict = pd.DataFrame({'id' : test_id})
#saving output
output = pd.DataFrame({'id' : test_id})

col = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

output[col]= y_predict
output.to_csv('toxic.csv', index = False)


