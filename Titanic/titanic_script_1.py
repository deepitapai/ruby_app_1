from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


import numpy as np
import pandas as pd

input_file = "/home/deepita/Kaggle/Titanic/final_postregression.csv"
input_test_file = "/home/deepita/Kaggle/Titanic/test_after_missing.csv"
df = pd.read_csv(input_file,header=0)
df_test = pd.read_csv(input_test_file,header=0)
#print df
#df = df._get_numeric_data()

#msk = np.random.rand(len(df)) < 0.8
train = df
train_target = train['Survived']
train = train.drop('Survived',axis=1)
#print train
test = df_test
#test_target = test['Survived']
#test = test.drop('Survived',axis=1)

print len(train)
print len(test)
#print newsgroups_train.filenames

#print newsgroups_test.filenames.shape

#print vectorizer
#print X_train

classifier = Perceptron(n_iter=5000, eta0=0.3)
classifier.fit_transform(train, train_target )
predictions = classifier.predict(test)
#print predictions
#print classification_report(test_target, predictions)
print predictions
np.savetxt('test_final.csv',predictions)

#cm = confusion_matrix(test_target,predictions)
#print cm
#print classifier.score(train,train_target)