import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, normalize, LabelBinarizer, StandardScaler, MinMaxScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB  
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier

"""
Constants
"""
FEATURELIST = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'tempo', 'valence']    
CATEGORICALFEATURES = ['mode','genre'] 
SCALINGLIST = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'tempo', 'valence']

"""
Datasets
"""
preds_DF = pd.DataFrame()
train_DF = pd.read_csv("datasets/train.tsv", sep='\t', header=0)
test_DF = pd.read_csv("datasets/test.tsv", sep='\t', header=0)

# randomize rows
# train_DF = train_DF.sample(frac=1).reset_index(drop=True)

#drop NaN rows
train_DF = train_DF.dropna(axis=0, how='any')
test_DF = test_DF.dropna(axis=0, how='any')

# loudness +ve
train_DF['loudness'] = train_DF['loudness'].abs()

# Drop feature "title"
train_DF.drop(['title'], axis=1, inplace=True)
test_DF.drop(['title'], axis=1, inplace=True)

# LABEL ENCODING FOR MODE, GENRE
leMode = LabelEncoder()
leGenre = LabelEncoder()

train_DF['genre'] = leGenre.fit_transform(train_DF['genre'])
train_DF['mode'] = leMode.fit_transform(train_DF['mode'])
test_DF['mode'] = leMode.fit_transform(test_DF['mode'])

# Feature Normalization
# train_DF[SCALINGLIST]=(train_DF[SCALINGLIST]-train_DF[SCALINGLIST].mean())/train_DF[SCALINGLIST].std()
# for feature in SCALINGLIST:
#     train_DF[feature] = normalize()



"""
Model Data
"""
train_DF_X = train_DF[FEATURELIST]
train_DF_X = train_DF_X[:int(0.80*len(train_DF_X))] 
train_DF_Y = train_DF['genre']
train_DF_Y = train_DF_Y[:int(0.80*len(train_DF_Y))]

train_DF_X_CV = train_DF_X[int(0.80*len(train_DF_X)):]
train_DF_Y_CV = train_DF_Y[int(0.80*len(train_DF_Y)):]

test_DF_X = test_DF[FEATURELIST]

"""
Classifier
"""

modelLog = Pipeline(steps=[('logreg',linear_model.LogisticRegression(C=1e5))]) 
modelSVM = Pipeline(steps=[('SVC', SVC(gamma=2, C=1))])
modelMLP = Pipeline(steps=[('MLP',  MLPClassifier(alpha=1))])
modelGNB = Pipeline(steps=[('GNB',GaussianNB())])
modelRF = Pipeline(steps=[('RandomForest',RandomForestClassifier(max_depth=2, random_state=0))])
modelQDA = Pipeline(steps=[('QDA',QuadraticDiscriminantAnalysis())])
# modelDT = Pipeline(steps=[('DT', DecisionTreeClassifier(max_depth=None, max_features='sqrt', splitter='best', min_samples_split=2 ,min_samples_leaf=1))])
modelDT = Pipeline(steps=[('DT', DecisionTreeClassifier(max_depth=23))])
"""
CV Scores
"""

# modelLog.fit(train_DF_X, train_DF_Y)
# predsLog = modelLog.predict(train_DF_X_CV)
# print(accuracy_score(train_DF_Y_CV, predsLog))

# modelSVM.fit(train_DF_X, train_DF_Y)
# predsSVM = modelSVM.predict(train_DF_X_CV)
# print(accuracy_score(train_DF_Y_CV, predsSVM))

# modelMLP.fit(train_DF_X, train_DF_Y)
# predsMLP = modelMLP.predict(train_DF_X_CV)
# print(accuracy_score(train_DF_Y_CV, predsMLP))

# modelGNB.fit(train_DF_X, train_DF_Y)
# predsGNB = modelGNB.predict(train_DF_X_CV)
# print(accuracy_score(train_DF_Y_CV, predsGNB))

# modelGNB.fit(train_DF_X, train_DF_Y)
# predsGNB = modelGNB.predict(train_DF_X_CV)
# print(accuracy_score(train_DF_Y_CV, predsGNB))

# modelQDA.fit(train_DF_X, train_DF_Y)
# predsQDA = modelQDA.predict(train_DF_X_CV)
# print(accuracy_score(train_DF_Y_CV, predsQDA))

modelDT.fit(train_DF_X, train_DF_Y)
predsDT = modelDT.predict(train_DF_X_CV)
print(accuracy_score(train_DF_Y_CV, predsDT))   

preds = modelDT.predict(test_DF_X)


print(preds)

# """
# SUBMISSION CSV
# """

submissions_DF = pd.DataFrame({'genre' : leGenre.inverse_transform(preds), 'id': test_DF['id']})
submissions_DF[['id','genre']].to_csv("submission.csv", index=False)

print(submissions_DF['genre'].value_counts())
