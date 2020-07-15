
#importing libraries
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn import metrics
from sklearn.metrics import fbeta_score
from sklearn.metrics import recall_score, roc_auc_score, brier_score_loss, confusion_matrix
import imblearn as imb
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import make_scorer
from sklearn.feature_selection import RFE, RFECV
from sklearn.pipeline import Pipeline
from numpy import mean, std
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import  mutual_info_classif


#Importing data
filename = r"creditcard.csv"
df = pd.read_csv(filename)
for col in ['Class']:
    df[col] = df[col].astype('category')

#splitting into features and class
X = df.loc[:, 'Time':'Amount']
y = df.loc[:, 'Class']

#############################################################################
#DECISION TREE WITHOUT FILTERING

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

#KFold Cross (with time series split) Validation approach
tss = TimeSeriesSplit(n_splits = 3)
tss.split(X)

# Initialize the accuracy of the models to blank list. The accuracy of each model will be appended to this list
F_measure_model = []
roc_auc_model = []
brier_score_model = []
confusion_matrix_model = []


# Iterate over each train-test split
for train_index, test_index in tss.split(X):
    # Split train-test
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    # Train the model
    model = clf.fit(X_train, y_train)
    # Append metrics to the list
    F_measure_model.append(fbeta_score(y_test, model.predict(X_test),beta=2))
    roc_auc_model.append(roc_auc_score(y_test, model.predict(X_test)))
    brier_score_model.append(brier_score_loss(y_test, model.predict(X_test)))
    confusion_matrix_model.append(confusion_matrix(y_test, model.predict(X_test)))

# Print the model metrics
metrics = pd.DataFrame(
    {'Result': ["Average"],
    'F_measure': np.average(F_measure_model),
    'ROC_AUC': np.average(roc_auc_model),
    'Brier_Score' : np.average(brier_score_model),
    })
print("Model Metrics:")
print(metrics)

print("Confusion Matrix")
confusion = pd.DataFrame(
    {'Trial': ["Trial 1", "Trial 2", "Trial 3"],
    'Confusion_Matrix' : confusion_matrix_model
    })
print(confusion)

'''
###############################################################################
#APPLYING RFE FEATURE SELECTION METHOD TO IDENTIFY OPTIMAL NUMBER OF COLUMNS

# get a list of models to evaluate
def get_models():
	models = dict()
	for i in range(10, 25):
		rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=i)
		model = DecisionTreeClassifier()
		models[str(i)] = Pipeline(steps=[('s',rfe),('m',model)])
	return models

#make balanced scorer
scorer = make_scorer(balanced_accuracy_score)

# evaluate a give model using cross-validation
def evaluate_model(model):
	cv = TimeSeriesSplit(n_splits = 3)
	scores = cross_val_score(model, X, y, scoring=scorer, cv=cv, n_jobs=-1, error_score='raise')
	return scores

# get the models to evaluate
models = get_models()

# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model)
	results.append(scores)
	names.append(name)
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))

###############################################################################
#IDENTIFY COLUMN IDENITIFED AS PART OF OPTIONAL NUMBER OF COLUMNS (RFE)

rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=19)
# fit RFE
rfe.fit(X, y)
# summarize all features
for i in range(X.shape[1]):
	print('Column: %d, Selected %s, Rank: %.3f' % (i, rfe.support_[i], rfe.ranking_[i]))
'''
'''
##############################################################################
#APPLYING SELECTKBEST FEATURES WITH MUTUAL INFORMATION

# Import the necessary libraries first
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import  mutual_info_classif

for i in range(10,25):
    sel_mutual = SelectKBest(mutual_info_classif, k=i)
    X_train_mutual = sel_mutual.fit_transform(X,y)
    print(sel_mutual.get_support())

###############################################################################
# APPLYING DECISION TREE ALL COMBINATIONS ABOVE TO FIND OPTIMAL NUMBER OF FEATURES


'''
'''
##############################################################################
#STEP FORWARD FEATURE MODEL SELECTION

import mlxtend as mlx
from mlxtend.feature_selection import SequentialFeatureSelector as sfs

#KFold Cross (with time series split) Validation approach
tss = TimeSeriesSplit(n_splits = 3)
tss.split(X)

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

#make balanced scorer
scorer = make_scorer(balanced_accuracy_score)

# Iterate over each train-test split
for train_index, test_index in tss.split(X):
    # Split train-test
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Build step forward feature selection
sfs1 = sfs(clf,
        k_features=24,
        forward=True,
        floating=False,
        verbose=2,
        scoring=scorer)
# Perform SFFS
sfs1 = sfs1.fit(X_train, y_train)

# Which features?
feat_cols = list(sfs1.k_feature_idx_)
print(feat_cols)

'''
'''
#############################################################################
#FINDING THE OPTIMAL NUMBER OF COLUMNS AUTOMATICALLY WITH RFECV (error)

# create pipeline
rfecv_model = RFECV(estimator=DecisionTreeClassifier())
model = DecisionTreeClassifier()
pipeline = Pipeline(steps=[('s',rfecv_model),('m',model)])

#make balanced scorer
scorer = make_scorer(balanced_accuracy_score)

# evaluate model
cv = TimeSeriesSplit(n_splits=3)
result = cross_validate(pipeline, X, y, scoring=scorer,
                          cv=cv, return_estimator=True)

#feature selected for each iteration
for iter, pipe in enumerate(result['estimator']):
    print(f'Iteration no: {iter}')
    for i in range(X.shape[1]):
        print('Column: %d, Selected %s, Rank: %d' %
            (i, pipe['s'].support_[i], pipe['s'].ranking_[i]))

#Identify which columns were used
print(X.shap
for i in range(X.shape[1]):
	print('Column: %d, Selected %s, Rank: %.3f' % (i, rfecv_model.support_[i], rfecv_model.ranking_[i]))
'''
