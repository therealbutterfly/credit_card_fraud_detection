
#importing libraries
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn import metrics
from sklearn.metrics import balanced_accuracy_score, make_scorer, average_precision_score
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.feature_selection import RFE, RFECV
from sklearn.pipeline import Pipeline
from numpy import mean, std

#Importing data
filename = r"creditcard.csv"
df = pd.read_csv(filename)
for col in ['Class']:
    df[col] = df[col].astype('category')

#splitting into features and class
X = df.loc[:, 'Time':'Amount']
y = df.loc[:, 'Class']

'''

####### Decision Tree without filtering


# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

#KFold Cross (with time series split) Validation approach
tss = TimeSeriesSplit(n_splits = 3)
tss.split(X)

# Initialize the accuracy of the models to blank list. The accuracy of each model will be appended to this list
balanced_accuracy_model = []
average_precision_model = []
f1_score_model = []
roc_auc_model = []

# Iterate over each train-test split
for train_index, test_index in tss.split(X):
    # Split train-test
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    # Train the model
    model = clf.fit(X_train, y_train)
    # Append metrics to the list
    balanced_accuracy_model.append(balanced_accuracy_score(y_test, model.predict(X_test)))
    average_precision_model.append(average_precision_score(y_test, model.predict(X_test)))
    f1_score_model.append(f1_score(y_test, model.predict(X_test)))
    roc_auc_model.append(roc_auc_score(y_test, model.predict(X_test)))

# Print the model metrics
metrics = pd.DataFrame(
    {'Trial': ["Trial1", "Trial2", "Trial3"],
    'Balanced Accuracy': balanced_accuracy_model,
    'Average Precision': average_precision_model,
    'F1 Score': f1_score_model,
    'ROC_AUC': roc_auc_model
    })
print("Model Metrics:")
print(metrics)
'''

'''
#### Applying RFE Feature Selection Method with manual checking...

# get a list of models to evaluate
def get_models():
	models = dict()
	for i in range(15, 25):
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
	print('>%s %.3f (%.3f)' % (name, mean(scores)))

for i in range(X.shape[1]):
	print('Column: %d, Selected %s, Rank: %.3f' % (i, rfe.support_[i], rfe.ranking_[i]))
'''

#### Applying RFE Feature Selection Method automatic

# create pipeline
rfe = RFECV(estimator=DecisionTreeClassifier())
model = DecisionTreeClassifier()
pipeline = Pipeline(steps=[('s',rfe),('m',model)])

#make balanced scorer
scorer = make_scorer(balanced_accuracy_score)

# evaluate model
cv = TimeSeriesSplit(n_splits=3)
n_scores = cross_val_score(pipeline, X, y, scoring=scorer, cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('Balanced_Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

#Identify which columns were used
for i in range(X.shape[1]):
	print('Column: %d, Selected %s, Rank: %.3f' % (i, rfecv.support_[i], rfecv.ranking_[i]))
