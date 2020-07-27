
#importing libraries
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn import metrics, svm
from sklearn.metrics import fbeta_score
from sklearn.metrics import recall_score, roc_auc_score, brier_score_loss, confusion_matrix
from numpy import mean, std
from imblearn.under_sampling import NeighbourhoodCleaningRule, EditedNearestNeighbours
from collections import Counter

#Importing data
filename = r"creditcard.csv"
df = pd.read_csv(filename)
for col in ['Class']:
    df[col] = df[col].astype('category')
'''
#splitting into features and class
X = df.loc[:, 'Time':'Amount']
y = df.loc[:, 'Class']
'''
#splitting into features and class as per SFS
X = df.loc[:, ['Time','V1','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14',
    'V18','V19','V23','V26','Amount']]
y = df.loc[:, 'Class']

# Initialize the accuracy of the models to blank list. The accuracy of each model will be appended to this list
F_measure_model = []
roc_auc_model = []
brier_score_model = []

'''
#############################################################################
#DECISION TREE

#create depths
#max_depths = np.linspace(1, 32, 32, endpoint=True)

#KFold Cross (with time series split) Validation approach
tss = TimeSeriesSplit(n_splits = 3)
tss.split(X)

# Iterate over each train-test split
F_measure_model = []
roc_auc_model = []
brier_score_model = []
for train_index, test_index in tss.split(X):
    # Split train-test
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    #Undersample data
    #ncr = NeighbourhoodCleaningRule(sampling_strategy='majority', n_neighbors=2)
    #X_res, y_res = ncr.fit_sample(X_train, y_train)
    # Train the model
    clf = DecisionTreeClassifier(max_depth=20,random_state=1)
    #model = clf.fit(X_res, y_res)
    model = clf.fit(X_train, y_train)
    # Append metrics to the list
    F_measure_model.append(fbeta_score(y_test, model.predict(X_test),beta=2))
    roc_auc_model.append(roc_auc_score(y_test, model.predict(X_test)))
    brier_score_model.append(brier_score_loss(y_test, model.predict(X_test)))

# Print the model metrics
metrics = pd.DataFrame(
    {'Result': ["Average"],
    'F_measure': np.average(F_measure_model),
    'ROC_AUC': np.average(roc_auc_model),
    'Brier_Score' : np.average(brier_score_model)
    })
print("Model Metrics:")
print(metrics)
'''

###############################################################################
#RANDOM FORESTS

n_estimators = [250, 300, 350]

# Create Random Forests classifer object
rf = RandomForestClassifier(random_state=1)

#KFold Cross (with time series split) Validation approach
tss = TimeSeriesSplit(n_splits = 3)
tss.split(X)

for i in n_estimators:
    print("n_estimators:",i)
    # Iterate over each train-test split
    F_measure_model = []
    roc_auc_model = []
    brier_score_model = []
    for train_index, test_index in tss.split(X):
        # Split train-test
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        #Undersample data
        ncr = NeighbourhoodCleaningRule(sampling_strategy='majority', n_neighbors=2)
        X_res, y_res = ncr.fit_sample(X_train, y_train)
        # Train the model
        rf = RandomForestClassifier(random_state=1, n_estimators=i)
        model = rf.fit(X_res, y_res)
        #model = clf.fit(X_train, y_train)
        # Append metrics to the list
        F_measure_model.append(fbeta_score(y_test, model.predict(X_test),beta=2))
        roc_auc_model.append(roc_auc_score(y_test, model.predict(X_test)))
        brier_score_model.append(brier_score_loss(y_test, model.predict(X_test)))
    # Print the model metrics
    metrics = pd.DataFrame(
        {'Result': ["Average"],
        'F_measure': np.average(F_measure_model),
        'ROC_AUC': np.average(roc_auc_model),
        'Brier_Score' : np.average(brier_score_model)
        })
    print("Model Metrics:")
    print(metrics)

'''
###############################################################################
#SVM weighted
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
# Create weighted SVM classifier object
#model_svm = svm.SVC(random_state=1)
model_svm = make_pipeline(StandardScaler(), svm.SVC(random_state=1))

#class_weight={1: 10},

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
    model = model_svm.fit(X_train, y_train)
    # Append metrics to the list
    F_measure_model.append(fbeta_score(y_test, model_svm.predict(X_test),beta=2))
    roc_auc_model.append(roc_auc_score(y_test, model_svm.predict(X_test)))
    brier_score_model.append(brier_score_loss(y_test, model_svm.predict(X_test)))

# Print the model metrics
metrics = pd.DataFrame(
    {'Result': ["Average"],
    'F_measure': np.average(F_measure_model),
    'ROC_AUC': np.average(roc_auc_model),
    'Brier_Score' : np.average(brier_score_model)
    })

print("Model Metrics:")
print(metrics)
'''
