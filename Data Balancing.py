
#importing libraries
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import imblearn as imbl
import scipy
import sklearn
import joblib
from imblearn.over_sampling import BorderlineSMOTE, SMOTE
from imblearn.under_sampling import NeighbourhoodCleaningRule, EditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn import metrics
from sklearn.metrics import fbeta_score
from sklearn.metrics import recall_score, roc_auc_score, brier_score_loss, confusion_matrix
from sklearn.feature_selection import RFE, RFECV
from sklearn.pipeline import Pipeline
from numpy import mean, std
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import  mutual_info_classif
from collections import Counter

#Importing data
filename = r"creditcard.csv"
df = pd.read_csv(filename)
for col in ['Class']:
    df[col] = df[col].astype('category')

#splitting into features and class as per SFS
X = df.loc[:, ['Time','V1','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14',
    'V18','V19','V23','V26','Amount']]
y = df.loc[:, 'Class']

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(random_state=1)

#KFold Cross (with time series split) Validation approach
tss = TimeSeriesSplit(n_splits = 2)
tss.split(X)

# Initialize the accuracy of the models to blank list. The accuracy of each model will be appended to this list
F_measure_model = []
roc_auc_model = []
brier_score_model = []

'''
#############################################################################
#DECISION TREE WITHOUT FEATURE SELECTION AND NO BALANCING

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
'''
###############################################################################
#BORDERLINE SMOTE (OVERSAMPLING) + DECISION TREE

for i in (0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9):
    print('sampling_strategy:', i)
    for train_index, test_index in tss.split(X):
        #split data
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        print('Original Data Shape %s' % Counter(y_train))
        #Oversample data
        sm = BorderlineSMOTE(sampling_strategy=i, random_state=1)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        print('Resampled dataset shape %s' % Counter(y_res))
        # Train the model
        model = clf.fit(X_res, y_res)
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
'''
###############################################################################
#NEIGHBOURHOOD CLEANING RULE (DISTANCE-BASED UNDERSAMPLING) + DECISION TREE

for i in (2,3,4,5):
    print("n_neighbours:", i)
    #initialize list
    F_measure_model = []
    roc_auc_model = []
    brier_score_model = []
    for train_index, test_index in tss.split(X):
        #split data
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        print('Original Data Shape %s' % Counter(y_train))
        #Undersample data
        ncr = NeighbourhoodCleaningRule(sampling_strategy='majority', n_neighbors=i)
        X_res, y_res = ncr.fit_sample(X_train, y_train)
        print('Resampled dataset shape %s' % Counter(y_res))
        # Train the model
        model = clf.fit(X_res, y_res)
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
'''
###############################################################################
#UNDER/OVERSAMPLING COMBO - SMOTE-ENN + DECISION TREE

for i in (0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9):
    print('sampling_strategy:', i)
    #initialize list
    F_measure_model = []
    roc_auc_model = []
    brier_score_model = []
    for train_index, test_index in tss.split(X):
        #split data
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        print('Original Data Shape %s' % Counter(y_train))
        #Run sampling model
        smote_enn = SMOTEENN(random_state=1, sampling_strategy=i, enn=EditedNearestNeighbours(sampling_strategy='majority'), smote=SMOTE(sampling_strategy='minority'))
        X_res, y_res = smote_enn.fit_resample(X_train, y_train)
        print('Resampled dataset shape %s' % Counter(y_res))
        # Train the model
        model = clf.fit(X_res, y_res)
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
'''
###############################################################################
#UNDER/OVERSAMPLING COMBO - SMOTE-TOMEK + DECISION TREE

for i in ("0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"):
    print('sampling_strategy:', i)
    #initialize list
    F_measure_model = []
    roc_auc_model = []
    brier_score_model = []
    for train_index, test_index in tss.split(X):
        #split data
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        print('Original Data Shape %s' % Counter(y_train))
        #Run sampling model
        smote_tomek = SMOTETomek(random_state=1, sampling_strategy=i)
        X_res, y_res = smote_tomek.fit_resample(X_train, y_train)
        print('Resampled dataset shape %s' % Counter(y_res))
        # Train the model
        model = clf.fit(X_res, y_res)
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
F_measure_model = []
roc_auc_model = []
brier_score_model = []

for train_index, test_index in tss.split(X):
    #split data
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    print('Original Data Shape %s' % Counter(y_train))
    #Run sampling model
    smote_enn = SMOTEENN(random_state=1, sampling_strategy='auto')
    X_res, y_res = smote_enn.fit_resample(X_train, y_train)
    print('Resampled dataset shape %s' % Counter(y_res))
    # Train the model
    model = clf.fit(X_res, y_res)
    # Append metrics to the list
    print(fbeta_score(y_test, model.predict(X_test),beta=2))
    print(roc_auc_score(y_test, model.predict(X_test)))
    print(brier_score_loss(y_test, model.predict(X_test)))
