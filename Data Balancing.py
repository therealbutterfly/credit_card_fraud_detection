
#importing libraries
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import imblearn as imbl
import scipy
import sklearn
import joblib
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import NeighbourhoodCleaningRule
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

#splitting into features and class
X = df.loc[:, ['Time','V1','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14',
    'V18','V19','V23','V26','Amount']]
y = df.loc[:, 'Class']

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(random_state=1)

#KFold Cross (with time series split) Validation approach
tss = TimeSeriesSplit(n_splits = 3)
tss.split(X)

# Initialize the accuracy of the models to blank list. The accuracy of each model will be appended to this list
F_measure_model = []
roc_auc_model = []
brier_score_model = []
confusion_matrix_model = []

'''
#############################################################################
#DECISION TREE WITHOUT BALANCING

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

for train_index, test_index in tss.split(X):
    #split data
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    print('Original Data Shape %s' % Counter(y_train))
    #Oversample data
    for i in (0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9):
        print('sampling_strategy:', i)
        sm = BorderlineSMOTE(sampling_strategy=i, random_state=1)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        print('Resampled dataset shape %s' % Counter(y_res))
        # Train the model
        model = clf.fit(X_res, y_res)
        # Append metrics to the list
        print("fbeta:", fbeta_score(y_test, model.predict(X_test),beta=2))
        print("AUC:", roc_auc_score(y_test, model.predict(X_test)))
        print("Brier:", brier_score_loss(y_test, model.predict(X_test)))

'''
## DECISION TREE WITH HIGHEST SMOTE RESULTS

# Iterate over each train-test split
for i in (0.3,0.6,0.7,0.8):
    print('sampling_strategy:', i)
    for train_index, test_index in tss.split(X):
    # Split train-test
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
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
###############################################################################
#NEIGHBOURHOOD CLEANING RULE (DISTANCE-BASED UNDERSAMPLING) + DECISION TREE

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Initialize the accuracy of the models to blank list. The accuracy of each model will be appended to this list
balanced_accuracy_model = []
average_precision_model = []
f1_score_model = []
roc_auc_model = []

#Split Data using TimeSeriesSplit
tss = TimeSeriesSplit(n_splits = 3)
tss.split(X)

for train_index, test_index in tss.split(X):
    #split data
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    print('Original Data Shape %s' % Counter(y_train))
    #Undersample data
    ncr = NeighbourhoodCleaningRule(sampling_strategy='majority')
    X_res, y_res = ncr.fit_sample(X_train, y_train)
    print('Resampled dataset shape %s' % Counter(y_res))
    # Train the model
    model = clf.fit(X_res, y_res)
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
###############################################################################
#UNDER/OVERSAMPLING COMBO - SMOTE-ENN + DECISION TREE

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Initialize the accuracy of the models to blank list. The accuracy of each model will be appended to this list
balanced_accuracy_model = []
average_precision_model = []
f1_score_model = []
roc_auc_model = []

#Split Data using TimeSeriesSplit
tss = TimeSeriesSplit(n_splits = 3)
tss.split(X)

for train_index, test_index in tss.split(X):
    #split data
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    print('Original Data Shape %s' % Counter(y_train))
    #Run sampling model
    smote_enn = SMOTEENN(random_state=1, sampling_strategy='majority')
    X_res, y_res = smote_enn.fit_resample(X_train, y_train)
    print('Resampled dataset shape %s' % Counter(y_res))
    # Train the model
    model = clf.fit(X_res, y_res)
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
###############################################################################
#UNDER/OVERSAMPLING COMBO - SMOTE-TOMEK + DECISION TREE

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Initialize the accuracy of the models to blank list. The accuracy of each model will be appended to this list
balanced_accuracy_model = []
average_precision_model = []
f1_score_model = []
roc_auc_model = []

#Split Data using TimeSeriesSplit
tss = TimeSeriesSplit(n_splits = 3)
tss.split(X)

for train_index, test_index in tss.split(X):
    #split data
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    print('Original Data Shape %s' % Counter(y_train))
    #Run sampling model
    smote_tomek = SMOTETomek(random_state=1, sampling_strategy=0.3)
    X_res, y_res = smote_tomek.fit_resample(X_train, y_train)
    print('Resampled dataset shape %s' % Counter(y_res))
    # Train the model
    model = clf.fit(X_res, y_res)
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
