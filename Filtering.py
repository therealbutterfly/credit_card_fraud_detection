#importing libraries
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

#Decision Tree Import
from sklearn.tree import DecisionTreeClassifier

# Import train_test_split function
from sklearn.model_selection import TimeSeriesSplit

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

#Importing data
filename = r"creditcard.csv"
df = pd.read_csv(filename)
for col in ['Class']:
    df[col] = df[col].astype('category')

#splitting into features and class
X = df.loc[:, 'Time':'Amount']
y = df.loc[:, 'Class']

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
    'Balanced Accurace': balanced_accuracy_model,
    'Average Precision': average_precision_model,
    'F1 Score': f1_score_model,
    'ROC_AUC': roc_auc_model
    })

print(metrics)

#Feature Selection using filtering 1 - SelecFromModel
