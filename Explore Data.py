#Preparation by importing libraries

import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

plt.style.use('ggplot')
pd.set_option('display.max_columns', 500)

#Importing data as dataframe

filename = r"..\Data\creditcard.csv"
df = pd.read_csv(filename)
'''
#Check to make sure data is imported correctly
print(df.head())

#Checking data types
print(df.dtypes)

#Changing Class datatype from int64 to Category

for col in ['Class']:
    df[col] = df[col].astype('category')

#Counting number of NaN values in dataframe
print(df.isnull().sum().sum())
'''
#Basic Description of datasets

print(df.describe())
'''
# Count of each feature is the same, confirming that there are no missing values
# Max time is 1720,792 confirming that the data covers just above 2 days from first transaction recorded
# 50th percentile is 84,692 which indicates that data is generally evenly distributed across two days
# All PCs have negative and positive values -->
# Maximum amount is 25,691 and the minimum transaction amount is $0, which is suspicious. The average amount is $88
# $77 is the 75th percentile which indicates that the amount field is positively skewed, with a few high value outliers
# The average class is 0.001727which confirms that the dataset is heavily imbalanced.

#Exploring data distribution
df.hist()
plt.show()
#At first glance, all  fields seem to be unimodal, with different means

#Further exploring the data distribution of Time
df.hist(column='Time',by='Class')
plt.show()

# Bi-modal shows that there are fewer transacations at night,
#There is a more distinct bi-modality for non-fraudulent transacations
#which indicates cycles to regular transacations that fraudulent transactions may not follow

#Further exploring the data distribution of Amount

df.hist(column='Amount',by="Class")
plt.show()

# Amount looks to have a high positive skew, with a few extreme values, as
# indicated by the percentile distribution

#Futher examining the distribution of Amount by Class without points beyond the whiskers (outliers)
df.boxplot(column='Amount',by="Class",showfliers=False)
plt.show()

##############Memory error -- need to fix!
scatter_matrix(df)
plt.show()

#Creating correlation matrix rounding to 4 digits
corr = df.corr().round(4)

##############Can improve this by eliminating class variable as it is not numeric

#Printing to textfile; a is used to append and w of overwrite
print(corr, file=open("corr_output.txt","w"))

################### Need to close file!

#Visualizing correlation matrix
sn.heatmap(corr,annot=False)

plt.show()
'''
