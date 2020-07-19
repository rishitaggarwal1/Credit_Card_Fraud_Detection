# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import sklearn
import sys


# print('Python: {}'.format(sys.version))
# print('Python: {}'.format(numpy.__version__))
# print('Python: {}'.format(pandas.__version__))
# print('Python: {}'.format(matplotlib.__version__))
# print('Python: {}'.format(seaborn.__version__))
# print('Python: {}'.format(scipy.__version__))
# print('Python: {}'.format(sklearn.__version__))


# load the dataset from csv file
data = pd.read_csv('creditcard.csv')

# exploring the data
print(data.columns)

print(data.shape)

print(data.describe())

data = data.sample(frac=0.1, random_state=1)
print(data.shape)

# plot histogram
data.hist(figsize=(20, 20))
plt.show()

# determine the number of fraud cases
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]
frac = len(fraud)/float(len(valid))
print(frac)

print('fraud cases: {}'.format(len(fraud)))
print('Valid cases: {}'.format(len(valid)))

# corelation matrix
mat = data.corr()
fig = plt.figure(figsize=(12, 9))

sns.heatmap(mat, vmax=.8, square=True)
plt.show()

# get al columns from data frames
columns = data.columns.tolist()

# filter the columns to remove data we do not want
columns = [c for c in columns if c not in ["Class"]]

# store the variable
target = "Class"
X = data[columns]
Y = data[target]

print(X.shape)
print(Y.shape)
