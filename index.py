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
