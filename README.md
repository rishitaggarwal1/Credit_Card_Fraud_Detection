# Credit Card Fraud Detection

# Project Aimed:
The Credit Card Fraud Detection Problem includes modeling past credit card transactions with the knowledge of the ones that turned out to be fraud. This model is then used to identify whether a new transaction is fraudulent or not. The aim of the project is that to detect fraud transaction if any. The context of the project is to help credit card companies recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.

# Pre-Requisites:
Basic about Python3
Some of the Python Libraries
1.  numpy           "https://numpy.org/"
2.  pandas          "https://pypi.org/project/pandas/"
3.  seaborn         "https://pypi.org/project/seaborn/"
4.  sklearn         "https://pypi.org/project/sklearn/"
5.  matplotlib      "https://pypi.org/project/matplotlib/"
6.  nltk            "https://pypi.org/project/nltk/"
7.  re              "https://pypi.org/project/regex/"

# Installation:
There are some steps for installation
1.  Install Python3.            "https://www.python.org/downloads/"
2.  Install Jupyter Notebook.   "https://jupyter.org/install"
3.  Install all the libraries above mentioned by using "pip install library_name".
4.  Download the project. Run it in your system.

# Dataset:
The dataset contains transactions made by credit card including fraud and real transactions both. Features V1, V2, … V28 are the principal components obtained with PCA.'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset.'Amount' is the transaction Amount.'Class' is the response variable and it has value 1 in case of fraud and 0 when the transaction is real.

Note: The dataset is downloaded from Kaggle: "https://www.kaggle.com/mlg-ulb/creditcardfraud/".

# Algorithm:
As the data is structured and after analyzing the dataset I came to know that we can use classifier in this project. In this seaborn library that uses matplotlib is used to plot graphs and pandas is used for reading the dataset. 
I am getting 75% accuracy in this project.
