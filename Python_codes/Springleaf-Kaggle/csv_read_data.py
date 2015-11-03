import pandas as pd
import numpy as np

##================================
# data directory

orgdir = '../../../Documents/My_code/Springleaf-Kaggle/'
ddir = 'data'
train_filename = 'train.csv'
test_filename = 'test.csv'
filename = orgdir+ ddir+ '/'+ train_filename

#======================================
# Read the first n rows
Totalrows = 145232
n = 10000
print("Rows per batch/Total rows: %d/%d" % (n, Totalrows))
train = pd.read_csv(filename, nrows= n)
print("train: %s" % (str(train.shape)))
train.drop('ID', axis = 1, inplace = True)
print("train: %s" % (str(train.shape)))
# Drop the columns containing NA
train_cl = train.dropna(axis = 1)
print("train: %s" % (str(train_cl.shape)))

#=============================================
# convert categorical data to numerical 

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
binarizer = preprocessing.Binarizer()

def conv_2_cat(df): 
		""" convert the non numerical columns to categorical indices """
		for col in df:
			nans = df[col].isnull().sum()

			if not np.isreal(df[col][0]):
				#the 1st entry of each column
				if nans > 0: #contain non-numerical
					df[col]= df[col].fillna('Void')

				df[col] = df[col].astype(str)
				le.fit(df[col])
				df[col] = le.transform(df[col])
			else:
				if nans > 0:
					df[col] = df[col].fillna(0)
		return df

train_cl = conv_2_cat(train_cl)
print("train: %s" % (str(train_cl.shape)))

#===============================================
# split and slice
from sklearn.cross_validation import train_test_split
train1, test1 = train_test_split(train_cl, test_size = 0.1)
nTrain, nFea = train1.shape
nTest, nFeaTst = test1.shape

# Training and Testing labels and data
y_train1 = train1['target'].values
y_test1 = test1['target'].values

X_train1 = train1.drop('target', axis = 1).values
X_test1 = test1.drop('target', axis = 1).values

#X_train1 = train1
#X_test1 = test1
#y_train1 = train1.astype(int)
#y_test1 = test[:,-1].astype(int)

#==============================================
#  feature selection
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import ExtraTreesClassifier
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier


#===============================================
#   Build classifier



